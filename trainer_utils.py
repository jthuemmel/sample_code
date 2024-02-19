import json
import os
import numpy as np
import torch
import torch.distributed as dist
import csv
import wandb

#wandb

def wandb_set_startup_timeout(seconds: int):
    assert isinstance(seconds, int)
    os.environ['WANDB__SERVICE_WAIT'] = f'{seconds}'

def wandb_is_initialized():
    return wandb.run is not None

### checkpointing
class ExtendedJSONEncoder(json.JSONEncoder):
    """
    JSONEncoder subclass that serializes classes and functions as well (by their name).
    """
    def default(self, o):
        if isinstance(o, type):
            return f'<cls {o.__module__}.{o.__name__}>'
        elif callable(o):
            return f'<fn {o.__module__}.{o.__name__}>'

        try:
            return super().default(o)
        except TypeError:
            return str(o)

class ExtendedJSONDecoder(json.JSONDecoder):
    pass

### utils
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### learning rate rescaling
def scale_lr(lr, per_worker_batch_size, base_batch_size):
    lr_scaling = dist.get_world_size()
    lr_scaling *= per_worker_batch_size / base_batch_size
    return lr * lr_scaling, lr_scaling

def scale_beta1(beta1, per_worker_batch_size, base_batch_size):
    factor = dist.get_world_size() * per_worker_batch_size / base_batch_size
    return beta1**factor

def scale_beta2(beta2, per_worker_batch_size, base_batch_size):
    factor = dist.get_world_size() * per_worker_batch_size / base_batch_size
    return beta2**factor

def scale_param_group(param_group, config):
    lr_enabled = param_group['scale_lr'] if 'scale_lr' in param_group else config.scale_lr
    beta1_enabled = param_group['scale_beta1'] if 'scale_beta1' in param_group else config.scale_beta1
    beta2_enabled = param_group['scale_beta2'] if 'scale_beta2' in param_group else config.scale_beta2

    scaled_params = dict(param_group)
    if 'lr' in param_group and lr_enabled:
        scaled_params['lr'], _ = scale_lr(param_group['lr'], config.batch_size, config.base_batch_size)

    if 'betas' in param_group:
        if beta1_enabled:
            beta1 = scale_beta1(param_group['betas'][0], config.batch_size, config.base_batch_size)
        else:
            beta1 = param_group['betas'][0]
        if beta2_enabled:
            beta2 = scale_beta2(param_group['betas'][1], config.batch_size, config.base_batch_size)
        else:
            beta2 = param_group['betas'][1]
        scaled_params['betas'] = (beta1, beta2)

    if 'momentum' in param_group and beta1_enabled:
        scaled_params['momentum'] = scale_beta1(param_group['momentum'], config.batch_size, config.base_batch_size)

    return scaled_params

### Metric handling
class Metric:
    def __init__(self, name, reduction=dist.ReduceOp.AVG, allreduce=True):
        self.name = name
        self.reduction = reduction
        self.batch_values = []
        self.allreduce = allreduce
        assert not allreduce or reduction is not None, 'Cannot allreduce without reduction'

    def _reduce(self, value, dim=None):
        if value.dim() == 0:
            return value
        elif self.reduction == dist.ReduceOp.AVG:
            return value.mean(dim=dim)
        elif self.reduction == dist.ReduceOp.SUM:
            return value.sum(dim=dim)
        elif self.reduction == dist.ReduceOp.MIN:
            return value.min(dim=dim)[0]
        elif self.reduction == dist.ReduceOp.MAX:
            return value.max(dim=dim)[0]
        elif self.reduction == dist.ReduceOp.PRODUCT:
            return value.prod(dim=dim)
        else:
            raise ValueError(f'Unknown reduction {self.reduction}')

    def add_batch_value(self, value):
        tensor = torch.as_tensor(value)
        if self.reduction is not None:
            tensor = self._reduce(tensor, dim=0)
        self.batch_values.append(tensor.detach().cpu()) # this is very important to avoid memory leaks and for performance

    def reduce_locally(self):
        if self.reduction is None:
            return self.batch_values[0]
        tensor = torch.stack(self.batch_values)
        tensor = self._reduce(tensor, dim=0)
        return tensor

    def reduce(self):
        tensor = self.reduce_locally()
        if self.allreduce:
            local_rank = os.environ.get("LOCAL_RANK", "0")
            dist.all_reduce(tensor.to(f"cuda:{local_rank}"), op=self.reduction) #this is very ugly, i'm sorry
        return tensor.detach().cpu()


class MetricSaver:
    def __init__(self, epochs=None):
        self.epochs = epochs or []
        self.current_metrics = {}

    @property
    def last(self):
        return self.epochs[-1]

    def get_metrics(self, name):
        return [epoch[name] for epoch in self.epochs]

    def reduce(self):
        reduced = {}
        for name, metric in self.current_metrics.items():
            reduced[name] = metric.reduce()
        self.epochs.append(reduced)
        self.current_metrics = {}

    def log_metric(self, name, value, reduction=dist.ReduceOp.AVG, allreduce=True):
        if name not in self.current_metrics:
            self.current_metrics[name] = Metric(name, reduction, allreduce)
        metric = self.current_metrics[name]
        metric.add_batch_value(value)

    def log_python_object(self, name, value):
        self.log_metric(name, value, reduction=None, allreduce=False)

    def scalar_metrics(self, with_epoch=False):
        scalars = []
        for epoch, metrics in enumerate(self.epochs):
            dct = {}
            if with_epoch:
                dct['epoch'] = epoch + 1

            for name, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.dim() == 0:
                    dct[name] = value.item()
                elif not isinstance(value, torch.Tensor):
                    dct[name] = value
            scalars.append(dct)

        return scalars

    def scalars_to_csv(self, path):
        with open(path, 'w') as file:
            scalar_metrics = self.scalar_metrics(with_epoch=True)
            writer = csv.DictWriter(file, fieldnames=scalar_metrics[0].keys())
            writer.writeheader()
            writer.writerows(scalar_metrics)
###
