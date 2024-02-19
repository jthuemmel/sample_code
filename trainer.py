### Adapted from Sebastian Hoffmann - https://github.com/sehoffmann/dmlcloud ###

import os
import wandb
import numpy as np
import random
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms.join import Join
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from datetime import datetime
from pathlib import Path

from trainer_utils import *

class TrainerInterface:
    """
    These methods must be implemented for each experiment
    """

    def create_loss(self):
        """
        Returns a loss function.
        Will be available as self.loss_fn.
        """
        raise NotImplementedError()   

    def create_dataset(self):
        """
        Returns a tuple of (train_dl, val_dl).
        Will be available as self.train_dl and self.val_dl.
        These shall be iterators that yield batches.
        """
        raise NotImplementedError()

    def create_model(self):
        """
        Returns a torch.nn.Module.
        Will be available as self.model.
        If you need multiple networks, e.g. for GANs, wrap them in a nn.Module.
        """
        raise NotImplementedError()

    def create_optimizer(self, params, lr):
        """
        Returns an optimizer.
        Will be available as self.optimizer.
        """
        raise NotImplementedError()

    def create_scheduler(self):
        """
        Returns a scheduler or None.
        """
        raise NotImplementedError()
    
    def create_metrics(self):
        """
        Returns a dictionary of metrics.
        Will be available as self.metrics.
        """
        raise NotImplementedError()

    def forward_step(self, batch_idx, batch):
        """
        Performs a forward pass and returns the loss.
        """
        raise NotImplementedError()
    
class DistributedTrainer(TrainerInterface):  
     
    @staticmethod
    def root_only(fn):
        """
        Decorator for methods that should only be called on the root rank.
        """

        def wrapper(self, *args, **kwargs):
            if self.is_root:
                return fn(self, *args, **kwargs)

        return wrapper

    @property
    def is_root(self):
        return self.rank == 0

    def __init__(self, config):
        self.cfg = config
        self.reset()
    
    def setup(self):
        if self.initialized:
            raise ValueError("Trainer already initialized. Call reset() first.")

        self.setup_process_group()

        if self.cfg.use_checkpointing:
            self.setup_checkpointing()

        if self.cfg.use_wandb:
            self.setup_wandb()

        self.setup_dataset()
        if self.is_root:
            print(f"Created dataset with {len(self.train_dl)} train and {len(self.val_dl)} validation batches.", flush=True)

        self.setup_model()
        if self.is_root:
            print(f"Created model with {count_parameters(self.model)} trainable parameters.", flush=True)

        self.setup_training()
        if self.is_root:
            print(f"Created optimizer {self.optimizer}.", flush=True)
            print(f"Created scheduler {self.scheduler}.", flush=True)

        if self.ckpt_path.exists() and self.cfg.resume_training:
            self.load_checkpoint()
            self.is_resumed = True

        if self.is_root:
            print(f"Peak VRAM: {torch.cuda.max_memory_allocated(self.device) // 1e6}MB", flush = True)

        self.initialized = True

    def reset(self):
        self.current_epoch = 1
        self.is_resumed = False
        self.initialized = False
        self.train_metrics = MetricSaver()
        self.val_metrics = MetricSaver()
        self.misc_metrics = MetricSaver()
        self.mode = 'train'

    def setup_checkpointing(self):
        self.create_job_name()
        self.create_paths()
        self.create_model_dir()
        
    @root_only
    def create_model_dir(self):        
        if not self.model_dir.exists() :
            self.model_dir.mkdir(parents=True)
            self.save_config(self.cfg)
            if self.is_root:
                print(f"Created model directory {self.model_dir}", flush=True)

    def check_config(self):
        old_cfg = self.load_config_dct()
        if old_cfg != self.cfg.as_dictionary():
            raise ValueError("Config mismatch between checkpoint and current config.")

    def create_job_name(self):
        if self.cfg.name:
            self.job_name = str(self.cfg.name).replace('/', '_')
        else:
            date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.job_name = f"{date}_{self.slurm_id}"
            self.cfg.name = self.job_name
        dist.broadcast_object_list([self.job_name])

    def create_paths(self):
        self.model_dir = Path(self.cfg.model_dir) / self.job_name
        self.ckpt_path = self.model_dir / 'ckpt.pth'
        self.best_path = self.model_dir / 'best.pth'
        self.cfg_path = self.model_dir / 'config.json'

    def setup_process_group(self):
        torch.set_num_threads(8) #by default each gpu is associated with 8 cores
        self.create_env_vars()
        self.create_device()
        self.create_process_group()
        self.create_seed()

    @root_only
    def setup_wandb(self):
        wandb_set_startup_timeout(600)
        exp_name = self.cfg.wb_name if self.cfg.wb_name else self.cfg.name
        wandb.init(
            project=self.cfg.wb_project,
            name=exp_name,
            tags=self.cfg.wb_tags,
            dir=self.model_dir,
            id=self.cfg.job_id,
            resume='must' if self.is_resumed else 'never',
            config=self.cfg.as_dictionary(),
        )

    def setup_dataset(self):
        if self.is_root:
            self.train_dl, self.val_dl = self.create_dataset()
        dist.barrier()
        if not self.is_root:
            self.train_dl, self.val_dl = self.create_dataset()

    def setup_model(self):
        model = self.create_model().to(self.device)
        self.model = DistributedDataParallel(model, device_ids=[self.local_rank], output_device=self.local_rank)
        if self.cfg.use_ema:
            self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn())
        if self.cfg.use_checkpointing and self.is_root:
            with open(self.model_dir / 'model.txt', 'w') as f:
                f.write(str(model))

    def setup_training(self):       
        self.loss_fn = self.create_loss().to(self.device) 
        self.optimizer = self.create_optimizer(self.model.parameters(), self.cfg.base_lr)
        scaled_lr, _ = self.rescale_optimizer(self.optimizer)
        self.scheduler = self.create_scheduler(scaled_lr)
        self.grad_scaler = GradScaler(enabled=self.cfg.mixed_precision)

    def rescale_optimizer(self, optimizer):
        scaled_lr, lr_scale_factor = scale_lr(optimizer.defaults['lr'], self.cfg.batch_size, self.cfg.base_batch_size)
        for i, param_group in enumerate(optimizer.param_groups):
            param_group_cpy = dict(param_group)
            scaled_params = scale_param_group(param_group, self.cfg)
            scaled_params_cpy = dict(scaled_params)
            optimizer.param_groups[i] = scaled_params
            del param_group_cpy['params']
            del scaled_params_cpy['params']
        return scaled_lr, lr_scale_factor

    def create_seed(self):
        """
        Creates a shared seed for all processes.
        """
        if self.cfg.seed is None:
            seed = int.from_bytes(random.randbytes(4), byteorder='little')
            self.cfg.seed = seed
            dist.broadcast_object_list([seed])
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    def create_env_vars(self):
        """
        Sets up the process group environment variables.
        """
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12356"
        if "RANK" not in os.environ:
            os.environ["RANK"] = os.environ.get("SLURM_PROCID", "0")
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", "1")
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", "0")

        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_world_size = int(torch.cuda.device_count())
        self.slurm_id = int(os.environ.get("SLURM_JOB_ID", "0"))
        
        print(f"Hello from rank {self.rank + 1} of {self.world_size} on {socket.gethostname()} where there are" \
            f" {self.local_world_size} allocated GPUs per node.", flush=True)
    
    def create_device(self):
        if torch.cuda.is_available():
            self.device = torch.device(type = 'cuda', index = int(self.local_rank))
            torch.cuda.set_device(device = self.device)
            print(f"Rank {self.rank} set device to {torch.cuda.current_device()}", flush=True)
        else:
            self.device = torch.device(type = 'cpu')
    
    def create_process_group(self):
        if not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")
        else:
            raise ValueError("Process group already initialized. Call destroy_process_group() first.")
        if self.is_root: 
            print(f"Group initialized? {dist.is_initialized()}", flush=True)

    def destroy_process_group(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        if self.is_root: 
            print(f"Group destroyed? {not dist.is_initialized()}", flush=True)

    def switch_mode(self, train=True):
            if train:
                self.model.train()
                self.current_metrics = self.train_metrics
                self.mode = 'train'
            else:
                self.model.eval()
                self.current_metrics = self.val_metrics
                self.mode = 'eval'

    def pre_training(self):
        self.start_time = datetime.now()
        if self.is_root:
            print(f"Training started at {self.start_time}.", flush=True)

    def pre_epoch(self):
        self.epoch_start = datetime.now()
        #if the sampler has a set_epoch method, call it to ensure proper shuffling
        if hasattr(self.train_dl, 'sampler') and hasattr(self.train_dl.sampler, 'set_epoch'):
            self.train_dl.sampler.set_epoch(self.current_epoch)

    def post_epoch(self):
        #time tracking
        self.epoch_end = datetime.now()
        n_remaining = self.cfg.epochs - self.current_epoch
        per_epoch = (self.epoch_end - self.epoch_start).total_seconds()
        eta = (n_remaining * per_epoch) / 60
        #calculate time per batch
        n_train_batches = self.misc_metrics.current_metrics['n_total_batches'].reduce().item()
        per_step = 1000 * per_epoch / n_train_batches
        self.misc_metrics.log_python_object('time_per_batch[ms]', per_step)
        self.misc_metrics.log_python_object('eta[min]', eta)
        self.misc_metrics.log_python_object('time_per_epoch[s]', per_epoch)
        #metric tracking
        self.train_metrics.reduce()
        self.val_metrics.reduce()
        self.misc_metrics.reduce()
        #checkpointing
        self.save_checkpoint()
        #log_wandb
        if wandb_is_initialized():
            self.log_wandb()
        #print progress
        self.print_progress()
        #increment epoch
        self.current_epoch += 1

    def train_epoch(self):
        self.switch_mode(train=True)
        #Join context manager prevents hangups due to uneven sharding
        joinables = [self.model, self.optimizer] if self.cfg.use_zero else [self.model]
        with Join(joinables):
            for batch_idx, batch in enumerate(self.train_dl):
                self.optimizer.zero_grad()
                #automatic mixed precision
                with autocast(enabled=self.cfg.mixed_precision):
                    loss = self.forward_step(batch_idx, batch)
                #backpropagation
                self.grad_scaler.scale(loss).backward()
                #manual unscaling to enable clipping
                self.grad_scaler.unscale_(self.optimizer)
                if self.cfg.clip_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_gradients)
                #optimizer step
                self.grad_scaler.step(self.optimizer)
                if hasattr(self, 'ema_model'):
                    self.ema_model.update_parameters(self.model)
                self.grad_scaler.update()
                #logging
                self.misc_metrics.log_metric('n_steps', 1, dist.ReduceOp.SUM, allreduce=False)
                self.misc_metrics.log_metric('n_total_batches', 1, dist.ReduceOp.SUM, allreduce=True)
                #scheduler step
                if self.cfg.scheduler_step == "batch":
                    self.scheduler.step()

            if self.cfg.scheduler_step == "epoch":
                self.scheduler.step()

            if self.cfg.use_zero:
                self.optimizer.consolidate_state_dict()

    def evaluate_epoch(self):
        self.switch_mode(train=False)
        if self.val_dl is None:
            return
        #Join context manager prevents hangups due to uneven sharding
        with Join([self.model]):
            for batch_idx, batch in enumerate(self.val_dl):
                #no gradients needed for evaluation
                with torch.no_grad():
                    _ = self.forward_step(batch_idx, batch)
                
    def post_training(self):
        self.end_time = datetime.now()
        if self.is_root:
            print(f"Training started at {self.start_time} and ended at {self.end_time}", flush=True)
            print(f"Training took {self.end_time - self.start_time}", flush=True)
    
    def train(self):
        if not self.initialized:
            self.setup()
        self.pre_training() 
        while self.current_epoch <= self.cfg.epochs:
            self.pre_epoch()
            self.train_epoch() 
            self.evaluate_epoch() 
            self.post_epoch() 
        self.post_training() 

    def log_wandb(self):
            metrics = {}
            for key, value in self.train_metrics.scalar_metrics()[-1].items():
                metrics[f'train/{key}'] = value

            for key, value in self.val_metrics.scalar_metrics()[-1].items():
                metrics[f'val/{key}'] = value

            for key, value in self.misc_metrics.scalar_metrics()[-1].items():
                metrics[f'misc/{key}'] = value

            wandb.log(metrics)
            if self.is_best_epoch():
                wandb.run.summary['best/epoch'] = self.current_epoch
                for key, value in metrics.items():
                    if not key.startswith('misc'):
                        wandb.run.summary[f'best/{key}'] = value

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['epoch']
        self.train_metrics = MetricSaver(state_dict['train_metrics'])
        self.val_metrics = MetricSaver(state_dict['val_metrics'])
        self.misc_metrics = MetricSaver(state_dict['misc_metrics'])
        self.model.load_state_dict(state_dict['model_state'])
        self.optimizer.load_state_dict(state_dict['optimizer_state'])
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.grad_scaler.load_state_dict(state_dict['scaler_state'])
        if hasattr(self, 'ema_model'):
            self.ema_model.load_state_dict(state_dict['ema_model_state'])

    def state_dict(self):
        state_dict = {
            'epoch': self.current_epoch,
            'train_metrics': self.train_metrics.epochs,
            'val_metrics': self.val_metrics.epochs,
            'misc_metrics': self.misc_metrics.epochs,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'scaler_state': self.grad_scaler.state_dict(),
        }
        if hasattr(self, 'ema_model'):
            state_dict['ema_model_state'] = self.ema_model.state_dict()
        return state_dict
    
    def load_checkpoint(self):
        if not self.cfg.use_checkpointing :
            return
        if not self.ckpt_path.exists():
            return
        if self.is_root:
            self.check_config()
            print(f"Loading checkpoint from {self.ckpt_path}", flush=True)
        dist.barrier() #make sure that all processes load the same checkpoint
        state_dict = torch.load(self.ckpt_path, map_location=self.device)
        self.load_state_dict(state_dict)
        self.current_epoch += 1

    def load_best_checkpoint(self):
        if not self.cfg.use_checkpointing :
            return
        if not self.best_path.exists():
            return
        if self.is_root:
            print(f"Loading best checkpoint from {self.best_path}", flush=True)
        dist.barrier()
        state_dict = torch.load(self.best_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save_checkpoint(self):
        if not (self.cfg.use_checkpointing and self.is_root):
            return

        torch.save(self.state_dict(), self.ckpt_path)
        if self.is_best_epoch():
            torch.save(self.state_dict(), self.best_path)
            if wandb_is_initialized():
                wandb.save(str(self.best_path), policy='now', base_path=str(self.model_dir))

        self.train_metrics.scalars_to_csv(self.model_dir / 'train_metrics.csv')
        self.val_metrics.scalars_to_csv(self.model_dir / 'val_metrics.csv')
        self.misc_metrics.scalars_to_csv(self.model_dir / 'misc_metrics.csv')

    @root_only
    def save_config(self, config):
        with open(self.cfg_path, 'w') as file:
            json.dump(config.as_dictionary(), file, cls=ExtendedJSONEncoder, indent=4)

    def load_config_dct(self):
        with open(self.cfg_path) as file:
            dct = json.load(file, cls=ExtendedJSONDecoder)
            return dct

    def is_best_epoch(self):
        best_val_loss = min(self.val_metrics.get_metrics(self.cfg.val_loss_name))
        return self.val_metrics.last[self.cfg.val_loss_name] == best_val_loss
          
    @root_only
    def print_progress(self):
        stats = self.misc_metrics.last
        print(f"Epoch {self.current_epoch}/{self.cfg.epochs} | Time per epoch [min]: {stats['time_per_epoch[s]'] / 60} ETA [min]: {stats['eta[min]']:.2f}", flush=True)
        print(f"Train Loss: {self.train_metrics.last['loss']:.2e} | Val Loss: {self.val_metrics.last['loss']:.2e}", flush=True)
        if self.cfg.test_loss_name in self.val_metrics.last:
            print(f"{self.cfg.test_loss_name}: {self.val_metrics.last[self.cfg.test_loss_name]:.2e}", flush=True)
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated(self.device) // 1e6}MB | ms per batch: {stats['time_per_batch[ms]']}", flush = True)
