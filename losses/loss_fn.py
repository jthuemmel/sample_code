import torch as th
from torch import nn
import numpy as np

class NormalCRPS(nn.Module):
    '''
    Continuous Ranked Probability Score (CRPS) loss for a normal distribution as described in the paper "Probabilistic Forecasting with Gated Neural Networks"
    '''
    def __init__(self, reduction = 'mean', sigma_transform = 'softplus'):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the std estimate, can be 'softplus', 'exp' or 'none'
        '''
        super().__init__()
        self.sqrtPi = th.as_tensor(np.pi).sqrt()
        self.sqrtTwo = th.as_tensor(2.).sqrt()

        if sigma_transform == 'softplus':
            self.sigma_transform = lambda x: nn.functional.softplus(x)
        elif sigma_transform == 'exp':
            self.sigma_transform = lambda x: th.exp(x)
        elif sigma_transform == 'none':
            self.sigma_transform = lambda x: x
        else:
            raise NotImplementedError(f'Sigma transform {sigma_transform} not implemented')

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')

    def forward(self, observation: th.Tensor, mu: th.Tensor, sigma: th.Tensor):
        '''
        Compute the CRPS for a normal distribution
            :param observation: (batch, *) tensor of observations
            :param mu: (batch, *) tensor of means
            :param log_sigma: (batch, *) tensor of log standard deviations
            :return: CRPS score     
            '''
        std = self.sigma_transform(sigma) #ensure positivity
        z = (observation - mu) / std #z transform
        phi = th.exp(-z ** 2 / 2).div(self.sqrtTwo * self.sqrtPi) #standard normal pdf
        score = std * (z * th.erf(z / self.sqrtTwo) + 2 * phi - 1 / self.sqrtPi) #crps as per Gneiting et al 2005
        reduced_score = self.reduce(score)
        return reduced_score
    
class Beta_NLL(nn.Module):
    '''
    Beta Negative Log Likelihood loss as described in the paper "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
    '''
    def __init__(self, beta: float = 0.5, reduction: str = 'mean', sigma_transform: str = 'softplus') -> None:
        super().__init__()
        '''
        beta: the beta parameter that controls the tradeoff between the mean and the variance of the predictive distribution
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        sigma_transform: the transform to apply to the variance estimate, can be 'softplus', 'exp' or 'none'
        '''
        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')
        
        if sigma_transform == 'softplus':
            self.sigma_transform = lambda x: nn.functional.softplus(x)
        elif sigma_transform == 'exp':
            self.sigma_transform = lambda x: th.exp(x)
        elif sigma_transform == 'none':
            self.sigma_transform = lambda x: x
        else:
            raise NotImplementedError(f'Sigma transform {sigma_transform} not implemented')
        
        self.beta = beta
        
    def forward(self, observation: th.Tensor, mu: th.Tensor, sigma: th.Tensor):
        '''
        Calculates the beta nll as described in the paper "On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks"
        :param observation: the observation
        :param mu: the mean of the predictive distribution
        :param variance: the variance of the predictive distribution
        '''
        variance = self.sigma_transform(sigma)
        loss = 0.5 * (((observation - mu) ** 2) / variance + th.log(variance))
        if self.beta > 0:
            loss = loss * (variance.detach() ** self.beta)
        return self.reduce(loss)
    
class StatisticalLoss(nn.Module):
    '''
    Statistical loss function as defined in 'AtmoRep: A stochastic model of atmosphere dynamics using large scale representation learning'
    '''
    def __init__(self, reduction = 'mean', ensemble_dim = -1):
        '''
        reduction: the reduction method to use, can be 'mean', 'sum' or 'none'
        '''
        super().__init__()

        if reduction == 'mean':
            self.reduce = lambda x: x.mean()
        elif reduction == 'sum':
            self.reduce = lambda x: x.sum()
        elif reduction == 'none':
            self.reduce = lambda x: x
        else:
            raise NotImplementedError(f'Reduction {reduction} not implemented')
        
        self.ensemble_dim = ensemble_dim

    def forward(self, observation: th.Tensor, prediction: th.Tensor):
        '''
        Compute the first order statistical loss from ensemble predictions
            :param observation: (batch, *) tensor of observations
            :param prediction: (batch, *, ensemble) tensor of ensemble predictions
            :return: CRPS score     
            '''
        #calculate first order ensemble statistics
        mu = prediction.mean(dim = self.ensemble_dim)
        sigma = prediction.std(dim = self.ensemble_dim)
        #calculate unnormalized Gaussian likelihood
        phi = th.exp(((mu - observation) / sigma).pow(2).div(2))
        #calculate squared distance between the Gaussian and the Dirac likelihood
        stat_dist = (1 - phi).pow(2)
        #calculate squared distance between each ensemble member and the observation
        member_dist = (prediction - observation.unsqueeze(-1)).pow(2).sum(-1)
        #regularization term controling the variance
        var_regularization = sigma.sqrt()
        #total score is the sum of the three terms
        score = stat_dist + member_dist + var_regularization
        #apply reduction
        reduced_score = self.reduce(score)
        return reduced_score