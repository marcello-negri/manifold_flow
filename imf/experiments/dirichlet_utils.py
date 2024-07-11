
import numpy as np
import torch

from portfolio_selection import dirichlet_prior


'''
alpha is a tensor [...,n], the output will be [...,n, samples,dim]
'''
def get_dirichlet_samples(alpha, samples, dim):
    shape = alpha.shape
    alpha = alpha.view(-1)
    m = torch.distributions.Gamma(alpha, torch.tensor([1.0], device=alpha.device, dtype=alpha.dtype))
    s = m.sample((samples,dim)).permute(2,0,1)
    s = s / s.sum(dim=-1, keepdim=True)
    return s.reshape(*shape, samples, dim)


'''
alpha is a tensor [c,n], x is a tensor [n, b, d], the output will be [c ,n, b]
'''
def get_logp_dirichlet(alpha, x):
    log_prob = dirichlet_prior(x[None,], alpha[...,None], Struct({'log_cond':False}))
    return log_prob


class Struct:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
