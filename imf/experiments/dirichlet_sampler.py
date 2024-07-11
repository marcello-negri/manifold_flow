
import os

import numpy as np
import torch

from portfolio_selection import dirichlet_prior
from dirichlet_utils import get_dirichlet_samples, get_logp_dirichlet

from abc import ABC, abstractmethod


class Markov_dirichlet(ABC):
    def __init__(self, init_x, proposal_alphas, step_sizes):
        self.cur = init_x  # (b, n)
        self.device = init_x.device
        self.dtype = init_x.dtype
        self.curp = None
        self.chains = init_x.shape[0]
        self.dim = init_x.shape[1]
        grid_alpha, grid_steps = torch.meshgrid(proposal_alphas, step_sizes)
        self.alphas = proposal_alphas
        self.step_sizes = step_sizes
        self.grid_alpha = grid_alpha
        self.grid_steps = grid_steps
        self.average = torch.ones((self.dim,), device=self.device, dtype=self.dtype) / self.dim

    def propose(self):
        proposed_grid = get_dirichlet_samples(self.alphas, self.chains, self.dim)
        step_indices = torch.randint(0, self.step_sizes.shape[0], (self.chains,), device=self.device)
        step = self.step_sizes[step_indices]
        alpha_indices = torch.randint(0, self.alphas.shape[0], (self.chains,), device=self.device)
        proposed = proposed_grid[alpha_indices, torch.arange(self.chains)]
        proposed_scaled = self.cur + step[:,None] * (proposed - self.cur)
        return proposed_scaled

    def eps_border(self, x, eps=1e-13):
        maskleq = x < 1.0-eps
        maskbeq = x > 0.0+eps
        x = torch.where(maskleq, x, 1.0-eps)
        x = torch.where(maskbeq, x, 0.0+eps)
        return x

    def transition(self, tox, fromx):
        tox = self.eps_border(tox)
        fromx = self.eps_border(fromx)
        real_prop = fromx + (tox - fromx) / self.step_sizes[:,None,None]
        mask = torch.logical_and(torch.all(real_prop < 1.0, dim=-1), torch.all(real_prop > 0.0, dim=-1))
        adjust_prop = torch.where(mask[...,None], real_prop, self.average)
        logps_unnorm = get_logp_dirichlet(alpha=self.grid_alpha, x=adjust_prop)
        logps = logps_unnorm - self.dim * torch.log(self.step_sizes[None,...,None])
        exp = torch.where(mask, torch.exp(logps), 0.0)
        return torch.log(exp.sum((0,1)))

    @abstractmethod
    def log_p(self, x):
        pass

    def accept(self, proposed):
        if self.curp is None:
            self.curp = self.log_p(self.cur)
        prop = self.log_p(proposed)
        t = self.transition(proposed, self.cur) - self.transition(self.cur, proposed)
        a = prop - self.curp - t
        mask = torch.logical_or(a > 0.0, torch.rand(a.shape, device=self.device, dtype=self.dtype) < torch.exp(a))
        self.cur = torch.where(mask[...,None], proposed, self.cur)
        self.curp = torch.where(mask, prop, self.curp)

    def iterator(self):
        while True:
            yield self.cur, self.curp
            self.accept(self.propose())




from torch.distributions.multivariate_normal import MultivariateNormal
from portfolio_selection import dirichlet_prior
from dirichlet_utils import Struct

class Test(Markov_dirichlet):

    def __init__(self, init_x, proposal_alphas, step_sizes):
        super().__init__(init_x, proposal_alphas, step_sizes)
        loc = torch.zeros(self.dim, device=self.device, dtype=self.dtype)
        loc[0] = 1.0
        self.dist_p = MultivariateNormal(loc, covariance_matrix=0.1 * torch.diag(torch.ones(self.dim, device=self.device, dtype=self.dtype)))
        self.alpha = torch.tensor([0.01], device=self.device, dtype=self.dtype)
        self.struct = Struct({'log_cond':False})

    def log_p(self, x):
        #p = self.dist_p.log_prob(x)
        p = dirichlet_prior(x, self.alpha, self.struct)
        return p


device = "cuda"
dtype = torch.float64
burnin = 10000
chains = 10000
dim = 3
num_samples = 2000
subsample = 200
alphas_proposal = [0.01, 0.5]#[0.1,1.0,10.0]
steps_proposal = [0.1,0.2,1.0]#[0.1,0.2,0.5,1.0]

init = torch.ones((chains,dim), device=device, dtype=dtype)
init = init / torch.norm(init, dim=-1, p=1, keepdim=True)
alphas = torch.tensor(alphas_proposal, device=device, dtype=dtype)
steps_sizes = torch.tensor(steps_proposal, device=device, dtype=dtype)
test = Test(init, alphas, steps_sizes)

it = test.iterator()
for i in range(burnin):
    next(it)
    if i % 100 == 0:
        print("burnin", i)

# here all the samples are stored on the cpu
samples = torch.zeros((num_samples//subsample, chains, dim), dtype=dtype)
startt = os.times()
for i in range(num_samples):
    if i % subsample == 0:
        samples[i//subsample] = next(it)[0].detach().cpu()
    else:
        next(it)
    if i % 100 == 0:
        print("iteration ", i)
endt = os.times()
dt = endt.user + endt.system - startt.user - startt.system
print("elapsed time ", dt)

from imf.experiments.plots import plot_dirichlet3dproj

samples = samples.view((-1, dim))
if dim == 3:
    plot_dirichlet3dproj(samples.numpy())

#samplesdirect = get_dirichlet_samples(test.alpha, 5000, 3)  # has issues for small alphas -> if all 0 then normalization is terrible
#plot_dirichlet3dproj(samplesdirect.view(-1,3).detach().cpu().numpy())


print("done")


