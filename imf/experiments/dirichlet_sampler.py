
import os

import numpy as np
import torch

from portfolio_selection import dirichlet_prior
from dirichlet_utils import get_dirichlet_samples, get_logp_dirichlet

from abc import ABC, abstractmethod


class Markov_dirichlet(ABC):
    def __init__(self, init_x, proposal_alphas, step_sizes):
        self.accrate = 0.0
        self.accrate_gamma = 0.98
        self.cur = init_x  # (b, n)
        self.device = init_x.device
        self.dtype = init_x.dtype
        self.curp = None
        self.chains = init_x.shape[0]
        self.dim = init_x.shape[1]
        grid_alpha, grid_steps = torch.meshgrid(proposal_alphas, step_sizes, indexing='ij')
        self.alphas = proposal_alphas
        self.step_sizes = step_sizes
        self.grid_alpha = grid_alpha
        self.grid_steps = grid_steps
        self.minf = torch.tensor(float('-inf'), device=self.device, dtype=self.dtype)
        #self.average = torch.ones((self.dim,), device=self.device, dtype=self.dtype) / self.dim

    def propose(self):
        proposed_grid = get_dirichlet_samples(self.alphas, self.chains, self.dim)
        step_indices = torch.randint(0, self.step_sizes.shape[0], (self.chains,), device=self.device)
        step = self.step_sizes[step_indices]
        alpha_indices = torch.randint(0, self.alphas.shape[0], (self.chains,), device=self.device)
        proposed = proposed_grid[alpha_indices, torch.arange(self.chains)]
        proposed_scaled = self.cur + step[:,None] * (proposed - self.cur)
        return proposed_scaled

    def eps_border(self, x, eps=1e-13):
        return torch.clamp(x, min=0.0+eps, max=1.0-eps)

    def transition(self, tox, fromx):
        tox = self.eps_border(tox)
        fromx = self.eps_border(fromx)
        real_prop = fromx + (tox - fromx) / self.step_sizes[:,None,None]
        mask = torch.logical_and(torch.all(real_prop < 1.0, dim=-1), torch.all(real_prop > 0.0, dim=-1))
        adjust_prop = torch.where(mask[...,None], real_prop, fromx)
        logps_unnorm = get_logp_dirichlet(alpha=self.grid_alpha, x=adjust_prop)[0]
        logps = logps_unnorm - self.dim * torch.log(self.step_sizes[...,None])
        masked_logps = torch.where(mask, logps, self.minf)
        return torch.logsumexp(masked_logps, dim=0)

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
        self.accrate = self.accrate_gamma * self.accrate + (1-self.accrate_gamma) * mask.to(dtype=torch.int32).sum() / mask.numel()
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

    def __init__(self, init_x, proposal_alphas, step_sizes, seed=1234):
        super().__init__(init_x, proposal_alphas, step_sizes)
        self.prior_alpha = 1.0  # TODO adjust
        self.alpha = torch.tensor([self.prior_alpha], device=self.device, dtype=self.dtype)
        self.struct = Struct({'log_cond':False})
        self.true_sigma = 0.090  # TODO adjust
        self.my_sigma = 0.090  # TODO adjust
        self.n = 20  # TODO adjust
        self.x, self.y = self.generate_regression_simplex(self.n, self.dim, sigma=self.true_sigma, seed=seed)

    def generate_regression_simplex(self, n, d, sigma, seed):
        np.random.seed(seed)
        X_np = np.random.randn(n, d)
        beta = np.random.rand(d)
        beta /= beta.sum()
        y_np = X_np @ beta + np.random.randn(n) * sigma

        return torch.tensor(X_np, device=self.device, dtype=self.dtype), torch.tensor(y_np, device=self.device, dtype=self.dtype)

    def gaussian_log_likelihood(self, beta: torch.Tensor, X: torch.Tensor, y: torch.Tensor):
        eps = 1e-11
        log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (self.my_sigma ** 2 + eps)
        log_lk_const = - X.shape[0] * np.log((self.my_sigma + eps) * np.sqrt(2. * np.pi))

        return log_lk + log_lk_const

    def log_p(self, x):
        p = self.gaussian_log_likelihood(x, self.x, self.y) + dirichlet_prior(x, self.alpha, self.struct)
        return p

class Markov_dirichlet_given_logp(Markov_dirichlet):

    def __init__(self, init_x, proposal_alphas, step_sizes, log_p, seed=1234):
        super().__init__(init_x, proposal_alphas, step_sizes)
        # self.prior_alpha = 1.0  # TODO adjust
        # self.alpha = torch.tensor([self.prior_alpha], device=self.device, dtype=self.dtype)
        # self.struct = Struct({'log_cond':False})
        self.n = 20  # TODO adjust
        self.log_p = log_p

    def log_p(self, x):
        p = self.log_p(x)#+ dirichlet_prior(x, self.alpha, self.struct)
        return p

# device = "cuda"
# dtype = torch.float64
# burnin = 10000
# chains = 10000
# dim = 50  # TODO adjust
# iterations = 20000
# subsample = 2000
# alphas_proposal = [0.1, 0.5, 1.0]
# # important to have 1.0 in the steps for successful sampling
# steps_proposal = [0.1,0.2,0.5,1.0]
# init_alpha = torch.tensor([1.0], device=device, dtype=dtype)  # TODO adjust
#
# init = get_dirichlet_samples(init_alpha, chains, dim)[0]
# alphas = torch.tensor(alphas_proposal, device=device, dtype=dtype)
# steps_sizes = torch.tensor(steps_proposal, device=device, dtype=dtype)
# test = Test(init, alphas, steps_sizes)
#
# it = test.iterator()
# startt = os.times()
# for i in range(burnin):
#     next(it)
#     if i % 500 == 0:
#         print("burnin", i)
#
# # here all the samples are stored on the cpu
# samples = torch.zeros((iterations // subsample, chains, dim), dtype=dtype)
# for i in range(iterations):
#     if i % subsample == 0:
#         samples[i//subsample] = next(it)[0].detach().cpu()
#     else:
#         next(it)
#     if i % 500 == 0:
#         print("iteration ", i)
# endt = os.times()
# dt = endt.user + endt.system - startt.user - startt.system
# print(f"elapsed time for {(burnin + iterations) * chains} total samples taken over {burnin + iterations} iterations in total:", dt)
#
# from imf.experiments.plots import plot_dirichlet_proj
#
# samples = samples.view((-1, dim))
# print(f"effectively created {samples.shape[0]} total samples at subsampling of {subsample}")
# plot_dirichlet_proj(samples.numpy())
#
# #samplesdirect = get_dirichlet_samples(test.alpha, 5000, 3)  # has issues for small alphas -> if all 0 then normalization is terrible
# #plot_dirichlet3dproj(samplesdirect.view(-1,3).detach().cpu().numpy())
#
#
# print("done")


