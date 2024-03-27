import torch
import matplotlib.pyplot as plt
import numpy as np

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor):
    eps = 1e-10

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha * K) - K * torch.lgamma(alpha)
    log_prior = (alpha - 1) * torch.log(beta+eps).sum(-1)

    return log_const + log_prior

def linear_interpolation(start, end, n):
    step_size = (end - start )/ (n - 1)
    intermediates = [start + step_size * i for i in range(0, n)]

    return torch.cat(intermediates).reshape(n, start.shape[-1])

n = 1000

beta_a = torch.tensor([1,0,0,0,0])
beta_b = torch.tensor([0,1,0,0,0])
betas = linear_interpolation(beta_a, beta_b, n=n)
alphas = torch.ones(betas.shape[0]) * .0001
log_p = dirichlet_prior(betas, alphas)

plt.plot(range(n), (log_p).detach().numpy(), marker='.')
plt.show()