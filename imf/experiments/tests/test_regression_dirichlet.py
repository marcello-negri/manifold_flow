import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from imf.experiments.datasets import generate_regression_dataset_positive_coeff


def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def gaussian_log_likelihood(beta: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):
    # implements Gaussian log-likelihood beta ~ Normal (X@beta, sigma^2 ID)
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def lp_norm_prior(beta: torch.Tensor, cond: torch.Tensor, args):
    if args.log_cond: lamb_ = 10 ** cond
    else: lamb_ = cond

    p = 0.5

    log_const = torch.log(lamb_) / p + np.log(0.5 * p) - sp.special.loggamma(1./p)
    log_prior = - lamb_ * (torch.linalg.vector_norm(beta, ord=p, dim=-1)**p)
    log_prior_lp = log_prior + beta.shape[-1] * log_const

    return log_prior_lp

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, log_cond=True):
    if log_cond: alpha_ = 10 ** alpha
    else: alpha_ = alpha
    # dim = beta.shape[-1]
    # alpha_ = torch.ones_like(alpha) * 100

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1).unsqueeze(-1)

    return log_const + log_prior

def dirichlet_prior_beta(beta: torch.Tensor):

    dim = beta.shape[-1]
    alpha = torch.ones((1, dim), device= beta.device) * 10
    # alpha [:,:5] = 0.0001

    log_const = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))
    # log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = ((alpha - 1) * torch.log(beta)).sum(-1)

    return log_const + log_prior

def unnorm_log_posterior(beta: torch.Tensor, cond: torch.Tensor, sigma:torch.Tensor, X: torch.Tensor, y: torch.Tensor, log_cond=True, flow_prior=None):
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigma, X=X, y=y)
    log_prior = dirichlet_prior(beta=beta, alpha=cond, log_cond=log_cond)
    # log_prior = dirichlet_prior_beta(beta=beta)

    return log_lik + log_prior


class Network(nn.Module):
    def __init__(self, input_size, output_size, num_layers):
        super(Network, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_size, output_size))
            input_size = output_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.sigmoid(x)

        x = F.softmax(x)
        return x


def main():
    torch.manual_seed(1234)
    np.random.seed(1234)
    device='cuda'

    n_features = 10
    log_cond = True # whether alpha should be learnt on a logarithmic scale or not
    network = Network(input_size=1, output_size=n_features, num_layers=5).to(device)

    X_np, y_np, true_beta = generate_regression_dataset_positive_coeff(n_samples=100, n_features=n_features, n_non_zero=5, noise_std=1)
    X_tensor = torch.from_numpy(X_np).float().to(device=device)
    y_tensor = torch.from_numpy(y_np).float().to(device=device)

    sigma = torch.tensor(.7, device=device)
    log_unnorm_posterior = partial(unnorm_log_posterior, sigma=sigma, X=X_tensor, y=y_tensor, log_cond=log_cond)

    # build model
    network.train()
    n_epochs = 3000
    n_context_samples = 10000
    cond_min, cond_max = -2, 2
    opt = torch.optim.Adam(network.parameters(), lr=1e-3)

    loss_list = []
    for epoch in range(n_epochs):
        opt.zero_grad()
        rand_cond = torch.rand(n_context_samples, device=device)
        uniform_cond = (rand_cond * (cond_max - cond_min) + cond_min).view(-1, 1)
        # uniform_cond = 10 ** uniform_cond # if you choose not to use the logarithmic scale, see above
        coeff = network(uniform_cond)

        loss = -log_unnorm_posterior(beta=coeff, cond=uniform_cond).mean()

        loss_list.append(loss.item())
        print(f"loss at epoch {epoch}: {loss.item():.3f}")

        loss.backward()
        opt.step()

    network.eval()
    alpha_grid = torch.linspace(start=cond_min, end=cond_max, steps=1000, device=device).reshape(-1,1)
    coeff = network(alpha_grid)

    plt.plot(list(range(n_epochs)), loss_list)
    plt.show()

    alpha_grid_np = alpha_grid.detach().cpu().numpy()
    coeff_np = coeff.detach().cpu().numpy()
    plt.plot(alpha_grid_np, coeff_np)
    plt.show()
    breakpoint()


if __name__ == "__main__":
    main()
