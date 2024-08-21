import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset, generate_regression_dataset_positive_coeff
from imf.experiments.architecture import build_circular_flow_l1_manifold, build_simple_circular_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_sparsity_distr, plot_cumulative_returns_singularly, plot_sparsity_patterns, plot_betas_lambda, plot_marginal_likelihood, plot_returns, plot_cumulative_returns

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=2_000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=1., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=0.1, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=2., help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=1_000, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=1, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
parser.add_argument('--beta', metavar='be', type=float, default=1.0, help='p of the lp norm')


# MODEL PARAMETERS
parser.add_argument("--n_layers", metavar='nl', type=int, default=5, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=128, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--n_context_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])

args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args=None):
    # if args.log_cond: alpha_ = 10 ** alpha
    # else: alpha_ = alpha

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha * K) - K * torch.lgamma(alpha)
    log_prior = (alpha - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior

def generate_archetypes(dim, n_samples, n_archetypes, noise=0.05, alpha=1., seed=1234):
    set_random_seeds(seed)

    archetypes = np.random.rand(n_archetypes, dim)
    coefficients = np.random.dirichlet(np.ones(n_archetypes) * alpha, size=n_samples,)

    data = coefficients @ archetypes
    data += noise * np.random.randn(n_samples, dim)

    plot_archetypes(data, archetypes)

    return data, archetypes

def plot_archetypes(data, archetypes):
    if data.shape[-1] == 2:
        plt.scatter(data[:, 0], data[:, 1], alpha=0.7)
        plt.scatter(archetypes[:, 0], archetypes[:, 1], c='red', marker='x')
        plt.title("Synthetic Dataset (2D)")
        plt.show()
    elif data.shape[-1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.7)
        ax.scatter(archetypes[:, 0], archetypes[:, 1], archetypes[:, 2], c='red', marker='x')
        ax.set_title("Synthetic Dataset (3D)")
        plt.show()

def log_likelihood(X, _W, H):
    W = torch.softmax(_W, dim=1)
    return -(X - H @ W @ X).square().sum(-1)

def unnorm_log_posterior(H: torch.Tensor, X: torch.Tensor, _W: torch.Tensor, args):
    # log_lik = log_likelihood(X=X, _W=_W, H=H)
    # alphas = torch.ones_like(beta)
    # log_prior = dirichlet_prior(beta=beta, alpha=alphas, args=args).sum(-1)
    log_prior = dirichlet_prior(beta=H, alpha=torch.ones_like(H[:,0])*0.1, args=args)
    return log_prior# log_lik #+ log_prior


from utils_manifold import gen_cooling_schedule
import time
from datetime import timedelta
def train_regression_cond(model, log_unnorm_posterior, args, **kwargs):
    optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        # {'params': [tensor], 'lr': 0.1}
    ])


    # set up cooling schedule
    num_iter = args.epochs // args.iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=args.T0, Tn=args.Tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss, loss_T = [], []
    try:
        start_time = time.monotonic()
        for epoch in range(args.epochs):
            T = cooling_function(epoch // (args.epochs / num_iter))
            optimizer.zero_grad()

            samples, log_prob = model.sample_and_log_prob(num_samples=args.n_samples)
            if torch.any(torch.isnan(samples)): breakpoint()

            log_posterior = log_unnorm_posterior(H=samples)
            kl_div = torch.mean(log_prob - log_posterior/T)
            kl_div.backward()

            optimizer.step()

            loss.append(torch.mean(log_prob - log_posterior).cpu().detach().numpy())
            loss_T.append(torch.mean(log_prob - log_posterior / T).cpu().detach().numpy())
            if epoch % 10 == 0:
                print(f"Training loss at step {epoch}: {loss[-1]:.3f} and {loss_T[-1]:.3f} * (T = {T:.3f})")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T

def train_linear_model(H_tensor, W_tensor, X):
    optimizer = torch.optim.Adam([
        {'params': [H_tensor, W_tensor], 'lr': 0.1},
    ])

    # m = torch.distributions.dirichlet.Dirichlet(torch.ones(W_tensor.shape[0]))

    loss_list = []
    try:
        start_time = time.monotonic()
        for epoch in range(args.epochs):
            optimizer.zero_grad()

            # H_tensor = torch.rand(W_tensor.shape[::-1]).to(W_tensor.device)
            H = torch.softmax(H_tensor, dim=1)
            # H = m.rsample([W_tensor.shape[1]]).to(W_tensor.device)
            W = torch.softmax(W_tensor, dim=1)
            loss = torch.square(X - H @ W @ X).sum()
            loss.backward()

            optimizer.step()

            loss_list.append(loss.cpu().detach().numpy())
            if epoch % 200 == 0:
                print(f"Training loss at step {epoch}: {loss_list[-1]:.4f}")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return loss_list


def main():
    set_random_seeds(args.seed)

    dim = 2
    n_samples = 200
    args.n_samples = n_samples
    n_archetypes = 5
    X_np, archetypes = generate_archetypes(dim=dim, n_samples=n_samples, n_archetypes=n_archetypes,
                                           noise=0.00, alpha=0.2, seed=1234)
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    args.datadim = n_archetypes


    # train simple linear model for MAP
    H_tensor = torch.nn.Parameter(torch.rand((n_samples, n_archetypes), requires_grad=True).to(device=args.device))
    W_tensor = torch.nn.Parameter(torch.rand((n_archetypes, n_samples), requires_grad=True).to(device=args.device))
    args.n_epochs = 5000
    loss = train_linear_model(H_tensor, W_tensor, X_tensor)
    plot_loss(loss)

    learnt_archetypes = torch.softmax(W_tensor, dim=1) @ X_tensor
    samples = torch.softmax(H_tensor, dim=1) @ learnt_archetypes
    plot_archetypes(samples.detach().cpu().numpy(), learnt_archetypes.detach().cpu().numpy())

    # build flow
    args.n_epochs = 1000
    flow = build_circular_flow_l1_manifold(args)
    # _W = torch.nn.Parameter(torch.rand((n_archetypes, n_samples), requires_grad=True).to(device=args.device))
    # log_unnorm_posterior = partial(unnorm_log_posterior, X=X_tensor, args=args)
    log_unnorm_posterior = partial(unnorm_log_posterior, X=X_tensor, _W=W_tensor, args=args)

    # train model
    flow.train()
    flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, tensor=W_tensor, args=args)
    plot_loss(loss)

    # evaluate model
    flow.eval()
    H, log_prob = flow.sample_and_log_prob(num_samples=n_samples)
    learnt_archetypes = torch.softmax(W_tensor, dim=1) @ X_tensor
    samples = H @ learnt_archetypes
    plot_archetypes(samples.detach().cpu().numpy(), learnt_archetypes.detach().cpu().numpy())

if __name__ == "__main__":
    main()