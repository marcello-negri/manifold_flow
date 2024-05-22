import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset, generate_regression_dataset_positive_coeff
from imf.experiments.architecture import build_cond_flow_reverse, build_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
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
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
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

def gaussian_log_likelihood(beta: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):
    # implements Gaussian log-likelihood beta ~ Normal (X@beta, sigma^2 ID)
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def lp_norm_prior(beta: torch.Tensor, cond: torch.Tensor, args):
    if args.log_cond: lamb_ = 10 ** cond
    else: lamb_ = cond

    p = 0.1

    log_const = torch.log(lamb_) / p + np.log(0.5 * p) - sp.special.loggamma(1./p)
    log_prior = - lamb_ * (torch.linalg.vector_norm(beta, ord=p, dim=-1)**p)
    log_prior_lp = log_prior + beta.shape[-1] * log_const

    return log_prior + log_prior_lp

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args):
    if args.log_cond: alpha_ = 10 ** alpha
    else: alpha_ = alpha
    # dim = beta.shape[-1]
    # alpha_ = torch.ones_like(alpha) * 100

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior


def unnorm_log_posterior(beta: torch.Tensor, prior_name: str, cond: torch.Tensor, sigma:torch.Tensor, X: torch.Tensor, y: torch.Tensor, args, flow_prior=None):
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigma, X=X, y=y)
    # log_lik = gaussian_log_likelihood_mean(beta=beta, sigma=sigma, X=X, mu=y)
    # log_prior = laplace_prior(beta=beta, lamb=cond, argss=args)
    if prior_name == "lp_norm":
        log_prior = lp_norm_prior(beta=beta, cond=cond, args=args)
    elif prior_name == "dirichlet":
        log_prior = dirichlet_prior(beta=beta, alpha=cond, args=args)
    elif prior_name == "uniform":
        log_prior = dirichlet_prior(beta=beta, alpha=torch.zeros_like(cond), args=args)
    else:
        raise ValueError(f"Prior {prior_name} not recognized")
    # log_prior = dirichlet_prior_beta(beta=beta)
    # log_prior = lp_norm_prior_on_manifold(beta=beta, lamb=cond, flow=flow_prior, args=args)
    # breakpoint()
    return log_lik + log_prior

def t_student_log_likelihood(beta: torch.Tensor, a0: torch.Tensor, b0: torch.Tensor, X: torch.Tensor, y: torch.Tensor):

    N = X.shape[0]
    log_lk = -(a0 + 0.5 * N) * torch.log(1 + 0.5 * (y - beta @ X.T).square().sum(-1) / b0 )
    log_lk_const = torch.lgamma(a0 + 0.5 * N) - torch.lgamma(a0) - 0.5 * N * torch.log(2 * np.pi * b0)

    return log_lk + log_lk_const

def compute_norm_generalized_gaussian(args):
    args.log_cond = True
    set_random_seeds(args.seed)
    flow = build_simple_cond_flow_l1_manifold(args, n_layers=3, n_hidden_features=64, n_context_features=64, clamp_theta=False)
    model_name = f"/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/models/generalized_gaussian_dim{args.datadim}_p{args.beta}_lmin{args.cond_min}_lmax{args.cond_max}"
    lp_norm_prior_ = partial(lp_norm_prior, args=args)
    if not os.path.isfile(model_name):
        flow, loss, loss_T = train_regression_cond(flow, lp_norm_prior_, args=args, manifold=False)
        torch.save(flow.state_dict(), model_name)
        plot_loss(loss)
    else:
        flow.load_state_dict(torch.load(model_name))

    return flow

def lp_norm_prior_on_manifold(beta: torch.Tensor, lamb: torch.Tensor, flow, args):
    if args.log_cond: lamb_ = 10 ** lamb
    else: lamb_ = lamb

    dim = beta.shape[-1]
    with torch.no_grad():
        log_prob = flow.log_prob(beta.reshape(-1, dim), context=lamb_)
    return log_prob

def load_returns_dataset(stock_to_replicate=0, timesteps=-1, n_stocks_portfolio=-1):
    # the dataset contains returns of 99 stocks expressed in relative terms r_i = (p_i - p_i-1)/p_i-1
    df = pd.read_csv("./ret_rf.csv")
    df = df.dropna(axis=1) # timesteps x stocks
    dates = df.iloc[:,0].values
    df_np = df.iloc[:,1:n_stocks_portfolio].values.T # stocks x timesteps
    df_np = df_np + 1 # convert relative returns to price ratios i.e. r'_i = p_i/p_i-1
    df_np = np.c_[np.ones((df_np.shape[0], 1)), df_np]

    cum_return = np.cumprod(df_np.T, axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(cum_return)
    axs[1].plot(cum_return.mean(1), label="average stock return")
    axs[1].plot(cum_return[:, stock_to_replicate], linestyle='--', color='r', label="stock to replicate")
    plt.legend()
    plt.show()

    cum_return = np.cumprod(df_np.T[:timesteps], axis=0)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(cum_return)
    axs[1].plot(cum_return.mean(1), label="average stock return")
    axs[1].plot(cum_return[:, stock_to_replicate], linestyle='--', color='r', label="stock to replicate")
    plt.show()

    X_np = df_np[np.arange(df_np.shape[0])!=stock_to_replicate, :timesteps].T
    y_np = df_np[stock_to_replicate, :timesteps].reshape(1,-1)

    return dates[:timesteps], X_np, y_np

def main():
    just_load = True # False: train model and save samples. True: load the samples
    args.log_cond = True
    prior_name = "dirichlet" # independent of args.cond_min and args.cond_max
    # prior_name = "uniform" # argscond_min=-2 args.cond_max = 0

    # set random seed for reproducibility
    set_random_seeds(args.seed)

    # load data
    dates, X_np, y_np = load_returns_dataset(stock_to_replicate=7, timesteps=-353, n_stocks_portfolio=-87)
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    y_tensor = torch.from_numpy(y_np).float().to(device=args.device)
    args.datadim = X_tensor.shape[1]

    sigma = torch.tensor(1, device=args.device)
    log_unnorm_posterior = partial(unnorm_log_posterior, prior_name=prior_name, sigma=sigma, X=X_tensor, y=y_tensor, args=args)

    if not just_load:
        # build model
        flow = build_circular_cond_flow_l1_manifold(args)

        # train model
        flow.train()
        flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False)
        plot_loss(loss)

        # evaluate model
        flow.eval()
        samples, cond, kl = generate_samples(flow, args, n_lambdas=20, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=10, sample_size=100, n_iter=1)
        plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=X_np.shape[-1], conf=0.95, n_plots=1, log_scale=args.log_cond)

        # uncomment to plot cumulative returns
        # samples, cond, kl = generate_samples(flow, args, n_lambdas=5, cond=True, log_unnorm_posterior=log_unnorm_posterior,
        #                                      manifold=False, context_size=1, sample_size=100, n_iter=1)
        # plot_returns(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, conf=0.95, n_plots=1)
        # plot_cumulative_returns(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, conf=0.95, n_plots=1, prior_name=prior_name)
        # plot_sparsity_patterns(samples=samples[:, :30], prior_name=prior_name)

        samples, cond, kl = generate_samples(flow, args, n_lambdas=100, cond=True, log_unnorm_posterior=log_unnorm_posterior,
                                             manifold=False, context_size=1, sample_size=500, n_iter=1)
        np.save(f'data_{prior_name}.npy', samples)
    else:
        samples_uniform = np.load('data_uniform.npy')  # load
        samples_dirichlet = np.load('data_dirichlet.npy')  # load

        plot_sparsity_distr(samples_uniform, samples_dirichlet, X_np, y_np, threshold=0.01, n_bins=25)

if __name__ == "__main__":
    main()