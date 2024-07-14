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
from imf.experiments.architecture import build_cond_flow_reverse, build_simple_circular_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_marginals, plot_sparsity_distr, plot_cumulative_returns_singularly, plot_sparsity_patterns, plot_betas_lambda, plot_marginal_likelihood, plot_returns, plot_cumulative_returns

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

def gaussian_log_likelihood(beta: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):
    # implements Gaussian log-likelihood beta ~ Normal (X@beta, sigma^2 ID)
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const


def gaussian_log_likelihood_sigmas(beta: torch.Tensor, sigmas: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):
    # implements Gaussian log-likelihood beta ~ Normal (X@beta, sigma^2 ID)
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigmas**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigmas + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args):
    K = beta.shape[-1]
    log_const = torch.lgamma(alpha * K) - K * torch.lgamma(alpha)
    log_prior = (alpha - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior


def unnorm_log_posterior(beta: torch.Tensor, prior_name: str, cond: torch.Tensor, sigma:torch.Tensor, X: torch.Tensor, y: torch.Tensor, args, flow_prior=None):
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigma, X=X, y=y)
    # log_lik = gaussian_log_likelihood_mean(beta=beta, sigma=sigma, X=X, mu=y)
    # log_prior = laplace_prior(beta=beta, lamb=cond, argss=args)
    if prior_name == "dirichlet":
        log_prior = dirichlet_prior(beta=beta, alpha=cond, args=args)
    elif prior_name == "uniform":
        log_prior = dirichlet_prior(beta=beta, alpha=torch.zeros_like(cond), args=args)
    else:
        raise ValueError(f"Prior {prior_name} not recognized")

    return log_lik + log_prior

def dirichlet_prior_beta(beta: torch.Tensor, alphas: torch.Tensor):

    log_const = torch.lgamma(alphas).sum(-1) - torch.lgamma(alphas.sum(-1))
    log_prior = ((alphas - 1) * torch.log(beta)).sum(-1)

    return log_const + log_prior

def unnorm_posterior_cond_likelihood(beta: torch.Tensor, cond: torch.Tensor, X: torch.Tensor, y: torch.Tensor, args, stds=None):
    # sigmas = 10 ** cond if args.log_cond else cond
    # log_lik = gaussian_log_likelihood_sigmas(beta=beta, sigmas=sigmas, X=X, y=y)
    log_lik = log_likelihood_mixture(beta=beta, stds=stds, X=X, y=y)

    alpha = torch.ones_like(beta[:,:,0]) * 0.01
    log_prior = dirichlet_prior(beta=beta, alpha=alpha, args=args)
    # alphas = torch.tensor([3.57, 0.36, 0.01, 0.01, 1.07], requires_grad=False).to(beta.device)
    # alphas = torch.tensor([0.36, 0.01, 1.07, 0.01, 3.57], requires_grad=False).to(beta.device)
    # if args.log_cond: alphas = alphas.log10()
    # log_prior = dirichlet_prior_beta(beta=beta, alphas=alphas)

    return log_lik + log_prior

def log_likelihood_mixture(X, y, beta, stds):
    eps = 1e-7
    y_reshaped = y.reshape(1,1,*y.shape)
    beta_X = beta@X.T
    # beta_var = (beta**2)@(stds**2)
    beta_var = beta@(stds**2+X.T**2) - beta_X**2
    log_lk = - 0.5 * (y_reshaped - beta_X.unsqueeze(-2))**2/(beta_var.unsqueeze(-2) + eps)
    log_lk = log_lk.sum(-1).sum(-1)
    log_lk_const = - y.shape[0] * torch.log((torch.sqrt(beta_var) + eps) * np.sqrt(2. * np.pi)).sum(-1)

    return log_lk + log_lk_const


def generate_regression_simplex (n, d, sigma=0.1):
    X_np = np.random.randn(n, d) #+ np.random.rand(1,d) - 0.5
    beta = np.random.rand(d)
    beta /= beta.sum() # normalize the weights on the simplex
    y_np = X_np @ beta + np.random.randn(n) * sigma

    return X_np, y_np

def load_wolves_data_(file_name):

    assert file_name in ["wolves", "killerwhale"]

    file_dir = "/home/negri0001/R/x86_64-pc-linux-gnu-library/4.4/MixSIAR/extdata/"
    source_df = pd.read_csv(file_dir + file_name + "_sources.csv")
    consumer_df = pd.read_csv(file_dir + file_name + "_consumer.csv")
    # discr_df = pd.read_csv(file_dir + "wolves_discrimination.csv")

    if file_name == "wolves":
        region = 2
        source_df =source_df[source_df["Region"]==region]
        consumer_df = consumer_df[consumer_df["Region"]==region]

    y_np = consumer_df[["d13C","d15N"]].to_numpy()
    X_np = source_df[["Meand13C", "Meand15N"]].to_numpy()
    stds = source_df[["SDd13C", "SDd15N"]].to_numpy()

    # mean = X_np.mean(0)
    # std = X_np.std(0)
    # X_np -= mean
    # X_np /= std

    # y_np -= mean
    # y_np /= std

    # we reformulate the problem as a single regression
    n_obs = y_np.shape[0]
    y_np = y_np.reshape(-1) # we ravel the features

    n_tracers = X_np.shape[0]
    X_np = np.expand_dims(X_np.T,0).repeat(n_obs,0).reshape(-1,n_tracers)

    # source data consist of mean and std. dev. values.
    # we use the mean to train the flow and use the average std.dev as ground truth

    # std_13c = source_df["Meand13C"].to_numpy().mean()
    # std_15n = source_df["Meand15N"].to_numpy().mean()
    # sigma_svg = 0.5 * (std_13c + std_15n)

    return X_np, y_np, stds

def load_wolves_data(file_name):

    assert file_name in ["wolves", "killerwhale"]

    file_dir = "/home/negri0001/R/x86_64-pc-linux-gnu-library/4.4/MixSIAR/extdata/"
    source_df = pd.read_csv(file_dir + file_name + "_sources.csv")
    consumer_df = pd.read_csv(file_dir + file_name + "_consumer.csv")
    # discr_df = pd.read_csv(file_dir + "wolves_discrimination.csv")

    if file_name == "wolves":
        region = 1
        # pack = 3
        source_df = source_df[source_df["Region"]==region]
        consumer_df = consumer_df[consumer_df["Region"]==region]
        # consumer_df = consumer_df[consumer_df["Pack"]==pack]

    y_np = consumer_df[["d13C","d15N"]].to_numpy()
    X_np = source_df[["Meand13C", "Meand15N"]].to_numpy()
    stds = source_df[["SDd13C", "SDd15N"]].to_numpy()

    name_dict = dict(zip(range(source_df.shape[0]), source_df.iloc[:, 0].to_numpy()))

    return X_np.T, y_np, stds, name_dict


def load_R_data(data_name="isopod"):
    file_dir = "/home/negri0001/R/x86_64-pc-linux-gnu-library/4.4/MixSIAR/extdata/"
    source_df = pd.read_csv(file_dir + data_name + "_sources.csv")
    consumer_df = pd.read_csv(file_dir + data_name + "_consumer.csv")
    # discr_df = pd.read_csv(file_dir + "wolves_discrimination.csv")

    n_tracers = len(source_df.columns[1:])//2
    X_np = consumer_df[consumer_df.columns[1:]].to_numpy()
    y_np = source_df[source_df.columns[1:n_tracers]].to_numpy()

    # source data consist of mean and std. dev. values.
    # we use the mean to train the flow and use the average std.dev as ground truth

    std_13c = source_df["Meand13C"].to_numpy().mean()
    std_15n = source_df["Meand15N"].to_numpy().mean()
    sigma_svg = 0.5 * (std_13c + std_15n)

    return X_np, y_np, sigma_svg

def main():
    just_load = False # False: train model and save samples. True: load the samples
    args.log_cond = True
    prior_name = "dirichlet" # independent of args.cond_min and args.cond_max
    # set random seed for reproducibility
    set_random_seeds(args.seed)

    # load data
    # dates, X_np, y_np = load_returns_dataset(stock_to_replicate=7, timesteps=-353, n_stocks_portfolio=-87)
    # dates, X_np, y_np = load_returns_dataset(stock_to_replicate=0, timesteps=-353, n_stocks_portfolio=-7)
    # sigma_true = 0.09
    X_np, y_np, stds, name_dict = load_wolves_data(file_name="wolves")
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    y_tensor = torch.from_numpy(y_np).float().to(device=args.device)
    stds_tensor = torch.from_numpy(stds).float().to(device=args.device)
    args.datadim = X_tensor.shape[1]

    # log_unnorm_posterior = partial(unnorm_log_posterior, prior_name=prior_name, sigma=sigma, X=X_tensor, y=y_tensor, args=args)#, flow_prior=flow_prior)
    log_unnorm_posterior = partial(unnorm_posterior_cond_likelihood, X=X_tensor, y=y_tensor, args=args, stds=stds_tensor)#, flow_prior=flow_prior)
    if not just_load:
        # build model
        flow = build_circular_cond_flow_l1_manifold(args)

        # train model
        flow.train()
        flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False)
        plot_loss(loss)

        # evaluate model
        flow.eval()
        samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=1, sample_size=10000, n_iter=100)
        plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=X_np.shape[-1], conf=0.95, n_plots=1, log_scale=args.log_cond)

        # uncomment to plot cumulative returns
        # samples, cond, kl = generate_samples(flow, args, n_lambdas=5, cond=True, log_unnorm_posterior=log_unnorm_posterior,
        #                                      manifold=False, context_size=1, sample_size=100, n_iter=1)
        # plot_returns(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, conf=0.95, n_plots=1)
        # plot_cumulative_returns(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, conf=0.95, n_plots=1, prior_name=prior_name)
        # plot_sparsity_patterns(samples=samples[:, :30], prior_name=prior_name)

        # samples, cond, kl = generate_samples(flow, args, n_lambdas=25, cond=True, log_unnorm_posterior=log_unnorm_posterior,
        #                                      manifold=False, context_size=1, sample_size=500, n_iter=1)
        opt_cond, opt_idx = plot_marginal_likelihood(kl, cond, args)
        # print(f"Optimal sigma via MLL: {opt_cond:.3f} (true: {sigma_true:.3f})")
        plot_marginals(samples, opt_idx, name_dict=name_dict, n_bins=100)
        np.save(f'data_{prior_name}.npy', samples)
    else:
        samples_uniform = np.load('data_uniform.npy')  # load
        samples_dirichlet = np.load('data_dirichlet.npy')  # load

        plot_sparsity_distr(samples_uniform, samples_dirichlet, X_np, y_np, threshold=0.01, n_bins=25)


if __name__ == "__main__":
    main()