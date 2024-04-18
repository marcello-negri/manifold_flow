import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset, generate_regression_dataset_positive_coeff
from imf.experiments.architecture import build_cond_flow_reverse, build_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=1_000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=0.1, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=2., help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=2_000, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=1, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
parser.add_argument('--beta', metavar='be', type=float, default=1.0, help='p of the lp norm')


# MODEL PARAMETERS
parser.add_argument("--n_layers", metavar='nl', type=int, default=10, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--n_context_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])

args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_directories():
    dir_name = "plots/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

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

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args):
    if args.log_cond: alpha_ = 10 ** alpha
    else: alpha_ = alpha
    # dim = beta.shape[-1]
    # alpha_ = torch.ones_like(alpha) * 100

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior

def dirichlet_prior_beta(beta: torch.Tensor):

    dim = beta.shape[-1]
    alpha = torch.ones((1, dim), device= beta.device) * 10
    # alpha [:,:5] = 0.0001

    log_const = torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha.sum(-1))
    # log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = ((alpha - 1) * torch.log(beta)).sum(-1)

    return log_const + log_prior


def unnorm_log_posterior(beta: torch.Tensor, cond: torch.Tensor, lamb:torch.Tensor, X: torch.Tensor, y: torch.Tensor, args, flow_prior=None):
    log_lik = gaussian_log_likelihood(beta=beta, sigma=cond, X=X, y=y)
    # log_prior = laplace_prior(beta=beta, lamb=cond, args=args)
    # log_prior = lp_norm_prior(beta=beta, cond=cond, args=args)
    log_prior = dirichlet_prior(beta=beta, alpha=lamb, args=args)
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
    create_directories()
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

def main():
    args.log_cond = False

    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # load data
    # X_tensor, y_tensor, X_np, y_np = load_diabetes_dataset(device=args.device)
    X_np, y_np, true_beta = generate_regression_dataset_positive_coeff(n_samples=500, n_features=5, n_non_zero=3, noise_std=1)
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    y_tensor = torch.from_numpy(y_np).float().to(device=args.device)
    args.datadim = X_tensor.shape[1]

    # flow_prior = compute_norm_generalized_gaussian(args)
    # flow_prior.eval()
    # sigma = torch.tensor(2, device=args.device)
    lamb = torch.tensor(1, device=args.device)
    log_unnorm_posterior = partial(unnorm_log_posterior, lamb=lamb, X=X_tensor, y=y_tensor, args=args)
    # log_unnorm_posterior = partial(unnorm_log_posterior, sigma=sigma, X=X_tensor, y=y_tensor, flow_prior=flow_prior, args=args)
    # a0 = torch.tensor(2., device=args.device)
    # b0 = torch.tensor(2., device=args.device)
    # log_unnorm_posterior = partial(t_student_log_likelihood, a0=a0, b0=b0, X=X_tensor, y=y_tensor)

    # build model

    # flow = build_circular_cond_flow_l1_manifold(args)
    flow = build_cond_flow_l1_manifold(args)

    # torch.autograd.set_detect_anomaly(True)
    # train model
    flow.train()
    flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False)
    plot_loss(loss)

    # evaluate model
    flow.eval()
    samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=10, sample_size=100, n_iter=100)
    # plot_betas_norm(samples_sorted=samples, norm_sorted=cond, X_np=X_np, y_np=y_np, norm=args.beta)#, true_coeff=true_beta)
    plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=X_np.shape[-1], conf=0.95, n_plots=1, true_coeff=true_beta, log_scale=args.log_cond)

    # plot_marginal_likelihood(kl, cond, args)


if __name__ == "__main__":
    main()