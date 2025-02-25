import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset
from imf.experiments.architecture import build_cond_flow_reverse
from imf.experiments.plots import plot_betas_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood

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
parser.add_argument('--cond_max', metavar='cmax', type=float, default=4., help='minimum value of conditional variable')
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

def compute_logsurface_sphere(dim, radius):
    log_surface_unit = 0.5 * dim * np.log(torch.pi) + np.log(2) - sp.special.loggamma(0.5 * dim)
    log_surface = log_surface_unit + (dim - 1) * torch.log(radius)
    return log_surface
from imf.experiments.utils_manifold import cartesian_to_spherical_torch
def unnorm_lop_posterior(beta: torch.Tensor, cond: torch.Tensor, sigma:torch.Tensor, X: torch.Tensor, y: torch.Tensor):
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigma, X=X, y=y)
    # log_prior = laplace_prior(beta=beta, lamb=cond, args=args)
    radius = cartesian_to_spherical_torch(beta)[...,0]
    log_prior = -compute_logsurface_sphere(dim=beta.shape[-1], radius=radius)
    # log_prior = - (beta.shape[-1]-1) * torch.log(cond)
    return log_lik + log_prior


def t_student_log_likelihood(beta: torch.Tensor, a0: torch.Tensor, b0: torch.Tensor, X: torch.Tensor, y: torch.Tensor):

    N = X.shape[0]
    log_lk = -(a0 + 0.5 * N) * torch.log(1 + 0.5 * (y - beta @ X.T).square().sum(-1) / b0 )
    log_lk_const = torch.lgamma(a0 + 0.5 * N) - torch.lgamma(a0) - 0.5 * N * torch.log(2 * np.pi * b0)

    return log_lk + log_lk_const



def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # load data
    X_tensor, y_tensor, X_np, y_np = load_diabetes_dataset(device=args.device)
    # X_np, y_np, true_beta = generate_regression_dataset(n_samples=20, n_features=10, n_non_zero=8, noise_std=1)
    # X_np, y_np, true_beta = generate_regression_dataset(n_samples=100, n_features=100, n_non_zero=80, noise_std=1)
    # X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    # y_tensor = torch.from_numpy(y_np).float().to(device=args.device)

    sigma = torch.tensor(0.7, device=args.device)
    # log_unnorm_posterior = partial(gaussian_log_likelihood, sigma=sigma, X=X_tensor, y=y_tensor)
    log_unnorm_posterior = partial(unnorm_lop_posterior, sigma=sigma, X=X_tensor, y=y_tensor)
    # a0 = torch.tensor(2., device=args.device)
    # b0 = torch.tensor(2., device=args.device)
    # log_unnorm_posterior = partial(t_student_log_likelihood, a0=a0, b0=b0, X=X_tensor, y=y_tensor)

    # build model
    args.datadim = X_tensor.shape[1]
    flow = build_cond_flow_reverse(args, clamp_theta=False)

    # torch.autograd.set_detect_anomaly(True)
    # train model
    args.likelihood_cond = False
    flow.train()
    flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=args.likelihood_cond)
    plot_loss(loss)

    # evaluate model
    flow.eval()
    samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, context_size=10, sample_size=1000, n_iter=100, manifold=args.likelihood_cond)
    plot_betas_norm(samples_sorted=samples, norm_sorted=cond, X_np=X_np, y_np=y_np, norm=args.beta)#, true_coeff=true_beta)
    np.save(f'./samples/samples_regression_manifold_{args.datadim}_{args.beta:.2f}.npy', samples)

    opt_cond, opt_idx = plot_marginal_likelihood(kl, cond, args)
    samples, cond, kl = generate_samples(flow, args, given_cond=opt_cond, cond=True, log_unnorm_posterior=log_unnorm_posterior, context_size=1, sample_size=500, n_iter=1, manifold=args.likelihood_cond)
    breakpoint()
    # print(f"Optimal sigma via MLL: {opt_cond:.3f} (true: {sigma_true:.3f})")
    np.save(f'./samples/samples_regression_manifold_{args.datadim}_{args.beta:.2f}_opt.npy', samples[0])

if __name__ == "__main__":
    main()