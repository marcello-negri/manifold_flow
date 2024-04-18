import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset, generate_regression_dataset_positive_coeff
from imf.experiments.architecture import build_flow_reverse, build_cond_flow_reverse, build_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
from imf.experiments.plots import plot_samples_3d, plot_angles_3d, plot_betas_lambda_fixed_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood, plot_clusters

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
parser.add_argument('--cond_min', metavar='cmin', type=float, default=1., help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=1., help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=2_000, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=1, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
parser.add_argument('--beta', metavar='be', type=float, default=1, help='p of the lp norm')
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument("--dataset", type=str, default="lp_uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform"])


# MODEL PARAMETERS
parser.add_argument("--n_layers", metavar='nl', type=int, default=10, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=128, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--n_context_features", metavar='nf', type=int, default=128, help='number of hidden features in the embedding space of the flow model')
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

def uniform_prior(beta: torch.Tensor, cond: torch.Tensor):
    return torch.ones_like(beta[:,:,0])

def uniform_plane_prior(beta: torch.Tensor, cond: torch.Tensor):
    normal = beta.new_ones(beta.shape[-1]) # dim
    offset_beta = beta - (1 / beta.shape[-1]) * normal # nc, nb, dim
    log_prior = 100 * (offset_beta * normal).sum(-1)

    return log_prior

def uniform_plane_prior_diff(beta: torch.Tensor, cond: torch.Tensor):
    mask = beta > 0
    mask = torch.all(mask, dim=-1)
    return torch.where(mask, 10, -10)


def distance_prior(beta: torch.Tensor, cond: torch.Tensor):
    fixed_point = beta.new_ones(beta.shape[-1])
    log_prob = -(torch.norm(4*(beta-fixed_point) , dim=-1))**2
    return log_prob

def uniform_positive_prior(beta: torch.Tensor, cond: torch.Tensor):
    log_prob = 1 * beta.new_ones(beta.shape[:-1])
    mask = (beta<=0).to(dtype=torch.float).prod(-1)
    log_prob = mask * log_prob - 3 * (1 - mask) * log_prob

    return log_prob

def uniform_positive_prior_sigmoid(beta: torch.Tensor, cond: torch.Tensor):
    # log_prob = 1 * beta.new_ones(beta.shape[:-1])
    sigmoid_beta = torch.sigmoid(1000*beta).prod(-1)
    return sigmoid_beta * 50

def dirichlet_prior(beta: torch.Tensor, cond: torch.Tensor):
    alpha_ = beta.new_ones(beta.shape[:-1]) * 0.1
    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1)
    # breakpoint()

    return log_const + log_prior

def main():
    args.log_cond = False

    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # log_unnorm_posterior = uniform_positive_prior
    # log_unnorm_posterior = uniform_prior
    # log_unnorm_posterior = distance_prior
    # log_unnorm_posterior = uniform_plane_prior
    # log_unnorm_posterior = uniform_positive_prior_sigmoid
    # log_unnorm_posterior = uniform_plane_prior_diff
    log_unnorm_posterior = dirichlet_prior

    # build model
    flow = build_cond_flow_reverse(args=args, clamp_theta=True)
    # flow = build_circular_cond_flow_l1_manifold(args)

    # torch.autograd.set_detect_anomaly(True)
    # train model
    flow.train()
    flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False, conditional=True)
    plot_loss(loss)

    # evaluate model
    flow.eval()
    samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=10, sample_size=100, n_iter=100)
    # plot_betas_norm(samples_sorted=samples, norm_sorted=cond, X_np=X_np, y_np=y_np, norm=args.beta)#, true_coeff=true_beta)
    # plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=args.datadim, conf=0.95, n_plots=1, log_scale=args.log_cond)

    if args.datadim == 3:
        plot_samples_3d(samples.reshape(-1,3))

    plot_angles_3d(samples, args)

    # cond_indices = np.linspace(0, cond.shape[-1]-1, 2, dtype='int')
    # plot_clusters(samples, cond, cond_indices, 5)
    # plot_marginal_likelihood(kl, cond, args)

if __name__ == "__main__":
    main()