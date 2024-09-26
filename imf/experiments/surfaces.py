import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import os
import argparse
import math

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples, train_model_reverse
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset
from imf.experiments.architecture import build_cond_flow_reverse,  build_flow_reverse
from imf.experiments.datasets import  create_dataset
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

parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform"])
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.00, help='std of the isotropic noise in the data')
parser.add_argument('--n_samples_dataset', metavar='nsd', type=int, default=10_000, help='number of data points in the dataset')
parser.add_argument('--radius', metavar='ra', type=float, default=1.0, help='radius of manifold')

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

def define_model_name(args, dataset):
    args.model_name = (f"./models/imf_{args.dataset}_{args.architecture}_lm{args.learn_manifold}_{args.logabs_jacobian}"
                       f"{dataset.dataset_suffix}_epochs{args.epochs}_seed{args.seed}")

def compute_logsurface_sphere(dim, radius):
    log_surface_unit = 0.5 * dim * np.log(torch.pi) + np.log(2) - sp.special.loggamma(0.5 * dim)
    log_surface = log_surface_unit + (dim - 1) * torch.log(radius)
    return log_surface

def compute_logsurface_l1ball(dim, radius):
    log_surface_unit_simplex = 0.5 * np.log(dim + 1) - sp.special.loggamma(dim + 1) - 0.5 * dim * np.log(2)
    log_surface_simplex = log_surface_unit_simplex + dim * torch.log(radius)
    log_surface_unit_l1ball = log_surface_simplex + (dim + 1) * np.log(2)
    return log_surface_unit_l1ball
from imf.experiments.utils_manifold import cartesian_to_spherical_torch
# def unnorm_lop_posterior(beta: torch.Tensor):
#     cosntant = torch.ones_like(beta[...,0])
#     return cosntant

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # args.dataset = "lp_uniform"
    dataset = create_dataset(args=args)
    define_model_name(args, dataset)

    flow = build_flow_reverse(args=args)

    args.likelihood_cond = False
    # train model
    flow.train()
    flow, loss = train_model_reverse(model=flow, args=args, dataset=dataset, batch_size=1000)
    plot_loss(loss)

    # evaluate model
    flow.eval()
    samples, logprob_flow = flow.sample_and_log_prob(num_samples=1000, context=None)
    # logprob_target = dataset.log_density(samples)  # uniform on lp manifold
    log_surface = torch.mean(-logprob_flow)
    if args.dataset == "uniform":
        log_surface_gt = compute_logsurface_sphere(dim=args.datadim, radius=torch.ones(1))
    elif args.dataset == "lp_uniform":
        log_surface_gt = compute_logsurface_l1ball(dim=args.datadim-1, radius=torch.ones(1)*np.sqrt(2))

    breakpoint()

if __name__ == "__main__":
    main()