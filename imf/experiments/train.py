import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from imf.experiments.utils_manifold import train_model_forward, train_model_reverse, generate_samples
from imf.experiments.utils_manifold import  define_model_name
from imf.experiments.architecture import build_flow_forward, build_flow_reverse
from imf.experiments.plots import plot_icosphere, plot_loss, plot_samples, plot_pairwise_angle_distribution, plot_angle_distribution, plot_samples_pca, density_gt
from imf.experiments.datasets import create_dataset

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process some integers.')

# MODEL PARAMETERS
parser.add_argument('--epochs', metavar='e', type=int, default=1_000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument("--n_layers", metavar='nl', type=int, default=10, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "cholesky", "analytical_lu"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')

# DATASETS PARAMETERS
parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform"])
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.00, help='std of the isotropic noise in the data')
parser.add_argument('--n_samples_dataset', metavar='nsd', type=int, default=10_000, help='number of data points in the dataset')
# von mises fisher parameters
parser.add_argument('--mu', metavar='m', type=float, default=None, help='mean of von mises distribution')
parser.add_argument('--kappa', metavar='k', type=float, default=5.0, help='concentration parameter of von mises distribution')
# von mises fisher mixture parameters
parser.add_argument('--n_mix', metavar='nm', type=int, default=50, help='number of mixture components for mixture of von mises fisher distribution')
parser.add_argument('--kappa_mix', metavar='km', type=float, default=30.0, help='concentration parameter of mixture of von mises distribution')
parser.add_argument('--alpha_mix', metavar='am', type=float, default=0.3, help='alpha parameter of mixture of von mises distribution')
parser.add_argument('--n_turns_spiral', metavar='ns', type=int, default=4, help='number of spiral turns for sphere spiral distribution')
# uniform checkerboard parameters
parser.add_argument('--n_theta', metavar='nt', type=int, default=6, help='number of rows in the checkerboard (n_theta>0)')
parser.add_argument('--n_phi', metavar='np', type=int, default=6, help='number of columns in the checkerboard (n_phi>0 and must be even)')
# lp uniform parameters
parser.add_argument('--beta', metavar='be', type=float, default=1.0, help='p of the lp norm')
parser.add_argument('--radius', metavar='ra', type=float, default=1.0, help='radius of manifold')


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

def evaluate_prob(samples, logp_flow, dataset, normalized_target=True):
    p_gt = density_gt(points=samples, dataset=dataset)
    norm_const_gt = 1. if normalized_target else p_gt.sum()
    norm_const_flow = 1. if normalized_target else np.exp(p_gt).sum()
    MSE_logp = np.sqrt(np.square(np.exp(logp_flow)/norm_const_flow - p_gt/norm_const_gt).mean())
    print(f"MSE prob: {MSE_logp:.5f}")
    return MSE_logp


def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # torch.autograd.set_detect_anomaly(True)
    spherical_datasets = ["uniform", "uniform_checkerboard", "vonmises_fisher",
                          "vonmises_fisher_mixture", "vonmises_fisher_mixture_spiral"]
    sphere = True if args.dataset in spherical_datasets else False

    # load dataset and samples
    dataset = create_dataset(args=args)
    train_data_np = dataset.load_samples(split="train", overwrite=True)
    test_data_np = dataset.load_samples(split="test", overwrite=True)
    train_data = torch.from_numpy(train_data_np).float().to(args.device)
    test_data = torch.from_numpy(test_data_np).float().to(args.device)

    plot_samples(train_data_np, n_samples=100000)

    # build flow
    if args.kl_div == "forward":
        flow = build_flow_forward(args)
    elif args.kl_div == "reverse":
        flow = build_flow_reverse(args)
    define_model_name(args, dataset)

    print(args)

    # torch.autograd.detect_anomaly(True)

    # train flow
    if not os.path.isfile(args.model_name) or args.overwrite:
        if args.kl_div == "forward":
            flow, loss = train_model_forward(model=flow, data=train_data, args=args, batch_size=10_000, early_stopping=False)
        elif args.kl_div == "reverse":
            flow, loss = train_model_reverse(model=flow, dataset=dataset, train_data_np=train_data_np, batch_size=10_000, args=args)
        plot_loss(loss)
        flow.eval()
    else:
        if args.kl_div == "forward":
            flow = build_flow_forward(args)
        elif args.kl_div == "reverse":
            flow = build_flow_reverse(args)
        flow.load_state_dict(torch.load(args.model_name))
        flow.eval()

    # evaluate learnt distribution
    samples_flow, log_probs_flow = generate_samples(flow, args=args, sample_size=100, n_iter=10)
    n_samples = min(samples_flow.shape[0], train_data_np.shape[0])
    MSE_prob = evaluate_prob(samples=samples_flow, logp_flow=log_probs_flow, dataset=dataset, normalized_target=True)
    plot_icosphere(data=train_data_np[:n_samples], dataset=dataset, flow=flow, samples_flow=samples_flow[:n_samples],
                   rnf=None, samples_rnf=None, device='cuda', args=args, plot_rnf=False, sphere=sphere)
    if args.dataset == "uniform":
        plot_pairwise_angle_distribution(samples_flow, n_samples=1000)
    plot_angle_distribution(samples_flow=samples_flow, samples_gt=train_data_np, device=args.device)
    plot_samples_pca(samples_flow, train_data_np)

if __name__ == "__main__":
    main()