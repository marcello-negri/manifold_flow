import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from imf.experiments.utils_manifold import train_model_forward, train_model_reverse, generate_samples, train_model_forward_
from imf.experiments.utils_manifold import  define_model_name
from imf.experiments.architecture import build_flow_forward, build_flow_reverse, build_paramhyper_flow_forward, build_paramhyper_flow_forward_
from imf.experiments.plots import plot_icosphere, plot_loss, plot_samples, plot_pairwise_angle_distribution, plot_angle_distribution, plot_samples_pca
from imf.experiments.datasets import create_dataset
from imf.experiments.plots import map_colors
from scipy.stats import gaussian_kde

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
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical", "cholesky", "analytical_lu"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')

# DATASETS PARAMETERS
parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform")# choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "uniform_torus", "vonmises_fisher_mixture_spiral", "lp_uniform"])
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.00, help='std of the isotropic noise in the data')
parser.add_argument('--n_samples_dataset', metavar='nsd', type=int, default=10_000, help='number of data points in the dataset')
# von mises fisher parameters
parser.add_argument('--mu', metavar='m', type=float, default=None, help='mean of von mises distribution')
parser.add_argument('--kappa', metavar='k', type=float, default=5.0, help='concentration parameter of von mises distribution')
# von mises fisher mixture parameters
parser.add_argument('--n_mix', metavar='nm', type=int, default=50, help='number of mixture components for mixture of von mises fisher distribution')
parser.add_argument('--kappa_mix', metavar='km', type=float, default=50.0, help='concentration parameter of mixture of von mises distribution')
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

def plot_density_manifold(samples, xyz, args, gt=True):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    kde_func = gaussian_kde(samples.T)
    trisurf = ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    mappable = map_colors(trisurf, kde_func, cmap='viridis', kde=True)
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    if gt:
        plt.savefig(f"./plots/density_plot_{(args.model_name).split('/')[-1]}_gt.pdf", dpi=300)
    else:
        plt.savefig(f"./plots/density_plot_{(args.model_name).split('/')[-1]}_flow.pdf", dpi=300)
    plt.show()

    probs = kde_func(xyz.T)
    return probs

def plot_comparison(dataset, flow, samples_gt, samples_flow, args, n_gridpoints=100, alternative=False):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(samples_gt[:, 0], samples_gt[:, 1], samples_gt[:, 2], marker='.', alpha=0.2)
    plt.savefig(f"./plots/sparsity_plot_{(args.model_name).split('/')[-1]}_gt.pdf", dpi=300)
    plt.show()

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(samples_flow[:, 0], samples_flow[:, 1], samples_flow[:, 2], marker='.', alpha=0.2)
    plt.savefig(f"./plots/sparsity_plot_{(args.model_name).split('/')[-1]}_flow.pdf", dpi=300)
    plt.show()

    lins = torch.linspace(-1,1, n_gridpoints, device=args.device)
    xy = torch.cartesian_prod(lins, lins)

    # manifold from gt
    xy_np = xy.detach().cpu()
    z = dataset.hypersurface(xy_np)
    xyz_gt = np.c_[xy_np, z]

    probs_gt = plot_density_manifold(samples_gt, xyz_gt, args, gt=True)

    # manifold from flow
    xy_ = torch.cat((xy, torch.ones_like(xy[:,:1])), dim=1)
    if alternative:
        xy_flow, _ = flow._transform.forward(xy_, context=None)
        xyz_flow, _ = flow._transform.inverse(xy_flow, context=None)
    else:
        xy_flow, _ = flow._transform._transforms[0].forward(xy_, context=None)
        xyz_flow, _ = flow._transform._transforms[0].inverse(xy_flow, context=None)

    xyz_flow = xyz_flow.detach().cpu().numpy()
    probs_flow = plot_density_manifold(samples_flow, xyz_flow, args, gt=False)

    MSE_prob = np.sqrt(np.square(probs_gt - probs_flow).mean())
    MSE_manifold = np.sqrt(np.square(z-xyz_flow[:,-1]).mean(-1))
    print(MSE_prob)
    print(MSE_manifold)

    f = open(args.model_name + ".txt", "a")
    f.write(f"MSE: {MSE_manifold:.5f}")
    f.close()

    return MSE

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    args.r = 0.25
    args.R = 0.5
    # args.dataset = "uniform_torus"
    # args.dataset = "uniform"
    # args.dataset = "uniform_hyper"
    # args.dataset = "vonmises_fisher_mixture_spiral"

    # load dataset and samples
    dataset = create_dataset(args=args)
    train_data_np = dataset.load_samples(split="train", overwrite=True)
    test_data_np = dataset.load_samples(split="test", overwrite=True)
    train_data = torch.from_numpy(train_data_np).float().to(args.device)
    test_data = torch.from_numpy(test_data_np).float().to(args.device)

    plot_samples(train_data_np, n_samples=100_000)

    # build flow
    if args.kl_div == "forward":
        flow = build_paramhyper_flow_forward(args)
    elif args.kl_div == "reverse":
        flow = build_flow_reverse(args)
    define_model_name(args, dataset)

    print(args)

    if not os.path.isfile(args.model_name) or args.overwrite:
        if args.kl_div == "forward":
            flow, loss = train_model_forward(model=flow, data=train_data, args=args, batch_size=1000, early_stopping=False)
        elif args.kl_div == "reverse":
            flow, loss = train_model_reverse(model=flow, dataset=dataset, train_data_np=train_data_np, args=args)
        plot_loss(loss)
        flow.eval()
    else:
        if args.kl_div == "forward":
            flow = build_paramhyper_flow_forward(args)
        elif args.kl_div == "reverse":
            flow = build_flow_reverse(args)
        flow.load_state_dict(torch.load(args.model_name))
        flow.eval()

    # evaluate learnt distribution

    samples_flow, log_probs_flow = flow.sample_and_log_prob(num_samples=10_000, context=None)
    # base_points = flow.transform_to_noise(samples_flow, context=None)
    # surface_points, logabsdet = flow._transform.inverse(base_points, context=None)
    samples_flow = samples_flow.detach().cpu().numpy()
    # fig = plt.figure(figsize=(15, 15))
    # if args.datadim == 2:
    #     ax = fig.add_subplot()
    #     # ax.scatter(test_data_np[:, 0], test_data_np[:, 1], marker='.', color='r')
    #     ax.scatter(samples_flow[:, 0], samples_flow[:, 1], marker='.', color='b', alpha=0.5)
    #     plt.show()
    # elif args.datadim == 3:
    MSE = plot_comparison(dataset=dataset, flow=flow, samples_gt=train_data_np, samples_flow=samples_flow, n_gridpoints=100, args=args)



if __name__ == "__main__":
    main()