import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from sklearn import metrics
from dask.array.random import vonmises
from torch.onnx.symbolic_opset9 import clamp

from imf.experiments.utils_manifold import train_model_forward, train_model_reverse, generate_samples
from imf.experiments.utils_manifold import  define_model_name
from imf.experiments.architecture import build_flow_forward, build_flow_reverse
from imf.experiments.plots import plot_icosphere, plot_loss, plot_samples, plot_3Dmanifold, plot_pairwise_angle_distribution, plot_angle_distribution, plot_samples_pca, density_gt
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
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "cholesky", "analytical_lu", "analytical_block", "fff", "rect"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')
parser.add_argument('--manifold_type', metavar='ics', type=int, default=1, help='deformed sphere manifold type', choices=[0,1,2,3,4,5])


# DATASETS PARAMETERS
parser.add_argument("--data_folder", type=str, default="~/Documents/Marcello/cond_flows/manifold_flow/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform", "deformed_sphere1", "custom_on_sphere"])
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

parser.add_argument('--n_hutchinson_samples', type=int, default=1, help='number of hutchinson samples')


args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_directories():
    dir_name = "plots/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_namei, exist_ok=True)
    model_dir = "models/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

def evaluate_prob(samples, logp_flow, model_name, dataset, normalized_target=True):
    p_gt = density_gt(points=samples, dataset=dataset)
    # norm_const_gt = 1. if normalized_target else p_gt.sum()
    # norm_const_flow = 1. if normalized_target else np.exp(p_gt).sum()
    # Compute the L2 norm for both distributions
    # MSE_logp = np.sqrt(np.square(np.exp(logp_flow)/norm_const_flow - p_gt/norm_const_gt).mean())

    if normalized_target:
        MSE_logp = np.sqrt(np.square(logp_flow - np.log(p_gt)).mean())
    else:
        avg_diff = np.mean(logp_flow - np.log(p_gt))
        # MSE_logp = np.sqrt(np.square(np.exp(logp_flow) - p_gt/(-avg_diff-1)).mean())
        # MSE_logp = np.sqrt(np.square(np.exp(logp_flow) - p_gt - avg_diff).mean())
        MSE_logp = np.sqrt(np.mean(np.square(logp_flow - np.log(p_gt) - avg_diff)))

    f = open(model_name + ".txt", "a")
    f.write(f"mse;{MSE_logp}")
    f.close()
    print(f"MSE prob: {MSE_logp:.5f}")
    return MSE_logp

def evaluate_samples(X, Y, model_name, gamma=1.0):
    """
    Code taked from https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py
    MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    #XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    #YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    #XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    #MMD_rbf = XX.mean() + YY.mean() - 2 * XY.mean()
    
    #degree = 2
    #coef0 = 1
    #XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    #YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    #XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    #MMD_poly = XX.mean() + YY.mean() - 2 * XY.mean()
    
    #f = open(model_name + ".txt", "a")
    #f.write(f"\n{MMD_rbf}")
    #f.write(f"\n{MMD_poly}")
    #f.close()
    #print(f"MMD_rbf: {MMD_rbf:.8f}\nMMD_poly: {MMD_poly:.8f}")
    
    #return MMD_rbf, MMD_poly

    from geomloss import SamplesLoss
    f = open(model_name + ".txt", "a")
    losses = ["sinkhorn", "energy", "gaussian", "laplacian"]
    for loss in losses:
        SamplesLoss(loss=loss)
        loss_function = SamplesLoss(loss)
        X_torch = torch.from_numpy(X).float()
        Y_torch = torch.from_numpy(Y).float()
        loss_value = loss_function(X_torch, Y_torch).item()
        print(loss, loss_value)
        f.write(f"\n{loss};{loss_value}")
    f.write("\n")
    f.close()

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # torch.autograd.set_detect_anomaly(True)
    spherical_datasets = ["uniform", "uniform_checkerboard", "vonmises_fisher",
                          "vonmises_fisher_mixture", "vonmises_fisher_mixture_spiral", "custom_on_sphere"]
    sphere = True if args.dataset in spherical_datasets else False
    normalized_densities = ["uniform", "vonmises_fisher", "vonmises_fisher_mixture", "vonmises_fisher_mixture_spiral"]
    norm_density = True if args.dataset in normalized_densities else False

    # load dataset and samples
    dataset = create_dataset(args=args)
    train_data_np = dataset.load_samples(split="train", overwrite=True)
    test_data_np = dataset.load_samples(split="test", overwrite=True)
    train_data = torch.from_numpy(train_data_np).float().to(args.device)
    test_data = torch.from_numpy(test_data_np).float().to(args.device)

    plot_samples(train_data_np, n_samples=100000)

    # build flow
    clamp_theta = 1e-1 # else None
    if args.kl_div == "forward":
        flow = build_flow_forward(args)#, clamp_theta=clamp_theta)
    elif args.kl_div == "reverse":
        flow = build_flow_reverse(args)#,clamp_theta=clamp_theta)
    define_model_name(args, dataset)

    print(args)

    # torch.autograd.detect_anomaly(True)

    # train flow
    if not os.path.isfile(args.model_name) or args.overwrite:
        if args.kl_div == "forward":
            flow, loss = train_model_forward(model=flow, data=train_data, args=args, batch_size=2_000, early_stopping=False)
        elif args.kl_div == "reverse":
            flow, loss = train_model_reverse(model=flow, dataset=dataset, train_data_np=train_data_np, batch_size=1000, args=args)
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
    # set logabs_jacobian to "analytical_block" to compute exact log likelihood
    flow._transform._transforms[0].logabs_jacobian = "analytical_block"
    samples_flow, log_probs_flow = generate_samples(flow, args=args, sample_size=50, n_iter=20)
    n_samples = min(samples_flow.shape[0], train_data_np.shape[0])
    MSE_prob = evaluate_prob(samples=samples_flow, logp_flow=log_probs_flow, dataset=dataset, model_name=args.model_name, normalized_target=False)
    MMD_samples = evaluate_samples(X=samples_flow, Y=test_data_np, model_name=args.model_name)
    # plot_3Dmanifold(dataset, flow, args=args)
    plot_samples(samples_flow)
    if args.datadim == 3:
        plot_icosphere(data=train_data_np[:n_samples], dataset=dataset, flow=flow, samples_flow=samples_flow[:n_samples],
                       rnf=None, samples_rnf=None, device='cuda', args=args, plot_rnf=False, sphere=sphere)


if __name__ == "__main__":
    main()
