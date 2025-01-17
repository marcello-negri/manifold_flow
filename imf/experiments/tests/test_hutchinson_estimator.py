import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse
import tqdm
from geomstats.visualization import Sphere
from enflows.transforms import NaiveLinear
from sklearn import metrics
from scipy.stats import wasserstein_distance
from dask.array.random import vonmises
from torch.onnx.symbolic_opset9 import clamp

from enflows.distributions import Uniform, StandardNormal
from enflows.distributions.uniform import UniformSphere
from enflows.flows.base import Flow
from enflows.transforms.base import InverseTransform, CompositeTransform
from enflows.transforms.injective import ScaleLastDim, LearnableManifoldFlow, SphereFlow, LpManifoldFlow, ConstrainedAnglesSigmoid

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
parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform", "deformed_sphere1"])
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

def build_flow_manifold_minimal(n_dim, logabs_jacobian, device='cuda'):
    # base_dist = UniformSphere(shape=[n_dim - 1])
    # base_dist = Uniform(shape=[args.datadim - 1], low=torch_zero, high=torch_pi)
    base_dist = StandardNormal(shape=[n_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        InverseTransform(
            NaiveLinear(features=n_dim - 1)
        )
    )
    transformation_layers.append(
        ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True)
    )

    transformation_layers.append(
            SphereFlow(n=n_dim - 1, logabs_jacobian=logabs_jacobian)
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)

    context = None
    n_dim = 100
    n_reps = 20
    n_samples = 20
    n_hutchinson_samples = np.linspace(1,100, n_samples).astype(int)

    # manifold_mapping = LearnableManifoldFlow(n=n_dim - 1, max_radius=2., logabs_jacobian="analytical_block").to(args.device)
    # manifold_mapping = SphereFlow(n=n_dim - 1, logabs_jacobian="analytical_block").to(args.device)
    # base_dist = UniformSphere(shape=[n_dim-1])
    # base_samples = base_dist.sample(n_reps).to(args.device)
    # base_samples = torch.ones(n_reps, n_dim-1).to(args.device)
    # eps = 1e-1
    # base_samples = torch.rand(n_reps, n_dim-1).to(args.device) * (torch.pi - eps) + eps
    # base_samples.requires_grad_(True)

    base_dist = StandardNormal(shape=[n_dim - 1])
    base_samples = base_dist.sample(n_reps).to(args.device)
    # base_samples = torch.ones(n_reps, n_dim-1).to(args.device)
    # eps = 1e-1
    # base_samples = torch.rand(n_reps, n_dim-1).to(args.device) * (torch.pi - eps) + eps
    # base_samples.requires_grad_(True)

    manifold_flow = build_flow_manifold_minimal(n_dim, logabs_jacobian="analytical_block")
    # ambient_samples = torch.randn(n_reps, n_dim).to(args.device)
    # theta_samples, _ = manifold_mapping.forward(ambient_samples)
    # proj_samples, _ = manifold_mapping.inverse(theta_samples)
    # proj_samples = proj_samples.detach()
    # proj_samples.requires_grad_(True)

    grad_jacobian_exact = []
    for i in range(n_reps):
        # data_manifold, logabsdet_exact = manifold_mapping.inverse(base_samples[i:i+1], context)
        data_manifold, logabsdet_exact = manifold_flow._transform.inverse(base_samples[i:i + 1], context)
        # data_manifold, logabsdet_exact = manifold_mapping.forward(proj_samples[i:i+1], context)
        # grad_exact_logdet = torch.autograd.grad(logabsdet_exact, manifold_mapping.parameters())
        grad_exact_logdet = torch.autograd.grad(logabsdet_exact, manifold_flow._transform._transforms[2].parameters())
        grad_jacobian_exact.append(grad_exact_logdet[0])
    breakpoint()
    manifold_flow._transform._transforms[0].logabs_jacobian = "fff"
    # manifold_mapping.logabs_jacobian = "fff"
    grad_jacobian_approx = []
    for n in n_hutchinson_samples:
        manifold_flow._transform._transforms[0].n_hutchinson_samples = n
        # manifold_mapping.n_hutchinson_samples = n
        inner_grads = []
        for i in range(n_reps):
            # data_manifold, logabsdet_approx = manifold_mapping.inverse(base_samples[i:i + 1], context)
            data_manifold, logabsdet_approx = manifold_flow._transform.inverse(base_samples[i:i + 1], context)
            # data_manifold, logabsdet_approx = manifold_mapping.forward(proj_samples[i:i + 1], context)
            # grad_approx_logdet = torch.autograd.grad(logabsdet_approx, manifold_mapping.parameters(), allow_unused=True)
            grad_approx_logdet = torch.autograd.grad(logabsdet_approx, manifold_flow._transform._transforms[2].parameters())
            inner_grads.append(grad_approx_logdet[0])
        grad_jacobian_approx.append(inner_grads)

    diff_mean = []
    diff_std = []
    for n in range(len(n_hutchinson_samples)):
        norms = [torch.norm(grad_jacobian_exact[i] - grad_jacobian_approx[n][i]).item() for i in range(n_reps)]
        norms = np.array(norms)
        print(norms)
        diff_mean.append(norms.mean())
        diff_std.append(norms.std())
    plt.errorbar(x=n_hutchinson_samples, y=diff_mean, yerr=diff_std)
    plt.show()


if __name__ == "__main__":
    main()