import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from imf.experiments.utils_manifold import train_model_forward, train_model_reverse, generate_samples
from imf.experiments.utils_manifold import  define_model_name
from imf.experiments.architecture import build_flow_forward, build_flow_reverse
from imf.experiments.plots import plot_icosphere, plot_loss, plot_samples, plot_pairwise_angle_distribution, plot_angle_distribution, plot_samples_pca
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
parser.add_argument("--logabs_jacobian", type=str, default="analytical", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")


# DATASETS PARAMETERS
parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher", "vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral", "lp_uniform"])
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.01, help='std of the isotropic noise in the data')
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

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    # load dataset and samples
    dataset = create_dataset(args=args)
    train_data_np, test_data_np = dataset.load_samples(overwrite=True)
    train_data = torch.from_numpy(train_data_np).float().to(args.device)
    test_data = torch.from_numpy(test_data_np).float().to(args.device)

    plot_samples(train_data_np)

    from enflows.transforms.injective.utils import SimpleNN
    import time
    from enflows.transforms.injective.utils import spherical_to_cartesian_torch, cartesian_to_spherical_torch

    model = SimpleNN(input_size=args.datadim-1, hidden_size=200, output_size=1).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=800, shuffle=True)
    start_time = time.monotonic()
    model.train()
    try:
        for epoch in range(args.epochs):
            for i, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                # add small noise to the dataset to prevent overfitting
                batch_data.requires_grad_(True)
                # project data on the manifold
                thetas = cartesian_to_spherical_torch(batch_data)[:,:-1]
                radius = model(thetas)
                spherical = torch.cat((thetas, radius), dim=-1)
                cartesian = spherical_to_cartesian_torch(spherical)
                loss = torch.mean(torch.abs(cartesian-batch_data))

                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    thetas_v = cartesian_to_spherical_torch(test_data)[:, :-1]
                    radius_v = model(thetas_v)
                    spherical_v = torch.cat((thetas_v, radius_v), dim=-1)
                    cartesian_v = spherical_to_cartesian_torch(spherical_v)
                    loss_v = torch.mean(torch.abs(cartesian_v - test_data))

                print(f"Loss epoch {epoch}: train {loss.item():.5f} - test {loss_v.item():.4f}")
    except KeyboardInterrupt:
        print("interrupted..")

    thetas = cartesian_to_spherical_torch(train_data)[:,:-1]
    radius = model(thetas)
    spherical = torch.cat((thetas, radius), dim=-1)
    cartesian = spherical_to_cartesian_torch(spherical)
    plot_samples(cartesian.detach().cpu().numpy())

    thetas = cartesian_to_spherical_torch(test_data)[:, :-1]
    radius = model(thetas)
    spherical = torch.cat((thetas, radius), dim=-1)
    cartesian = spherical_to_cartesian_torch(spherical)
    plot_samples(cartesian.detach().cpu().numpy())


if __name__ == "__main__":
    main()