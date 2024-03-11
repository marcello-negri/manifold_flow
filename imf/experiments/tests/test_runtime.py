import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse


from enflows.distributions import Uniform
from enflows.distributions.uniform import UniformSphere
from enflows.flows.base import Flow
from enflows.transforms.base import InverseTransform, CompositeTransform
from enflows.transforms.injective import ScaleLastDim, LearnableManifoldFlow, SphereFlow, LpManifoldFlow

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
# parser.add_argument("--bruteforce", action="store_true", help="compute the rectangular jacobian term bruteforce")
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")

# DATASETS PARAMETERS
parser.add_argument("--data_folder", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/manifold_flow/imf/experiments/data")
parser.add_argument("--dataset", type=str, default="uniform", choices=["vonmises_fisher","vonmises_fisher_mixture", "uniform", "uniform_checkerboard", "vonmises_fisher_mixture_spiral"])
parser.add_argument('--datadim', metavar='d', type=int, default=3, help='number of dimensions')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.1, help='std of the isotropic noise in the data')
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

from normflows.flows.neural_spline.wrapper import CircularAutoregressiveRationalQuadraticSpline
from enflows.transforms.linear import ScalarScale, ScalarShift

def build_minimal_flow_forward(args):
    transformation_layers = []
    base_dist = UniformSphere(shape=[args.datadim - 1])
    #
    # torch_pi = torch.tensor(np.pi, device=args.device)
    # torch_zero = torch.zeros(1, device=args.device)
    # base_dist = Uniform(shape=[args.datadim - 1], low=torch_zero, high=torch_pi)

    if args.learn_manifold:
        manifold_mapping = LearnableManifoldFlow(n=args.datadim - 1, max_radius=2., logabs_jacobian=args.logabs_jacobian)
    else:
        if args.dataset == "lp_uniform":
            manifold_mapping = LpManifoldFlow(norm=1, p=args.beta, logabs_jacobian=args.logabs_jacobian)
        else:
            manifold_mapping = SphereFlow(n=args.datadim - 1, r=1., logabs_jacobian=args.logabs_jacobian)

    transformation_layers.append(manifold_mapping)

    # transformation_layers.append(
    #     ScaleLastDim(scale=2)
    # )

    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow


def build_flow_manifold_minimal(args, device='cuda'):
    torch_pi = torch.tensor(np.pi).to(device)
    torch_zero = torch.zeros(1).to(device)
    # torch_one = torch.ones(1).to(device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_pi, high=torch_pi)
    # base_dist = Uniform(shape=[args.datadim - 1], low=torch_zero, high=torch_pi)
    base_dist = Uniform(shape=[args.datadim - 1], low=torch_zero, high=torch_pi)

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        InverseTransform(
            ScaleLastDim()
        )
    )

    if args.learn_manifold:
        manifold_mapping = LearnableManifoldFlow(n=args.datadim - 1, max_radius=2., logabs_jacobian=args.logabs_jacobian)
    else:
        manifold_mapping = SphereFlow(n=args.datadim - 1, r=1., logabs_jacobian=args.logabs_jacobian)

    transformation_layers.append(
        InverseTransform(
            manifold_mapping
        )
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow

def main():

    import time
    from datetime import timedelta
    import tqdm
    import pandas as pd
    import seaborn as sns

    n_reps = 5
    n_sums = 10
    n_dims = 5

    dimensions = np.linspace(10, 5000, n_dims, dtype='int')

    np.random.seed(1234)
    torch.manual_seed(1234)
    # torch.set_default_dtype(torch.float64)
    # data_base = data_base.double()

    if os.path.isfile('results_runtime.csv'):
        runtime_df = pd.read_csv('results_runtime.csv')
    else:
        context=None
        results = []
        # args.learn_manifold = True
        for dim in dimensions:
            args.datadim = int(dim)
            unif_sphere_dist = UniformSphere(shape=[args.datadim - 1])
            data_base = unif_sphere_dist.sample(1)
            data_base.requires_grad_(True)
            # data_base = torch.tensor(torch.rand((1, args.datadim - 1)), requires_grad=True).to('cuda')
            # data_base = data_base * 2 - 1
            # data_base[0, -1] *= 2
            for logabs_jacobian in ["cholesky","analytical_sm","analytical_lu"]:
                args.logabs_jacobian = logabs_jacobian
                print(args)
                # build flow
                flow = build_minimal_flow_forward(args)
                data_manifold, logabsdet = flow._transform.inverse(data_base, context)

                for _ in tqdm.tqdm(range(n_reps)):
                    # torch.cuda.synchronize()
                    start_time = time.monotonic()
                    for _ in range(n_sums):
                        data_manifold, logabsdet = flow._transform.inverse(data_base, context)
                        # import gc
                        # print("="*100)
                        # print(torch.cuda.memory_summary())
                        # for obj in gc.get_objects():
                        #     try:
                        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        #             print(type(obj), obj.size())
                        #     except:
                        #         pass
                    # torch.cuda.synchronize()
                    end_time = time.monotonic()
                    time_diff = timedelta(seconds=end_time - start_time)
                    time_diff_seconds= str(time_diff).split(":")[-1]
                    print(time_diff)

                    entry_dict = dict(datadim=dim, logabs_jacobian=logabs_jacobian, learn_manifold=args.learn_manifold,
                                      runtime=float(time_diff_seconds))
                    results.append(entry_dict)

        runtime_df = pd.DataFrame(results)
        runtime_df.to_csv("results_runtime.csv", index=False)


    def load_results_csv(logabs_jacobian, dimensions, n_sums, n_reps):
        n_dims = dimensions.shape[0]
        n_per_dim = n_reps * n_sums
        results_pd = pd.read_csv(f"results_{logabs_jacobian}_manifold.csv", header=None)
        results_pd = results_pd.drop(index=[i*n_per_dim for i in range(n_dims)]).reset_index(drop=True)
        results_pd.columns = ["runtime"]
        results_pd["logabs_jacobian"] = logabs_jacobian
        results_pd["datadim"] = 0
        for i in range(n_dims):
            results_pd.loc[i * n_per_dim:(i + 1) * n_per_dim, 'datadim'] = dimensions[i]

        return results_pd

    plt.figure(figsize=(12, 5), dpi=80)
    ax = sns.boxplot(x="datadim", y="runtime", hue="logabs_jacobian", data=runtime_df)  # RUN PLOT
    plt.show()

    # results_cholesky_pd = load_results_csv("cholesky", dimensions=dimensions, n_sums=n_sums, n_reps=n_reps)
    # results_analytical_pd = load_results_csv("analytical", dimensions=dimensions, n_sums=n_sums, n_reps=n_reps)
    # results_pd = pd.concat([results_cholesky_pd, results_analytical_pd])
    #
    # plt.figure(figsize=(12, 5), dpi=80)
    # ax = sns.boxplot(x="datadim", y="runtime", hue="logabs_jacobian", data=results_pd)  # RUN PLOT
    # plt.show()
    #
    # plt.figure(figsize=(12, 5), dpi=80)
    # ax = sns.boxplot(x="datadim", y="runtime", hue="logabs_jacobian", data=runtime_df)  # RUN PLOT
    # plt.show()

    breakpoint()

if __name__ == "__main__":
    main()