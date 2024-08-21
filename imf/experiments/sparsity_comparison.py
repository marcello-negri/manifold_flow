import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset
from imf.experiments.architecture import build_cond_flow_reverse
from imf.experiments.plots import plot_betas_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood
from imf.experiments.plots import plot_sparsity_distr

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

def t_student_log_likelihood(beta: torch.Tensor, a0: torch.Tensor, b0: torch.Tensor, X: torch.Tensor, y: torch.Tensor):

    N = X.shape[0]
    log_lk = -(a0 + 0.5 * N) * torch.log(1 + 0.5 * (y - beta @ X.T).square().sum(-1) / b0 )
    log_lk_const = torch.lgamma(a0 + 0.5 * N) - torch.lgamma(a0) - 0.5 * N * torch.log(2 * np.pi * b0)

    return log_lk + log_lk_const

def plot_sparsity(samples1, samples2, norm=1):
    import pandas as pd
    import seaborn as sns

    # df_flow = pd.DataFrame(samples1)
    values = samples1.flatten()
    column_numbers = np.tile(np.arange(samples1.shape[1]), samples1.shape[0])
    df_flow = pd.DataFrame({'value': values, 'coefficient': column_numbers})
    df_flow["method"] = "flow"

    values = samples2.flatten()
    column_numbers = np.tile(np.arange(samples2.shape[1]), samples2.shape[0])
    df_manifold = pd.DataFrame({'value': values, 'coefficient': column_numbers})
    df_manifold["method"] = "manifold"

    df = pd.concat([df_flow, df_manifold], axis=0)
    sns.boxplot(data=df, y="value", x="coefficient", hue="method")
    plt.show()

    # breakpoint()

def evaluate_likelihood(samples, sigma, X, Y, iter, device="cuda"):
    import tqdm
    likelihood = []
    n_samples = samples.shape[0]
    samples_per_iter = n_samples // iter
    for i in tqdm.tqdm(range(iter)):
        samples_torch = torch.from_numpy(samples[i*samples_per_iter:(i+1)*samples_per_iter]).to(device).float()
        lik = gaussian_log_likelihood(beta=samples_torch, sigma=sigma, X=X, y=Y)
        likelihood.append(lik.detach().cpu().numpy())

    return np.r_[likelihood].ravel()

from collections import Counter
import seaborn as sns

def plot_sparsity_likelihood(samples1, samples2, likelihood1, likelihood2, threshold=None, min_lik=None, max_lik=None, n_bins=25):

    def keys_values(samples, threshold):
        non_zero_counts = np.sum(np.abs(samples) > threshold, axis=-1)
        counts_dict = Counter(non_zero_counts)
        keys = np.array(list(counts_dict.keys()))
        values = np.array(list(counts_dict.values()))
        sort_idx = np.argsort(keys)
        # breakpoint()
        values_ = values / values.sum()
        return keys[sort_idx], values_[sort_idx]

    if threshold is None:
        threshold = 1. / samples1.shape[-1] # anything more than uniform


    lik1, lik2 = likelihood1.flatten(), likelihood2.flatten()

    min_bin = max(lik1.min(), lik2.min())
    max_bin = min(lik1.max(), lik2.max())

    if min_lik is None:
        min_lik = (max_bin + min_bin) / 2

    presel_idx1 = lik1 > min_lik
    presel_idx2 = lik2 > min_lik

    flatten_samples1 = samples1.reshape(-1, samples1.shape[-1])
    flatten_samples2 = samples2.reshape(-1, samples2.shape[-1])

    flatten_samples1 = flatten_samples1[presel_idx1]
    flatten_samples2 = flatten_samples2[presel_idx2]
    lik1 = lik1[presel_idx1]
    lik2 = lik2[presel_idx2]

    min_bin = max(lik1.min(), lik2.min())
    max_bin = min(lik1.max(), lik2.max())

    bin_edges = np.linspace(min_bin, max_bin, num=n_bins)
    idx1 = np.digitize(lik1, bin_edges)
    idx2 = np.digitize(lik2, bin_edges)

    # n_dim = n_bins//3 - 1
    n_dim = n_bins
    n_rows = int(np.sqrt(n_dim))
    if n_rows ** 2 != n_dim: n_rows += 1
    n_cols = n_rows

    # reshaped_samples = samples1.reshape(-1, samples1.shape[-1])
    # non_zero_counts = np.sum(reshaped_samples > 0.01, axis=-1)
    # H, xedges, yedges = np.histogram2d(non_zero_counts, 10*(flatten_mse1-flatten_mse1.min())/(flatten_mse1.max() - flatten_mse1.min()))
    # plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    average1, average2 = [], []
    with sns.axes_style("whitegrid"):
        fig, axs = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)
        # fig1, axs1 = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)
        # fig2, axs2 = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)

        for i_r in np.arange(n_rows):
            for i_c in np.arange(n_rows):
                try:
                    samples_range1 = flatten_samples1[idx1 == (n_rows * i_r + i_c + 1)]
                    keys1, values1 = keys_values(samples_range1, threshold=threshold)
                    samples_range2 = flatten_samples2[idx2 == (n_rows * i_r + i_c + 1)]
                    keys2, values2 = keys_values(samples_range2, threshold=threshold)
                    axs[i_r, i_c].bar(x=keys1, height=values1, alpha=0.5, label='flow')
                    axs[i_r, i_c].bar(x=keys2, height=values2, alpha=0.5, label='manifold')
                    avg1 = (values1*keys1).sum()
                    avg2 = (values2*keys2).sum()
                    average1.append(avg1)
                    average2.append(avg2)
                    # breakpoint()
                    axs[i_r, i_c].set_title(f'lik={bin_edges[n_rows * i_r + i_c ]:.1f} avg={avg1:.2f}/{avg2:.2f}')
                    # breakpoint()
                    if i_r + i_c == 0:
                        axs[i_r, i_c].legend()

                    # axs1[i_r, i_c].imshow(samples_range1[:50].T, cmap="Blues")
                    # axs2[i_r, i_c].imshow(samples_range2[:50].T, cmap="Oranges")
                except:
                    pass

        # fig.savefig("non_zero_distr.pdf", bbox_inches='tight')
        # fig1.figure.savefig("non_zero_uniform.pdf", bbox_inches='tight')
        # fig2.figure.savefig("non_zero_dirichlet.pdf", bbox_inches='tight')
        plt.tight_layout()
        plt.show()

        avg1 = np.mean(average1, axis=0)
        avg2 = np.mean(average2, axis=0)
        print(avg1, avg2)
        print((avg1-avg2)/avg1)
def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    create_directories()

    set_random_seeds(args.seed)
    create_directories()
    args.datadim = 10

    X_tensor, y_tensor, X_np, y_np = load_diabetes_dataset(device="cpu")

    # X_np, y_np, true_beta = generate_regression_dataset(n_samples=20, n_features=args.datadim, n_non_zero=8, noise_std=1)
    # X_np, y_np, true_beta = generate_regression_dataset(n_samples=100, n_features=100, n_non_zero=80, noise_std=1)

    # X_tensor = torch.from_numpy(X_np).float()
    # y_tensor = torch.from_numpy(y_np).float()
    sigma = torch.tensor(.7)

    samples_flow = np.load(f'./samples/samples_regression_lambda_{args.datadim}_{args.beta:.2f}_opt.npy')
    samples_manifold = np.load(f'./samples/samples_regression_manifold_{args.datadim}_{args.beta:.2f}_opt.npy')
    plot_sparsity(samples_flow, samples_manifold)
    breakpoint()
    samples_flow_full = np.load(f'./samples/samples_regression_lambda_{args.datadim}_{args.beta:.2f}.npy')
    samples_manifold_full = np.load(f'./samples/samples_regression_manifold_{args.datadim}_{args.beta:.2f}.npy')
    lik_flow = gaussian_log_likelihood(beta=torch.from_numpy(samples_flow_full).float(), sigma=sigma, X=X_tensor,y=y_tensor)
    lik_manifold = gaussian_log_likelihood(beta=torch.from_numpy(samples_manifold_full).float(), sigma=sigma, X=X_tensor,y=y_tensor)

    breakpoint()
    plot_sparsity_likelihood(samples_flow_full, samples_manifold_full, lik_flow, lik_manifold, threshold=0.01, n_bins=25, min_lik=-200)


if __name__ == "__main__":
    main()