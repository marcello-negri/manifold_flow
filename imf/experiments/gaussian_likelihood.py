import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import torch
import os
import argparse

from functools import partial

from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.datasets import load_diabetes_dataset, generate_regression_dataset, generate_regression_dataset_positive_coeff
from imf.experiments.architecture import build_cond_flow_reverse, build_simple_circular_cond_flow_l1_manifold, build_simple_cond_flow_l1_manifold, build_circular_cond_flow_l1_manifold
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_sparsity_distr, plot_cumulative_returns_singularly, plot_sparsity_patterns, plot_betas_lambda, plot_marginal_likelihood, plot_returns, plot_cumulative_returns
from imf.experiments.dirichlet_sampler import Test, get_dirichlet_samples

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=2_000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=5., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=100, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=-1, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=1., help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=1_000, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=1, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
parser.add_argument('--beta', metavar='be', type=float, default=1.0, help='p of the lp norm')


# MODEL PARAMETERS
parser.add_argument("--n_layers", metavar='nl', type=int, default=5, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=128, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--n_context_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])

args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def gaussian_log_likelihood(beta: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):
    # implements Gaussian log-likelihood beta ~ Normal (X@beta, sigma^2 ID)
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args):
    if args.log_cond: alpha_ = 10 ** alpha
    else: alpha_ = alpha

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior

def unnorm_posterior_cond_likelihood(beta, cond=None, X=None, y=None, args=None):
    if cond is None: cond = torch.zeros_like(beta[...,0]) if args.log_cond else torch.ones_like(beta[...,0])
    sigmas = 10 ** cond if args.log_cond else cond
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigmas, X=X, y=y)
    alpha = torch.zeros_like(cond) if args.log_cond else torch.ones_like(cond)
    log_prior = dirichlet_prior(beta=beta, alpha=alpha, args=args)
    return log_lik + log_prior

def unnorm_posterior_fixed_sigma(beta, sigma, cond=None, X=None, y=None, args=None):
    sigmas = torch.ones_like(beta[...,0]) * sigma
    log_lik = gaussian_log_likelihood(beta=beta, sigma=sigmas, X=X, y=y)
    alpha = torch.zeros_like(sigmas) if args.log_cond else torch.ones_like(sigmas)
    log_prior = dirichlet_prior(beta=beta, alpha=alpha, args=args)
    return log_lik + log_prior

def multinomial_log_lik(X, beta):
    log_lik = (X * torch.log(beta)).sum(-1)
    log_const = torch.lgamma(X.sum()+1) - torch.lgamma(X+1).sum()

    return log_lik + log_const
def unnorm_posterior_multinomial(beta, cond=None, X=None):
    log_lik = multinomial_log_lik(beta=beta, X=X)
    alpha = torch.zeros_like(beta[...,0]) if args.log_cond else torch.ones_like(beta[...,0])
    log_prior = dirichlet_prior(beta=beta, alpha=alpha, args=args)
    return log_lik + log_prior

def generate_regression_simplex (n, d, sigma=0.1, seed=1234):
    np.random.seed(seed)
    X_np = np.random.randn(n, d)
    beta = np.random.rand(d)
    beta /= beta.sum() # normalize the weights on the simplex
    y_np = X_np @ beta + np.random.randn(n) * sigma

    return X_np, y_np

def generate_multinomial_simplex(n, d):
    probs = np.random.rand(d)
    probs /= probs.sum()
    X = np.random.multinomial(n, probs)

    return X


def plot_posterior_boxplot(samples_mcmc, samples_flow):
    import seaborn as sns

    assert samples_mcmc.shape == samples_flow.shape

    df_list_mcmc = [pd.DataFrame(samples_mcmc[:, i], columns=["value"]).assign(coefficient=f"{i}") for i in
                    range(samples_mcmc.shape[-1])]
    mcmc_df = pd.concat(df_list_mcmc, ignore_index=True, sort=False)
    mcmc_df["sampler"] = "mcmc"

    df_list_flow = [pd.DataFrame(samples_flow[:, i], columns=["value"]).assign(coefficient=f"{i}") for i in
                    range(samples_flow.shape[-1])]
    flow_df = pd.concat(df_list_flow, ignore_index=True, sort=False)
    flow_df["sampler"] = "flow"

    samples_df = pd.concat([mcmc_df, flow_df], ignore_index=True, sort=False)
    sns.boxplot(data=samples_df, x="coefficient", y="value", hue="sampler")
    plt.savefig("./mcmc_flow_boxplot.pdf", dpi=300)
    plt.show()

def mh_sampler(burnin=10000, chains=1000, iterations=20000, subsample=2000, dim=10, alphas_proposal=[0.1, 0.5, 1.0], steps_proposal=[0.1,0.2,0.5,1.0], log_p=None, dtype = torch.float64, device="cuda"):
    init_alpha = torch.tensor([1.0], device=device, dtype=dtype)  # TODO adjust

    init = get_dirichlet_samples(init_alpha, chains, dim)[0]
    alphas = torch.tensor(alphas_proposal, device=device, dtype=dtype)
    steps_sizes = torch.tensor(steps_proposal, device=device, dtype=dtype)
    # test = Test(init, alphas, steps_sizes)
    from imf.experiments.dirichlet_sampler import Markov_dirichlet_given_logp
    test = Markov_dirichlet_given_logp(init, alphas, steps_sizes, log_p=log_p)

    it = test.iterator()
    startt = os.times()
    for i in range(burnin):
        next(it)
        if i % 500 == 0:
            print(f"burnin {i} accrate {test.accrate}")

    # here all the samples are stored on the cpu
    samples = torch.zeros((iterations // subsample, chains, dim), dtype=dtype)
    for i in range(iterations):
        if i % subsample == 0:
            samples[i//subsample] = next(it)[0].detach().cpu()
        else:
            next(it)
        if i % 500 == 0:
            print(f"iteration {i} accrate {test.accrate}")
    endt = os.times()
    dt = endt.user + endt.system - startt.user - startt.system
    print(f"total used time {dt} for sampling")

    return samples

def main():
    just_load = False # False: train model and save samples. True: load the samples
    args.log_cond = True
    prior_name = "dirichlet"
    set_random_seeds(args.seed)

    # load data
    sigma_true = 0.09
    dim = 50
    chains = 1000
    n = 20
    iterations = 20000
    X_np, y_np = generate_regression_simplex(n=n, d=dim, sigma=sigma_true)
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    y_tensor = torch.from_numpy(y_np).float().to(device=args.device)
    args.datadim = X_tensor.shape[1]

    log_unnorm_posterior = partial(unnorm_posterior_cond_likelihood, X=X_tensor, y=y_tensor, args=args)
    log_unnorm_posterior_mh = partial(unnorm_posterior_fixed_sigma, sigma=sigma_true, X=X_tensor, y=y_tensor, args=args)

    if dim < 6:  # these values are rough guesstimates
        steps = [0.05, 0.2, 1.0]
        alphas = [0.1, 1.0]
    elif dim < 15:
        steps = [0.01, 0.05, 0.2, 1.0]
        alphas = [0.1, 1.0]
    elif dim < 60:
        steps = [0.0001, 0.003, 0.1, 1.0]
        alphas = [0.1, 1.0]
    else:
        steps = [0.00001, 0.001, 0.1, 1.0]
        alphas = [0.1, 1.0]
    mh_samples = mh_sampler(burnin=10000, chains=chains, iterations=iterations, subsample=2000, steps_proposal=steps,
                            alphas_proposal=alphas, dim=args.datadim, log_p=log_unnorm_posterior_mh, dtype=torch.float32)
    mh_samples = mh_samples.reshape(-1, args.datadim)

    if not just_load:
        # build model
        flow = build_circular_cond_flow_l1_manifold(args)

        # train model
        flow.train()
        flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False)
        plot_loss(loss)

        # evaluate model
        flow.eval()
        samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=1, sample_size=1000, n_iter=500)
        plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=X_np.shape[-1], conf=0.95, n_plots=1, log_scale=args.log_cond)

        opt_cond, opt_idx = plot_marginal_likelihood(kl, cond, args)
        rand_shuffle = torch.randperm(mh_samples.shape[0])
        mh_samples_shuffled = mh_samples[rand_shuffle]
        plot_posterior_boxplot(mh_samples_shuffled[:samples.shape[1]], samples[opt_idx])
        print(f"Optimal sigma via MLL: {opt_cond:.3f} (true: {sigma_true:.3f})")
        np.save(f'data_{prior_name}.npy', samples)
    else:
        samples_uniform = np.load('data_uniform.npy')  # load
        samples_dirichlet = np.load('data_dirichlet.npy')  # load

        plot_sparsity_distr(samples_uniform, samples_dirichlet, X_np, y_np, threshold=0.01, n_bins=25)


if __name__ == "__main__":
    main()



