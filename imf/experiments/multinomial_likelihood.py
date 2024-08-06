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

def dirichlet_prior(beta: torch.Tensor, alpha: torch.Tensor, args):
    if args.log_cond: alpha_ = 10 ** alpha
    else: alpha_ = alpha

    K = beta.shape[-1]
    log_const = torch.lgamma(alpha_ * K) - K * torch.lgamma(alpha_)
    log_prior = (alpha_ - 1) * torch.log(beta).sum(-1)

    return log_const + log_prior

def dirichlet_prior_beta(beta: torch.Tensor, alphas: torch.Tensor):
    log_const = torch.lgamma(alphas).sum(-1) - torch.lgamma(alphas.sum(-1))
    log_prior = ((alphas - 1) * torch.log(beta)).sum(-1)

    return log_const + log_prior

def multinomial_log_lik(X, beta):
    log_lik = (X * torch.log(beta)).sum(-1)
    log_const = torch.lgamma(X.sum()+1) - torch.lgamma(X+1).sum()

    return log_lik + log_const
def unnorm_posterior_multinomial(beta, alphas, cond=None, X=None):
    log_lik = multinomial_log_lik(beta=beta, X=X)
    # alpha = torch.zeros_like(beta[...,0]) if args.log_cond else torch.ones_like(beta[...,0])
    # log_prior = dirichlet_prior(beta=beta, alpha=alpha, args=args)
    log_prior = dirichlet_prior_beta(beta=beta, alphas=alphas)
    return log_lik + log_prior

def generate_multinomial_simplex(n, d):
    probs = np.random.rand(d)
    probs /= probs.sum()
    X = np.random.multinomial(n, probs)  # RIGHT

    return X

def plot_posterior_boxplot(samples_gt, samples_mcmc, samples_flow=None, idx=None, jumps=None):
    import seaborn as sns

    assert samples_gt.shape == samples_mcmc.shape

    if idx is not None:
        samples_gt = samples_gt[:, idx]
        samples_mcmc = samples_mcmc[:, idx]
        if samples_flow is not None:
            samples_flow = samples_flow[:, idx]

    df_list_gt = [pd.DataFrame(samples_gt[:, i], columns=["value"]).assign(coefficient=f"{idx[i]+1 if idx else i}") for i in
                  range(samples_gt.shape[-1])]
    gt_df = pd.concat(df_list_gt, ignore_index=True, sort=False)
    gt_df["sampler"] = "gt"

    df_list_mcmc = [pd.DataFrame(samples_mcmc[:, i], columns=["value"]).assign(coefficient=f"{idx[i]+1 if idx else i}") for i in
                    range(samples_mcmc.shape[-1])]
    mcmc_df = pd.concat(df_list_mcmc, ignore_index=True, sort=False)
    mcmc_df["sampler"] = "mcmc"

    if samples_flow is not None:
        df_list_flow = [pd.DataFrame(samples_flow[:, i], columns=["value"]).assign(coefficient=f"{idx[i]+1 if idx else i}") for i in
                        range(samples_flow.shape[-1])]
        flow_df = pd.concat(df_list_flow, ignore_index=True, sort=False)
        flow_df["sampler"] = "flow"
        samples_df = pd.concat([mcmc_df, flow_df, gt_df], ignore_index=True, sort=False)
    else:
        samples_df = pd.concat([mcmc_df, gt_df], ignore_index=True, sort=False)

    sns.boxplot(data=samples_df, x="coefficient", y="value", hue="sampler", showfliers=False)

    if jumps is not None:
        for jump in jumps:
            plt.axvline(x=jump - 0.5, color='red', linestyle='--', linewidth=1)

    plt.savefig("./mcmc_flow_boxplot.pdf", dpi=300)
    plt.show()

def mh_sampler(burnin=10000, chains=1000, iterations=20000, subsample=2000, dim=10, log_p=None, dtype = torch.float64, device="cuda"):
    init_alpha = torch.tensor([1.0], device=device, dtype=dtype)  # TODO adjust

    init = get_dirichlet_samples(init_alpha, chains, dim)[0]
    # test = Test(init, alphas, steps_sizes)
    from imf.experiments.dirichlet_sampler import Markov_dirichlet_given_logp

    if dim < 10:  # these values are rough guesstimates
        steps = [1e-1, 1.0]
        alphas = [1.0]
    elif dim < 25:
        steps = [5e-2, 1.0]#, 0.001, 1.0]
        alphas = [1.0]
    elif dim < 60:
        steps = [1e-4, 1.0]#, 0.02, 0.05, 1.0]
        alphas = [1.0]
    t_0 = 1
    centered_props = False
    steps_0 = [1e-1] #[min(np.sqrt(t_0) * step, 1.0) for step in steps]#[0.05, 0.2, 1.0] #[min(np.sqrt(t_0) * step, 1.0) for step in steps]
    print("steps ", steps)
    print("steps_0 ", steps_0)
    print("alphas ", alphas)
    print("proposals centered: ", centered_props)
    alphas = torch.tensor(alphas, device=device, dtype=dtype)
    steps = torch.tensor(steps, device=device, dtype=dtype)
    steps_0 = torch.tensor(steps_0, device=device, dtype=dtype)
    test = Markov_dirichlet_given_logp(init, alphas, steps, log_p=log_p, centered_props=centered_props, T_0=t_0, step_sizes_0=steps_0, temp_its_0=0.1 * burnin, temp_its=burnin * 0.4)

    it = test.iterator()
    startt = os.times()
    for i in range(burnin):
        next(it)
        if i % 1000 == 0:
            print(f"burnin {i} accrate {test.accrate} curA {test.get_a()} temp {test.get_temp()}")

    # here all the samples are stored on the cpu
    samples = torch.zeros((iterations // subsample, chains, dim), dtype=dtype)
    for i in range(iterations):
        if i % subsample == 0:
            samples[i//subsample] = next(it)[0].detach().cpu()
        else:
            next(it)
        if i % 1000 == 0:
            print(f"iteration {i} accrate {test.accrate} curA {test.get_a()} temp {test.get_temp()}")
    endt = os.times()
    dt = endt.user + endt.system - startt.user - startt.system
    print(f"total used time {dt} for sampling")

    return samples

def main():
    args.log_cond = True
    prior_name = "dirichlet"
    set_random_seeds(args.seed)

    # dimensions
    d = 15
    # gt setting
    n = 9
    alphas = 0.3 * torch.ones((1, d), device=args.device)
    num_gt_samples = 1000
    target_multinomial = True
    # mcmc setting
    chains = 2000
    burnin = 50000
    iterations = 100
    subsample = 100
    # flow setting
    args.epochs = 2000
    args.n_context_samples = 500
    sample_size = 500
    n_iter = 100
    # viz settings:
    idx = [0,1,2,4,9,d-1]
    jumps = [3,4,5]

    X_np = generate_multinomial_simplex(n=n, d=d)
    X_tensor = torch.from_numpy(X_np).float().to(device=args.device)
    args.datadim = X_tensor.shape[0]
    gt_samples = np.random.dirichlet(alpha=X_np + alphas.detach().cpu().numpy().ravel(), size=num_gt_samples)
    dim_sort = np.argsort(gt_samples.mean(0))  # ascending
    dim_sort = dim_sort[::-1].copy()  # descending

    if target_multinomial:
        log_unnorm_posterior = partial(unnorm_posterior_multinomial, X=X_tensor, alphas=alphas)#, flow_prior=flow_prior)
    else:
        from dirichlet_utils import Struct
        log_unnorm_posterior = partial(dirichlet_prior, alpha=alphas[:,0], args=Struct({'log_cond':False}))
        gt_samples = get_dirichlet_samples(alphas[0,0], num_gt_samples, d).detach().cpu().numpy()
        #gt_samples = np.random.dirichlet(alphas.detach().cpu().numpy().ravel(), size=num_gt_samples)

    #import time
    #time.sleep(1)
    with torch.no_grad():
        mh_samples = mh_sampler(burnin=burnin, chains=chains, iterations=iterations, subsample=subsample, dim=args.datadim,
                                log_p=log_unnorm_posterior, dtype=torch.float64)
    mh_samples = mh_samples.reshape(-1, args.datadim).to(dtype=torch.float32)
    plot_posterior_boxplot(gt_samples[:,dim_sort], mh_samples[:num_gt_samples,dim_sort], idx=idx, jumps=jumps)

    #print("sleep")
    #time.sleep(10000)

    # build model
    flow = build_circular_cond_flow_l1_manifold(args)

    # train model
    flow.train()
    startt = os.times()
    flow, loss, loss_T = train_regression_cond(flow, log_unnorm_posterior, args=args, manifold=False)
    endt = os.times()
    dt = endt.user + endt.system - startt.user - startt.system
    print(f"total used time {dt} for flow training")
    plot_loss(loss)

    # evaluate model
    flow.eval()
    samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=log_unnorm_posterior, manifold=False, context_size=1, sample_size=sample_size, n_iter=n_iter)
    plot_betas_lambda_fixed_norm(samples=samples, lambdas=cond, dim=X_np.shape[-1], conf=0.95, n_plots=1, log_scale=args.log_cond)

    # shuffles samples
    shuff_mh = torch.randperm(mh_samples.shape[0])
    mh_samples = mh_samples[shuff_mh].detach().cpu().numpy()
    samples = samples.reshape(-1, d)
    np.random.shuffle(samples)

    plot_posterior_boxplot(samples_gt=gt_samples[:,dim_sort], samples_mcmc=mh_samples[:num_gt_samples,dim_sort], samples_flow=samples[:num_gt_samples,dim_sort], idx=idx, jumps=jumps)


if __name__ == "__main__":
    main()




