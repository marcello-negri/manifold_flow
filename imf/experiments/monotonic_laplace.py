import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

from rpy2 import robjects
import pandas as pd

import tqdm
import argparse
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from functools import partial
from imf.experiments.architecture import build_circular_cond_flow_l1_manifold, build_cond_flow_reverse
from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=2_000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1, help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=50, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=-2, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=2, help='minimum value of conditional variable')
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

def synthetic_dataset(n=400, p=10, device='cuda', seed=1234):
    torch.manual_seed(seed)
    # parameters used: sample_size = 10, lambda_size = 500, T0=10, Tn=0.01, epochs//20
    X_tensor = (torch.randn(n, p) - .5).to(device)
    # add column of zero to fit intercept
    # X_tensor = torch.hstack((X_tensor, torch.ones((X_tensor.shape[0], 1)).to(device)))
    # beta = (torch.rand(p + 1) * 2 - 1.).to(device)
    beta = (torch.rand(p) * 2 - 1.).to(device) * .5
    # add gaussian noise eps to observations
    eps = 5e-1
    y_tensor = torch.normal(X_tensor @ beta, torch.ones(n).to(device) * eps)

    X_np = X_tensor.cpu().detach().numpy()
    y_np = y_tensor.cpu().detach().numpy()

    # compute regression parameters
    reg = LinearRegression().fit(X_np, y_np)
    r2_score = reg.score(X_np, y_np)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y_np - X_np @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")

    return X_tensor, y_tensor, X_np, y_np


def scikit_regression(n_samples, n_features, noise, seed=1234):
    scaler = StandardScaler()
    X_np, y_np = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=seed)
    X_np = scaler.fit_transform(X_np)
    y_np = scaler.fit_transform(y_np.reshape(-1, 1))
    X_tensor = torch.from_numpy(X_np).float().to(device)
    y_tensor = torch.from_numpy(y_np.reshape(-1)).float().to(device)

    # compute regression parameters
    reg = LinearRegression().fit(X_np, y_np)
    r2_score = reg.score(X_np, y_np)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y_np - X_np @ reg.coef_.T)))
    print(f"Sigma regression: {sigma_regr:.4f}")

    return X_tensor, y_tensor, X_np, y_np

def log_likelihood(beta, sigma, X, y):
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * np.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def log_prior_laplace(beta, lamb):

    lamb_ = 10 ** lamb
    log_prior_DE = - (lamb_) * beta.abs().sum(-1)
    log_prior_const = beta.shape[-1] * torch.log(0.5 * lamb_)

    return log_prior_DE + log_prior_const

def compute_norm_const(pdf, lambdas, n_points=10_000):
    x_mesh = torch.linspace(0, 10, n_points).repeat(lambdas.shape[0], 1)
    x_mesh = x_mesh.to(lambdas.device) / lambdas
    y_mesh = pdf(x_mesh, lambdas)
    integral = torch.trapezoid(y=y_mesh, x=x_mesh, dim=-1) * 2

    return -torch.log(integral).reshape(-1,1)

def laplace(beta, lambdas):
    return torch.exp(- torch.clamp(lambdas * beta.abs(), max=95))

def power(beta, lambdas, p):
    return laplace(beta, lambdas) ** p

def tanh(beta, lambdas, scale):
    return torch.tanh(scale * laplace(beta, lambdas))

def elu(beta, lambdas, scale):
    return -torch.nn.functional.elu(-laplace(beta, lambdas), alpha=scale)

def selu(beta, lambdas, scale):
    return -torch.nn.functional.selu(-laplace(beta, lambdas), alpha=scale)

def sigmoid(beta, lambdas):
    return torch.sigmoid(laplace(beta, lambdas))-0.5

def softplus(beta, lambdas):
    return torch.nn.functional.softplus(- torch.clamp(lambdas * beta.abs(), max=95))-torch.log(torch.tensor(2.0))

def exp(beta, lambdas):
    return torch.exp(laplace(beta, lambdas))

def log_prior_act_laplace(beta, lamb, act, args):
    if args.log_cond: lamb_ = 10 ** lamb
    else: lamb_ = lamb
    # eps = 1e-8

    # q_reshaped_beta = q.view(-1, *len(beta.shape[1:]) * (1,))  # (-1, 1, 1)
    # beta_q = (beta.abs() + eps).pow(q_reshaped_beta)

    laplace_prior = torch.exp(- torch.clamp(lamb_.unsqueeze(-1) * beta.abs(), max=95))
    # laplace_prior = torch.exp(- lamb_.unsqueeze(-1) * beta.abs())
    if act == "laplace_exact":
        log_const = beta.shape[-1] * torch.log(0.5 * lamb_)
        log_prior = - lamb_ * beta.abs().sum(-1)
        log_prior_DE = log_prior + log_const
    elif act == "laplace_approx":
        log_const = beta.shape[-1] * compute_norm_const(laplace, lamb_)
        log_prior = - lamb_ * beta.abs().sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:5] == "power":
        scalar = float(act.split("_")[1])
        log_const = beta.shape[-1] * scalar * torch.log(0.5 * lamb_)
        log_prior = - scalar * lamb_ * beta.abs().sum(-1)
        log_prior_DE = log_prior + log_const
    elif act == "identity":
        log_prior_DE = torch.log(laplace_prior).sum(-1)
        # print(lamb_)
        # print("lamb beta: ", - lamb_.unsqueeze(-1) * beta.abs())
        # print("log exp lamb beta 1: ", torch.log(torch.exp(- lamb_.unsqueeze(-1) * beta.abs()) + eps))
        # print("log exp lamb beta 2: ", torch.log(laplace_prior+eps))
        # print("sum lamb beta: ", (- lamb_.unsqueeze(-1) * beta.abs()).sum(-1))
        # print("sum log exp lamb beta: ", log_prior_DE)
    # elif act =="log_sigmoid":
    #     m = torch.nn.LogSigmoid()
    #     log_prior_DE = (m(laplace_prior)-torch.log(torch.tensor(0.5))).sum(-1)
    elif act == "sigmoid":
        log_const = compute_norm_const(sigmoid, lamb_).sum(-1)
        log_prior = torch.log(sigmoid(beta, lamb_.unsqueeze(-1))).sum(-1)
        log_prior_DE = log_prior +log_const
    elif act == "softplus":
        # log_const = beta.shape[-1] * compute_norm_const(softplus, lamb_)
        log_prior = torch.log(softplus(- lamb_.unsqueeze(-1), beta)).sum(-1)
        log_prior_DE = log_prior #+ log_const
    elif act[:4] == "tanh":
        scale = float(act.split("_")[1])
        log_const = beta.shape[-1] * compute_norm_const(partial(tanh, scale=scale), lamb_)
        log_prior = torch.log( tanh(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:3] == "elu":
        scale = float(act.split("_")[1])
        log_const = beta.shape[-1] * compute_norm_const(partial(elu, scale=scale), lamb_)
        log_prior = torch.log(elu(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:4] == "selu":
        scale = float(act.split("_")[1])
        log_const = beta.shape[-1] * compute_norm_const(partial(selu, scale=scale), lamb_)
        log_prior = torch.log(selu(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:3] == "exp":
        shift = float(act.split("_")[1])
        log_prior_DE = torch.log(torch.exp(laplace_prior+shift)-torch.exp(torch.tensor(shift))).sum(-1)
    else:
        raise ValueError("invalid activation name")

    return log_prior_DE

def log_unnorm_posterior(beta, cond, X, y, sigma, act):
    log_likelihood_ = log_likelihood(beta, sigma, X, y)
    log_prior_beta_ = log_prior_act_laplace(beta=beta, lamb=cond, act=act, args=args)

    return log_likelihood_ + log_prior_beta_

def main():
    seed = 666
    set_random_seeds(seed)

    args.log_cond = True
    args.cond_min = 0
    args.cond_max = 3
    args.datadim = 10
    n_samples = 50
    X_tensor, y_tensor, X_np, y_np = synthetic_dataset(n=n_samples, p=args.datadim, seed=seed)

    sns.pairplot(pd.DataFrame(X_np))
    plt.show()

    # the sigma parameter should be tuned depending on the noise in the dataset
    # however, for the final experiment it would be nicer to use a hyperprior (inverse gamma)
    # in that case the likelihood is not Gaussian anymore but rather the student t distribution
    sigma_regr = 0.5 # this parameter needs to be tuned

    alphas_lasso = np.logspace(args.cond_min, args.cond_max, 200)
    beta_sklearn = np.array(
        [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in
         tqdm.tqdm(alphas_lasso * sigma_regr ** 2 / n_samples)])
    plt.figure(figsize=(14, 14))
    plt.plot(alphas_lasso, beta_sklearn)
    plt.xscale('log')
    plt.show()

    # define target distribution
    monotone_activations = ["laplace_exact", "power_4", "power_2", "power_1", "power_0.5", "power_0.25"]
    monotone_activations = ["tanh_20.0", "tanh_10.0", "tanh_1.0", "elu_20.0", "elu_10.0", "elu_1.0"]

    monotonic_act = 'laplace_exact'
    target_distr = partial(log_unnorm_posterior, X=X_tensor, y=y_tensor, sigma=sigma_regr, act=monotonic_act)

    # build model
    # conditional flow on lambda
    args.architecture = "ambient"
    flow = build_cond_flow_reverse(args, clamp_theta=False)

    # conditional flow with fixed norm
    # flow = build_circular_cond_flow_l1_manifold(args)

    # train model
    flow.train()
    flow, loss, loss_T = train_regression_cond(model=flow, log_unnorm_posterior=target_distr, args=args, manifold=False)
    plot_loss(loss)

    # evaluate model
    samples, cond, kl = generate_samples(flow, args, cond=True, log_unnorm_posterior=target_distr,
                                         manifold=False, context_size=10, sample_size=100, n_iter=50)
    plot_betas_lambda(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                      min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', true_coeff=None)
    opt_cond = plot_marginal_likelihood(kl_sorted=kl, cond_sorted=cond, args=args)


if __name__ == "__main__":
    main()