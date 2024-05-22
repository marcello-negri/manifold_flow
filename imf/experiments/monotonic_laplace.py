import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import numpy as np
import torch

import pandas as pd

import tqdm
import argparse
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes, make_regression
from sklearn.preprocessing import StandardScaler
from functools import partial
from imf.experiments.architecture import build_circular_cond_flow_l1_manifold, build_cond_flow_reverse, build_simple_cond_flow_l1_manifold
from imf.experiments.utils_manifold import train_regression_cond, generate_samples
from imf.experiments.plots import plot_betas_lambda_fixed_norm, plot_loss, plot_betas_lambda, plot_marginal_likelihood, plot_samples_3d, plot_simplex, to_file
from imf.experiments.datasets import generate_regression_dataset

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=2000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1., help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=20, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=-2, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=2, help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=200, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=5, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
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


def gaussian_kernel(window_size, sigma):
    gauss = torch.tensor([np.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def moving_average(tensor, window_size=5, sigma=1.0):
    ndarr = isinstance(tensor, np.ndarray)
    tensor = torch.tensor(tensor) if ndarr else tensor
    # Create a 1D convolutional kernel for moving average and reshape tensors
    kernel = gaussian_kernel(window_size, sigma)
    kernel = kernel.view(1, 1, -1)
    kernel = kernel.expand((5,5,-1))
    tensor = tensor.permute(1, 2, 0)
    #tensor = tensor.unsqueeze(0)

    smoothed_tensor = torch.nn.functional.conv1d(tensor, kernel, padding=window_size // 2)

    smoothed_tensor = smoothed_tensor.squeeze(0).permute(2, 0, 1)[:tensor.size(-1)]  # Shape back to (n, b, d)
    return smoothed_tensor if not ndarr else smoothed_tensor.numpy()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def synthetic_dataset(n=400, p=10, device='cuda', seed=1234, eps=5e-1):
    torch.manual_seed(seed)
    # parameters used: sample_size = 10, lambda_size = 500, T0=10, Tn=0.01, epochs//20
    X_tensor = (torch.randn(n, p) - .5).to(device)
    # add column of zero to fit intercept
    # X_tensor = torch.hstack((X_tensor, torch.ones((X_tensor.shape[0], 1)).to(device)))
    # beta = (torch.rand(p + 1) * 2 - 1.).to(device)
    beta = (torch.rand(p) * 2 - 1.).to(device) * .5
    # add gaussian noise eps to observations
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

def compute_norm_const(pdf, lambdas, dim, n_points=10_000, extent=1.0):
    with torch.no_grad():
        x_points = torch.rand(lambdas.shape[0], n_points//lambdas.shape[0], dim, device=lambdas.device) * extent
        y_mesh = pdf(x_points, lambdas)
        integral = torch.logsumexp(y_mesh, dim=-1)

        return - integral.sum(-1) + extent*dim - n_points

def laplace(beta, lambdas):
    return torch.exp(torch.clamp(llaplace(beta, lambdas), min=-95))

def llaplace(beta, lambdas):
    return - lambdas * beta.abs().sum(-1)

def power(beta, lambdas, p):
    return laplace(beta, lambdas) ** p

def tanh(beta, lambdas, scale):
    return torch.tanh(scale * laplace(beta, lambdas))

def elu(beta, lambdas, scale):
    return -torch.nn.functional.elu(-laplace(beta, lambdas), alpha=scale)

def selu(beta, lambdas, scale):
    return -torch.nn.functional.selu(-laplace(beta, lambdas), alpha=scale)

def sigmoid(beta, lambdas):
    return torch.sigmoid(laplace(beta, lambdas))-0.499

def sigmoid_offset(beta, lamb_):
    shape_ = beta.shape
    beta_1 = - llaplace(beta, lamb_)
    # prev [0.0,24.0] mult 12.0
    beta_ = beta_1[None,] - torch.tensor([1.0, 12.0], device=beta.device)[:, None, None]
    vals = 2.0 * (torch.sigmoid(8.0 * beta_) - 0.499).sum(0)
    vals = vals + 0.1 * torch.nn.functional.relu(beta_1 - 8.0)
    vals = vals.reshape(shape_[:-1])
    return - vals

def softplus_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)  # - lamb_.unsqueeze(-1) * beta.abs()
    spllap = torch.nn.functional.softplus(- 200.0 * llap - 480.0)
    return - spllap

    # vals = torch.sign(spllap) * spllap ** 2

    # spllap = - torch.nn.functional.softplus(- 80.0 * llap - 160.0)
    # return spllap

    # spllap1 = - torch.nn.functional.softplus(- 2.0 * llap)
    # spllap2 = - torch.nn.functional.softplus(- 48.0 * llap - 96.0)
    # vals = torch.sign(spllap2) * spllap2**2 + spllap1
    # return vals

def lrelu_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)  # - lamb_.unsqueeze(-1) * beta.abs()
    spllap = torch.nn.functional.leaky_relu(- 200.0 * llap - 480)
    return - spllap

def root_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)
    spllap = - torch.sqrt(-llap+1)
    return spllap

def square_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)
    spllap = - torch.pow(-llap+1.0, 4)
    return spllap

def softplus(beta, lambdas):
    return torch.nn.functional.softplus(llaplace(beta, lambdas))-torch.log(torch.tensor(2.0))

def exp(beta, lambdas):
    return torch.exp(laplace(beta, lambdas))

def log_prior_act_laplace(beta, lamb, act, args):
    if args.log_cond: lamb_ = 10 ** lamb
    else: lamb_ = lamb
    # eps = 1e-8

    # q_reshaped_beta = q.view(-1, *len(beta.shape[1:]) * (1,))  # (-1, 1, 1)
    # beta_q = (beta.abs() + eps).pow(q_reshaped_beta)

    # laplace_prior = torch.exp(- torch.clamp(lamb_.unsqueeze(-1) * beta.abs(), max=95))
    # laplace_prior = torch.exp(- lamb_.unsqueeze(-1) * beta.abs())
    if act == "laplace_exact":
        log_const = beta.shape[-1] * torch.log(0.5 * lamb_)
        log_prior = llaplace(beta, lamb_)
        log_prior_DE = log_prior + log_const
    elif act == "sigmoid_offset":
        log_prior = sigmoid_offset(beta, lamb_)
        log_const = 0.0  # compute_norm_const(sigmoid_offset, lamb_, dim=beta.shape[-1]).sum(-1)
        log_prior_DE = log_prior  # + log_const[...,None]
    elif act == "softplus_offset":
        log_prior = softplus_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    elif act == "lrelu_offset":
        log_prior = lrelu_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    elif act == "root_offset":
        log_prior = root_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    elif act == "square_offset":
        log_prior = square_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    elif act == "laplace_approx":
        log_const = 0.0 # compute_norm_const(laplace, lamb_, dim=beta.shape[-1])
        log_prior = llaplace(beta, lamb_)
        log_prior_DE = log_prior + log_const
    elif act[:5] == "power":
        scalar = float(act.split("_")[1])
        log_const = 0.0  # beta.shape[-1] * scalar * torch.log(0.5 * lamb_)
        log_prior = - scalar * llaplace(beta, lamb_)
        log_prior_DE = log_prior + log_const
    elif act == "identity":
        log_prior_DE = torch.log(laplace(beta, lamb_))
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
        log_const = 0.0  # compute_norm_const(sigmoid, lamb_, dim=beta.shape[-1]).sum(-1)
        log_prior = torch.log(sigmoid(beta, lamb_.unsqueeze(-1))).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act == "softplus":
        log_const = 0.0  # compute_norm_const(softplus, lamb_, dim=beta.shape[-1])
        log_prior = torch.log(softplus(- lamb_.unsqueeze(-1), beta)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:4] == "tanh":
        scale = float(act.split("_")[1])
        log_const = 0.0  # compute_norm_const(partial(tanh, scale=scale), lamb_, dim=beta.shape[-1])
        log_prior = torch.log(tanh(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:3] == "elu":
        scale = float(act.split("_")[1])
        log_const = 0.0  # compute_norm_const(partial(elu, scale=scale), lamb_, dim=beta.shape[-1])
        log_prior = torch.log(elu(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:4] == "selu":
        scale = float(act.split("_")[1])
        log_const = 0.0  # compute_norm_const(partial(selu, scale=scale), lamb_, dim=beta.shape[-1])
        log_prior = torch.log(selu(beta, lamb_.unsqueeze(-1), scale)).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act[:3] == "exp":
        shift = float(act.split("_")[1])
        log_prior = llaplace(beta, lamb_)
        log_prior_DE = -torch.exp(torch.clamp(-log_prior+shift, max=6.0))
    elif act[:3] == "log":
        log_prior = torch.log((lamb_.unsqueeze(-1) * beta.abs()+1e-5).sum(-1))
        log_const = 0.0
        log_prior_DE = log_prior + log_const
    else:
        raise ValueError("invalid activation name")

    return log_prior_DE

def log_unnorm_posterior(beta, cond, X, y, sigma, act, use_l, use_p):
    log_likelihood_ = log_likelihood(beta, sigma, X, y) if use_l else X.new_zeros((1))
    log_prior_beta_ = log_prior_act_laplace(beta=beta, lamb=cond, act=act, args=args) if use_p else X.new_zeros((1))

    return log_likelihood_ + log_prior_beta_

def main():
    seed = 666
    set_random_seeds(seed)

    if visualize_all:
        path_print(0)
        group_print(1)
        exit(0)

    args.log_cond = to_use_log_cond
    match monotonic_act:
        case "square_offset":
            args.cond_min = -3.5
            args.cond_max = 0.2
        case "root_offset":
            args.cond_min = -2.5
            args.cond_max = 2.7
        case "laplace_exact":
            if on_manifold:
                args.cond_min = 0.1#-1.
                args.cond_max = 32.0#np.log10(30)
            else:
                args.cond_min = -2.5
                args.cond_max = 1.2
    Tn = 0.01
    ratio_nonzero = 1
    args.datadim = 5
    n_samples = 7
    nn_manifold = on_manifold

    args.use_likelihood = True
    args.use_prior = True

    chosen_norm = 5.0
    use_map_norm_matching = True
    map_its = int(0.5 * args.epochs)

    # the sigma parameter should be tuned depending on the noise in the dataset
    # however, for the final experiment it would be nicer to use a hyperprior (inverse gamma)
    # in that case the likelihood is not Gaussian anymore but rather the student t distribution
    sigma_regr = 4.0  # this parameter needs to be tuned

    X_np, y_np, true_beta = generate_regression_dataset(n_samples=n_samples, n_features=args.datadim, n_non_zero=int(ratio_nonzero * args.datadim), noise_std=sigma_regr)
    X_tensor = torch.tensor(X_np, device=args.device, dtype=torch.float)
    y_tensor = torch.tensor(y_np, device=args.device, dtype=torch.float)
    # X_tensor, y_tensor, X_np, y_np = synthetic_dataset(n=n_samples, p=args.datadim, seed=seed, eps=sigma_regr)

    # sns.pairplot(pd.DataFrame(X_np))
    # plt.show()

    alphas_lasso = np.logspace(args.cond_min, args.cond_max, 200)
    beta_sklearn = np.array(
        [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in
         tqdm.tqdm(alphas_lasso * sigma_regr ** 2 / n_samples)])
    plt.figure(figsize=(14, 14))
    plt.plot(alphas_lasso, beta_sklearn)
    plt.xscale('log')
    plt.show()

    # define target distribution
    monotone_activations = ["laplace_exact", "power_100", "power_0.01", "power_4", "power_2", "power_1", "power_0.5", "power_0.25"]
    monotone_activations = monotone_activations + ["tanh_20.0", "tanh_10.0", "tanh_1.0", "elu_20.0", "elu_10.0", "elu_1.0"]
    monotone_activations = monotone_activations + ["sigmoid", "softplus", "softplus_offset"]  # "sigmoid_offset" still needs fixing
    true_beta_post = []
    true_beta_prior = []
    true_beta_tensor = torch.tensor(true_beta, device=args.device, dtype=torch.float)
    test_cond = X_tensor.new_ones(1) * (args.cond_min+args.cond_max)/2
    for ac in monotone_activations:
        true_beta_post.append(log_unnorm_posterior(beta=true_beta_tensor, cond=test_cond, X=X_tensor, y=y_tensor, sigma=sigma_regr, act=ac, use_l=args.use_likelihood, use_p=args.use_prior))
        true_beta_prior.append(log_prior_act_laplace(beta=true_beta_tensor, lamb=test_cond, act=ac, args=args))

    #monotonic_act = "laplace_exact"
    #monotonic_act = "softplus_offset"
    #monotonic_act = "sigmoid_offset"
    target_distr = partial(log_unnorm_posterior, X=X_tensor, y=y_tensor, sigma=sigma_regr, act=monotonic_act, use_l=args.use_likelihood, use_p=args.use_prior)


    # build model
    # conditional flow on lambda
    args.architecture = "ambient"
    if nn_manifold:
        args.norm = chosen_norm
        #flow = build_simple_cond_flow_l1_manifold(args, n_layers=3, n_hidden_features=64, n_context_features=64, clamp_theta=False)
        args.n_layers = 3
        args.n_hidden_features = 64
        args.n_context_features = 64
        flow = build_circular_cond_flow_l1_manifold(args, star_like=True)
    else:
        flow = build_cond_flow_reverse(args, clamp_theta=False)

    # conditional flow with fixed norm
    # flow = build_circular_cond_flow_l1_manifold(args)
    out_name = ("mani_" if nn_manifold else "") + monotonic_act + ".csv"
    if use_map_norm_matching:
        # train model
        flow.train()
        iter_norm = args.epochs
        args.epochs = map_its
        flow, loss, loss_T = train_regression_cond(model=flow, log_unnorm_posterior=target_distr, args=args, tn=Tn, manifold=False)
        args.epochs = iter_norm
        flow.eval()
    #    plot_loss(loss)
        samples, cond, kl = generate_samples(flow, args, n_lambdas=2000, cond=True, log_unnorm_posterior=target_distr,
                                             manifold=False, context_size=2000, sample_size=5, n_iter=1)
        plot_betas_lambda(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                          min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', true_coeff=None)
        if on_manifold:
            our_cond = chosen_norm
        else:
            beta_norms = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            beta_norms_mask = beta_norms > chosen_norm
            bindex = beta_norms_mask.astype(int).sum(-1)
            our_cond = cond[bindex-1]
        print("our_cond", our_cond)
        to_file(samples.reshape(-1, args.datadim), "mapall_" + out_name)
        to_file(cond, "mapcall_" + out_name)

        if nn_manifold:
            args.norm = chosen_norm
            # flow = build_simple_cond_flow_l1_manifold(args, n_layers=3, n_hidden_features=64, n_context_features=64, clamp_theta=False)
            args.n_layers = 3
            args.n_hidden_features = 64
            args.n_context_features = 64
            flow = build_circular_cond_flow_l1_manifold(args, star_like=True)
        else:
            flow = build_cond_flow_reverse(args, clamp_theta=False)

    flow.train()
    flow, loss, loss_T = train_regression_cond(model=flow, log_unnorm_posterior=target_distr, args=args, tn=args.Tn, manifold=False)
    flow.eval()

    # evaluate model
    samples, cond, kl = generate_samples(flow, args, n_lambdas=2000, cond=True, log_unnorm_posterior=target_distr,
                                         manifold=False, context_size=20, sample_size=1000, n_iter=100)
    plot_betas_lambda(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                      min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', true_coeff=None)
    opt_cond = plot_marginal_likelihood(kl_sorted=kl, cond_sorted=cond, args=args)
    print("optimal condition: ", opt_cond.item())

    to_file(samples.reshape(-1,args.datadim), "all_" + out_name)
    to_file(cond, "call_" + out_name)
    to_file(X_np, "xnp_" + out_name)
    to_file(y_np, "ynp_" + out_name)

    if not use_map_norm_matching:
        if on_manifold:
            our_cond = chosen_norm
        else:
            beta_norms = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            beta_norms_mask = beta_norms > chosen_norm
            bindex = beta_norms_mask.astype(int).sum(-1)
            our_cond = cond[bindex - 1]
        print("our_cond", our_cond)

    posterior_samples, psslogprob = flow.sample_and_log_prob(1000, context=(np.log10(our_cond) if args.log_cond else our_cond)*torch.ones(1,1,device=args.device))
    posterior_samples = posterior_samples.detach().cpu().numpy()
    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])
    posterior_samples_cl = 0.0 * posterior_samples[:,0:1] + np.arange(posterior_samples.shape[-1])[None,:]
    post_s_norm = np.linalg.norm(posterior_samples, axis=-1, ord=1)
    # plot_samples_3d(posterior_samples, s=10, alpha=1.0)
    # plot_simplex(posterior_samples, dim_3=False, args=args, alpha=0.5, s=8)
    # plot_simplex(posterior_samples, dim_3=True, args=args, alpha=0.7, s=10)
    # plot_simplex(posterior_samples, dim_3=True, shift3to2=True, args=args, alpha=0.5, s=8)
    ps = pd.DataFrame(data={'x_ind_coeff': posterior_samples_cl.reshape(-1), 'y': posterior_samples.reshape(-1)})
    sns.boxplot(data=ps, x="x_ind_coeff", y="y")
    plt.show()
    ps2 = pd.DataFrame(data={'n_single_lambda': post_s_norm})
    # sns.boxplot(data=ps2, y="n_single_lambda")
    # plt.show()
    sns.violinplot(data=ps2, x="n_single_lambda")
    plt.show()
    cond_norms = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
    ps3 = pd.DataFrame(data={'x_cond': cond, 'avg norm of all samples': cond_norms})
    sns.lineplot(data=ps3, x='x_cond', y='avg norm of all samples')
    plt.grid()
    plt.show()

    to_file(posterior_samples, "fnorm_" + out_name)

    print("done")


name_map = {"laplace_exact": "laplace", "square_offset": "square lap",
                "mani_laplace_exact": "objective", "root_offset": "root lap"}
def path_print(variant):
    import os
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from mpl_toolkits.mplot3d import Axes3D
    nlambdas = 2000
    subsample_for_path = 20
    files = [f for f in os.listdir("./") if f.endswith(".csv") and (f.startswith("all_") or f.startswith("mapall_"))]
    names = [f[4:-4] if f.startswith("all_") else "m"+f[7:-4] for f in files]

    sigma_regr = 4.0
    d = [pd.read_csv(f).loc[:, '0':] for f in files]
    dms = {}
    dma = []
    name_d_dic = dict(zip(names, d))
    for di, on, map_data in zip(d, names, [f.startswith("mapall_") for f in files]):
        if map_data:
            on = on[1:]
        n = name_map[on]
        print(n if not map_data else "map " + n)
        samples = di.to_numpy().reshape(nlambdas,-1,5) if not map_data else di.to_numpy().reshape(nlambdas,-1,5)
        cond = pd.read_csv(("call_" if not map_data else "mapcall_") + on + ".csv").loc[:, '0':].to_numpy()[...,0]
        X_np = pd.read_csv("xnp_" + on + ".csv").loc[:, '0':].to_numpy()
        y_np = pd.read_csv("ynp_" + on + ".csv").loc[:, '0':].to_numpy()[...,0]
        of = "data/out/" + n
        di['s'] = n
        dm = di.melt(id_vars='s', var_name='coeff', value_name='value')
        if name_map["mani_laplace_exact"] in n:
            dms[n] = samples
        if not map_data:
            normmeans = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            pdt = pd.DataFrame(normmeans)
            dms[n] = pdt.melt(value_name="norm")
            dms[n]["source"] = n
            mapsamples = pd.read_csv("mapall_" + on + ".csv").loc[:, '0':].to_numpy().reshape(nlambdas, -1, 5)
            mapnormmeans = np.linalg.norm(mapsamples, axis=-1, ord=1).mean(-1)
            norms_mask = mapnormmeans > 5.0 if not name_map["mani_laplace_exact"] in n else mapnormmeans < 5.0
            mapindex = norms_mask.astype(int).sum(-1)
            mapcond = pd.read_csv("mapcall_" + on + ".csv").loc[:, '0':].to_numpy()[...,0]
            targetcond = mapcond[mapindex]
            index = (cond < targetcond).astype(int).sum(-1)
            # this approach can lead to small deviations from the desired norm. saves model storing
            # mostly unnoticeable (ie norm==4.99 instead of 5.0)
            index = index if not name_map["mani_laplace_exact"] in n else index - 1
            pda = pd.DataFrame(np.linalg.norm(samples[index], axis=-1, ord=1))
            pda = pda.melt(value_name="norm")
            pda["source"] = n
            dma.append(pda)

            ssamples = samples[index]
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ssamples = ssamples.reshape(-1, 5)
            dat = ssamples[:, :3]
            # this is a hacky way to show 5 dim samples on the manifold in 3 dim space
            dat = np.linalg.norm(ssamples, axis=-1, ord=1, keepdims=True) * dat / np.linalg.norm(dat, axis=-1, ord=1,
                                                                                           keepdims=True)
            # the norm manifold
            vertices = 5.0 * np.array(
                [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            faces = [[vertices[0], vertices[1], vertices[2]],
                     [vertices[0], vertices[2], vertices[3]],
                     [vertices[0], vertices[3], vertices[4]],
                     [vertices[0], vertices[4], vertices[1]],
                     [vertices[5], vertices[1], vertices[2]],
                     [vertices[5], vertices[2], vertices[3]],
                     [vertices[5], vertices[3], vertices[4]],
                     [vertices[5], vertices[4], vertices[1]]]

            # Define the faces of the diamond
            poly3d = Poly3DCollection(faces, facecolors='#789cb330', linewidths=0)  # 1, edgecolors='#789cb390')
            ax.add_collection3d(poly3d)
            ax.plot([0, 0, 0, 5, 0], [-5, 0, 5, 0, -5], [0, 5, 0, 0, 0], c='#789cb390', zorder=1)
            ax.plot([0, -5, 0, 5, 0, 0], [-5, 0, 0, 0, 0, -5], [0, 0, 5, 0, -5, 0], c='#789cb390', zorder=1)

            # the samples of the norm
            ax.scatter(dat[:, 1], -dat[:, 0], -dat[:, 2], c=np.abs(dat[:, 0]) + np.abs(dat[:, 2]),
                       cmap='Blues', s=8, zorder=5, marker="o", edgecolors='none')
            ax.grid(False)
            ax._axis3don = False

            plt.savefig(of + "_with_mani.pdf", format='pdf')
            plt.show()

        match variant:
            case "trace_norm" | 0:
                if map_data:
                    smoothed = samples#moving_average(samples, 20, 5.0)
                    plot_betas_lambda(samples=smoothed[::subsample_for_path], lambdas=cond[::subsample_for_path], X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                                      min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1,
                                      gt='linear_regression', true_coeff=None, name=of + ("_betas" if not map_data else "_betas_map"))
                else:
                    plot_betas_lambda(samples=samples[::subsample_for_path], lambdas=cond[::subsample_for_path], X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                                      min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1,
                                      gt='linear_regression', true_coeff=None, name=of+("_betas" if not map_data else "_betas_map"))
                sns.boxplot(data=dm, y="coeff", x="value", fliersize=0)
                plt.title(n)
                plt.savefig(of + "_norms.pdf", format='pdf')
                plt.show()

    of = "data/out/"
    match variant:
        case "trace_norm" | 0:
            plt.figure(figsize=(4, 7))
            # dma = pd.concat([dm for k, dm in dms.items() if not (name_map["mani_laplace_exact"] in k)], ignore_index=True)
            dma = [dma[1], dma[2], dma[0]]
            dma = pd.concat(dma, ignore_index=True)
            sns.violinplot(data=dma, x="norm", y="source")
            plt.tight_layout()
            plt.savefig(of + "all_std_norms.pdf", format='pdf')
            plt.show()


def group_print(variant):
    import os
    with_obj = False
    files = [f for f in os.listdir("./") if f.endswith(".csv") and f.startswith("fnorm") and (with_obj or not "mani" in f)]
    names = [f[6:-4] for f in files]
    names = [name_map[n] if n in name_map else n for n in names]

    d = [pd.read_csv(f).loc[:, '0':] for f in files]
    for di, n in zip(d, names):
        di['s'] = n
    d = [d[2],d[0],d[1]]
    dm = [di.melt(id_vars='s', var_name='coeff', value_name='value') for di in d]
    d = pd.concat(dm)

    of = "data/out/"
    match variant:
        case "box" | 0:
            sns.boxplot(x='coeff', y='value', hue='s', data=d)
            plt.show()
        case "box_no_outs" | 1:
            palette_tab10 = sns.color_palette("tab10", 10)
            palette_blue = list(sns.light_palette(palette_tab10[0], n_colors=3))[::-1][:3]
            palette_salmon = list(sns.light_palette(palette_tab10[1], n_colors=6))[::-1][:1]
            palette = palette_blue + palette_salmon if with_obj else palette_blue
            if with_obj:
                pal = {name_map["laplace_exact"]: palette[0], name_map["square_offset"]: palette[1],
                       name_map["root_offset"]: palette[2], name_map["mani_laplace_exact"]: palette[3]}
            else:
                pal = {name_map["laplace_exact"]: palette[0], name_map["square_offset"]: palette[1],
                       name_map["root_offset"]: palette[2]}#, name_map["mani_laplace_exact"]: palette[3]}
            plt.figure(figsize=(8, 4))
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern']
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=2.0)
            # , palette=pal
            sns.boxplot(x='coeff', y='value', hue='s', palette="bright", data=d, fliersize=0)
            sns.set(font_scale=1.7)
            plt.legend(ncol=4 if with_obj else 3, loc="upper left")
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.tight_layout()
            plt.savefig(of + "compare.pdf", format='pdf', bbox_inches='tight')
            plt.show()
        case "violin" | 2:
            sns.violinplot(x='coeff', y='value', hue='s', data=d)
            #plt.ylim(-5,5)
            plt.show()


monotonic_act = ""
on_manifold = True
visualize_all = False
if __name__ == "__main__":
    monotonic_act = "laplace_exact"
    to_use_log_cond = False
    main()
    to_use_log_cond = True
    on_manifold = False
    main()
    monotonic_act = "square_offset"
    main()
    monotonic_act = "root_offset"
    main()
    visualize_all = True
    main()
