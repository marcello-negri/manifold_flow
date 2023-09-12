import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time
from datetime import timedelta

import tqdm
from enflows.distributions import StandardNormal, Uniform, DiagonalNormal, MOG
from enflows.transforms import Sigmoid, ScalarScale, ScalarShift, RandomPermutation, MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.flows.base import Flow
from enflows.transforms.injective import FixedNorm, ConditionalFixedNorm, ConstrainedAnglesSigmoid, ResidualNetInput, ClampedAngles
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def log_likelihood(beta, sigma, X, y):

    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def log_prior_beta(beta, lamb):

    lamb_ = 10 ** lamb
    log_prior_DE = - (lamb_) * beta.abs().sum(-1)
    log_prior_const = beta.shape[-1] * torch.log(0.5 * lamb_)

    return log_prior_DE + log_prior_const

def log_unnorm_posterior(beta, X, y, sigma, lamb):

    log_likelihood_ = log_likelihood(beta, sigma, X, y)
    log_prior_beta_ = log_prior_beta(beta, lamb)

    return log_likelihood_ + log_prior_beta_

# conditional flow on manifold
def build_cond_flow_manifold(flow_dim, q, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    base_dist = StandardNormal(shape=[flow_dim - 1])

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim - 1))

        transformation_layers.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features,
                                             context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )

        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim - 1)
            )
        )

    transformation_layers.append(
        InverseTransform(
            ConstrainedAnglesSigmoid(temperature=1, learn_temperature=False)
        )
    )

    # transformation_layers.append(
    #     InverseTransform(
    #         ClampedAngles(eps=1e-3)
    #     )
    # )

    transformation_layers.append(
        InverseTransform(
            ConditionalFixedNorm(q=q)
        )
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=context_features, hidden_features=256,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(device)

    return flow

def train_cond_manifold_model(model, X, y, sigma, epochs=2_001, lr=1e-3, sample_size=1, context_size=1_000, norm_min=-1,
                              norm_max=1., T0=5, Tn=1e-2, iter_per_cool_step=20, device="cuda", **kwargs, ):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # set up cooling schedule
    num_iter = epochs // iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=T0, Tn=Tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss, loss_T = [], []
    try:
        start_time = time.monotonic()
        for epoch in range(epochs):
            T = cooling_function(epoch // (epochs / num_iter))
            optimizer.zero_grad()

            rand_norm = torch.rand(context_size).to(device)
            # log_uniform_norm = 10 ** (rand_norm * (norm_max_exp - norm_min_exp) + norm_min_exp).view(-1, 1)
            uniform_norm = (rand_norm * (norm_max - norm_min) + norm_min).view(-1, 1)

            q_samples, q_log_prob = model.sample_and_log_prob(num_samples=sample_size, context=uniform_norm)
            if torch.any(torch.isnan(q_samples)): breakpoint()

            log_lik = log_likelihood(beta=q_samples, sigma=sigma, X=X, y=y)
            kl_div = torch.mean(q_log_prob - log_lik / T)
            kl_div.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            loss.append(torch.mean(q_log_prob - log_lik).cpu().detach().numpy())
            loss_T.append(torch.mean(q_log_prob - log_lik / T).cpu().detach().numpy())
            print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T

def sample_beta(model, X, y, sigma, norm_min, norm_max, context_size=10, sample_size=100, n_iter=100, device='cuda'):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    sample_list, kl_list, norm_list = [], [], []
    eps = 1e-7
    for _ in tqdm.tqdm(range(n_iter)):
        rand_norm = torch.rand(context_size).to(device)
        uniform_norm = (rand_norm * (norm_max - norm_min) + norm_min).view(-1, 1)
        posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=uniform_norm)
        sample_list.append(posterior_samples.cpu().detach().numpy())
        norm_list.append(uniform_norm.view(-1).cpu().detach().numpy())
        log_lik = log_likelihood(beta=posterior_samples, sigma=sigma, X=X, y=y)
        kl_div = log_probs_samples - log_lik
        kl_list.append(kl_div.cpu().detach().numpy())

    sample_list, norm_list, kl_list = np.concatenate(sample_list, 0), np.concatenate(norm_list, 0), np.concatenate(
        kl_list, 0)

    norm_sorted_idx = norm_list.argsort()
    samples_sorted, norm_sorted, kl_sorted = sample_list[norm_sorted_idx], norm_list[norm_sorted_idx], kl_list[
        norm_sorted_idx]

    return samples_sorted, norm_sorted, kl_sorted

def plot_betas_norm(samples_sorted, norm_sorted, X_np, y_np, norm=1, a=0.95, n_plots=1, folder_name='./'):
    if norm == 2:
        alphas_ridge = np.logspace(-2, 4, 2000)
        beta_sklearn = np.array(
            [Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_ridge)])
    else:
        alphas_lasso = np.logspace(-4, 2, 2000)
        beta_sklearn = np.array(
            [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_lasso)])

    sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)

    # sklearn_norm /= sklearn_norm.max()
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_sklearn[sklearn_sorted_idx]

    l_quant = np.quantile(samples_sorted, 1 - a, axis=1)
    sample_mean = np.mean(samples_sorted, axis=1)
    r_quant = np.quantile(samples_sorted, a, axis=1)
    norm_sorted_ = norm_sorted / norm_sorted.max()

    n_lines = X_np.shape[-1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == X_np.shape[-1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(norm_sorted, sample_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(norm_sorted, l_quant[:, j], r_quant[:, j], alpha=0.2, facecolor=color)
                ax.plot(sklearn_norm, sklearn_sorted[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            # ax.set_xscale('log')
            plt.xlabel(f'$||\beta||_{{{norm}}}$', fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            # plt.xscale('log')
            plt.ylim(-1, 1)
            # plt.savefig(f"{folder_name}flow_manifold_norm_T_1_{j}.png", dpi=200, bbox_inches='tight')
            plt.show()

# conditional flow with Laplace prior
def build_cond_flow(flow_dim, q, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    base_dist = StandardNormal(shape=[flow_dim])

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))

        transformation_layers.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features,
                                             context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )

        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=context_features, hidden_features=256,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(device)

    return flow

def train_cond_model(model, X, y, sigma, epochs=2_001, lr=1e-3, sample_size=1, context_size=1_000, lambda_min_exp=-2,
                     lambda_max_exp=2., T0=5, Tn=1e-2, iter_per_cool_step=20, device="cuda", **kwargs, ):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # set up cooling schedule
    num_iter = epochs // iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=T0, Tn=Tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss, loss_T = [], []
    try:
        start_time = time.monotonic()
        for epoch in range(epochs):
            T = cooling_function(epoch // (epochs / num_iter))
            optimizer.zero_grad()

            rand_lambda = torch.rand(context_size).to(device)
            uniform_lambda = (rand_lambda * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)

            q_samples, q_log_prob = model.sample_and_log_prob(num_samples=sample_size, context=uniform_lambda)
            if torch.any(torch.isnan(q_samples)): breakpoint()

            log_posterior = log_unnorm_posterior(beta=q_samples, X=X, y=y, sigma=sigma, lamb=uniform_lambda)
            kl_div = torch.mean(q_log_prob - log_posterior / T)
            kl_div.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            loss.append(torch.mean(q_log_prob - log_posterior).cpu().detach().numpy())
            loss_T.append(torch.mean(q_log_prob - log_posterior / T).cpu().detach().numpy())
            print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T

def sample_beta_exp(model, X, y, sigma, lambda_min_exp, lambda_max_exp, context_size=10, sample_size=100, n_iter=100,
                    device='cuda'):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    sample_list, kl_list, lambda_list = [], [], []

    for _ in tqdm.tqdm(range(n_iter)):
        rand_lambda = torch.rand(context_size).to(device)
        uniform_lambda = (rand_lambda * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
        posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=uniform_lambda)
        sample_list.append(posterior_samples.cpu().detach().numpy())
        lambda_list.append((10 ** uniform_lambda).view(-1).cpu().detach().numpy())
        log_lik = log_likelihood(beta=posterior_samples, sigma=sigma, X=X, y=y)
        kl_div = log_probs_samples - log_lik
        kl_list.append(kl_div.cpu().detach().numpy())

    sample_list, lambda_list, kl_list = np.concatenate(sample_list, 0), np.concatenate(lambda_list, 0), np.concatenate(
        kl_list, 0)

    lambda_sorted_idx = lambda_list.argsort()
    samples_sorted, lambda_sorted, kl_sorted = sample_list[lambda_sorted_idx], lambda_list[lambda_sorted_idx], kl_list[
        lambda_sorted_idx]

    return samples_sorted, lambda_sorted, kl_sorted

def plot_betas_lambda(samples, lambdas, X_np, y_np, norm=1, a=0.95, n_plots=1, folder_name='./'):
    if norm == 2:
        alphas_ridge = np.logspace(-2, 4, 2000)
        beta_sklearn = np.array(
            [Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_ridge)])
    else:
        alphas_lasso = np.logspace(-4, 2, 2000)
        beta_sklearn = np.array(
            [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_lasso)])
        beta_sklearn_ = np.array([Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
                                  for alpha in tqdm.tqdm(lambdas / 2 / X_np.shape[0])])

    sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)
    # sklearn_norm /= sklearn_norm.max()
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_sklearn[sklearn_sorted_idx]

    all_samples = samples.reshape(-1, X_np.shape[-1])
    all_samples_norms = np.abs(all_samples).sum(-1)
    bins = np.linspace(0, 2, 51)
    bins_midpoint = 0.5 * (bins[1:] + bins[:-1])
    digitized = np.digitize(all_samples_norms, bins)
    n_per_bin, _ = np.histogram(all_samples_norms, bins)
    min_n_per_bin = min(n_per_bin)
    # min_n_per_bin = -1
    bin_means = np.array([all_samples[digitized == i][:min_n_per_bin].mean(0) for i in range(1, len(bins))])
    bin_l_quant = np.array(
        [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.05, axis=0) for i in range(1, len(bins))])
    bin_r_quant = np.array(
        [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.95, axis=0) for i in range(1, len(bins))])

    n_plots = 2
    n_lines = X_np.shape[-1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == X_np.shape[-1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(bins_midpoint, bin_means[:, j], alpha=0.7, linewidth=1.5, c=color)
                ax.fill_between(bins_midpoint, bin_l_quant[:, j], bin_r_quant[:, j], alpha=0.2, facecolor=color)
                ax.plot(sklearn_norm, sklearn_sorted[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)
            plt.ylim(-1, 1)
            plt.xlabel(r'$||\beta||_{}$'.format(norm), fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(f"{folder_name}beta_norm_T_1_{j}.png", dpi=200, bbox_inches='tight')
            plt.show()

    l_quant = np.quantile(samples, 1 - a, axis=1)
    sample_mean = np.mean(samples, axis=1)
    r_quant = np.quantile(samples, a, axis=1)
    n_lines = X_np.shape[-1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == X_np.shape[-1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(lambdas, sample_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambdas, l_quant[:, j], r_quant[:, j], alpha=0.2, facecolor=color)
                ax.plot(lambdas, beta_sklearn_[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            # ax.set_xscale('log')
            plt.xlabel(f'$||\beta||_{{{norm}}}$', fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xscale('log')
            # plt.savefig(f"{folder_name}beta_norm_T_001_.png", dpi=200, bbox_inches='tight')
            plt.show()

    samples_norm = np.array([all_samples[digitized == i][:min_n_per_bin] for i in range(1, len(bins))])
    return bins, samples_norm, bin_l_quant, bin_means, bin_r_quant

def plot_betas_lambda_old(samples, lambdas, X_np, y_np, norm=1, a=0.95, n_plots=1, folder_name='./'):
    if norm == 2:
        alphas_ridge = np.logspace(-2, 4, 2000)
        beta_sklearn = np.array(
            [Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_ridge)])
    else:
        alphas_lasso = np.logspace(-4, 2, 2000)
        beta_sklearn = np.array(
            [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_lasso)])
        beta_sklearn_ = np.array([Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
                                  for alpha in tqdm.tqdm(lambdas / 2 / X_np.shape[0])])

    sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)
    # sklearn_norm /= sklearn_norm.max()
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_sklearn[sklearn_sorted_idx]

    l_quant = np.quantile(samples, 1 - a, axis=1)
    sample_mean = np.mean(samples, axis=1)
    r_quant = np.quantile(samples, a, axis=1)

    samples_norm = np.power(np.power(np.abs(sample_mean), norm).sum(-1), 1 / norm)
    samples_sorted_idx = samples_norm.argsort()
    norms_sorted = samples_norm[samples_sorted_idx]
    samples_sorted = sample_mean[samples_sorted_idx]
    r_quant_sorted = r_quant[samples_sorted_idx]
    l_quant_sorted = l_quant[samples_sorted_idx]

    n_lines = X_np.shape[-1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == X_np.shape[-1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(norms_sorted, samples_sorted[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(norms_sorted, l_quant_sorted[:, j], r_quant_sorted[:, j], alpha=0.2, facecolor=color)
                ax.plot(sklearn_norm, sklearn_sorted[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            # ax.set_xscale('log')
            plt.xlabel(f'$||\beta||_{{{norm}}}$', fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            # plt.xscale('log')
            plt.ylim(-1, 1)
            plt.savefig(f"{folder_name}flow_norm_T_001_{j}.png", dpi=200, bbox_inches='tight')
            plt.show()

    n_lines = X_np.shape[-1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == X_np.shape[-1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(lambdas, sample_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambdas, l_quant[:, j], r_quant[:, j], alpha=0.2, facecolor=color)
                ax.plot(lambdas, beta_sklearn_[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            # ax.set_xscale('log')
            plt.xlabel(f'$||\beta||_{{{norm}}}$', fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xscale('log')
            # plt.savefig(f"{folder_name}beta_norm_T_001_.png", dpi=200, bbox_inches='tight')
            plt.show()

def gen_cooling_schedule(T0, Tn, num_iter, scheme):
    def cooling_schedule(t):
        if t < num_iter:
            k = t / num_iter
            if scheme == 'exp_mult':
                alpha = Tn / T0
                return T0 * (alpha ** k)
            #elif scheme == 'log_mult':
            #    return T0 / (1 + alpha * math.log(1 + k))
            elif scheme == 'lin_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * k)
            elif scheme == 'quad_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * (k ** 2))
        else:
            return Tn
    return cooling_schedule
