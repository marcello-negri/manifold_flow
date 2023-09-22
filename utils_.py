import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time
from datetime import timedelta
from functools import partial

import tqdm
from enflows.distributions import StandardNormal
from enflows.transforms import Sigmoid, ScalarScale, ScalarShift, RandomPermutation, MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.flows.base import Flow
from enflows.transforms.injective import FixedNorm, ConditionalFixedNorm, ConstrainedAnglesSigmoid, ResidualNetInput, ClampedAngles
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression

def log_likelihood(beta, sigma, X, y):
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const

def log_likelihood_bernoulli(beta, X, y):
    log_p_x = torch.nn.functional.logsigmoid(beta @ X.T)
    log_1_p_x = - torch.nn.functional.softplus(beta @ X.T)
    log_lk = y * log_p_x + (1 - y) * log_1_p_x

    return log_lk.sum(-1)

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
    return torch.exp(- lambdas * beta.abs())

def power(beta, lambdas, p):
    return laplace(beta, lambdas) ** p

def tanh(beta, lambdas, scale, shift, eps=1e-8):
    return torch.tanh(scale * laplace(beta, lambdas) + eps + shift) - torch.tanh(shift)

def sigmoid(beta, lambdas, eps=1e-8):
    return torch.sigmoid(laplace(beta, lambdas)-0.5+eps)

def softplus(beta, lambdas, eps=1e-8):
    return torch.nn.functional.softplus(- lambdas * beta.abs() + eps)

def log_prior_act_laplace(beta, lamb, q, act):

    lamb_ = 10 ** lamb
    eps = 1e-8

    # q_reshaped_beta = q.view(-1, *len(beta.shape[1:]) * (1,))  # (-1, 1, 1)
    # beta_q = (beta.abs() + eps).pow(q_reshaped_beta)

    # laplace_prior = torch.exp(- torch.clamp(lamb_.unsqueeze(-1) * beta.abs(), max=1e2))
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
        log_const = beta.shape[-1] * torch.log(0.5 * lamb_ * scalar)
        log_prior = - scalar * lamb_ * beta.abs().sum(-1)
        log_prior_DE = log_prior + log_const
    elif act == "identity":
        log_prior_DE = torch.log(laplace_prior + eps).sum(-1)
    elif act == "sigmoid":
        log_const = compute_norm_const(sigmoid, lamb_).sum(-1)
        log_prior = torch.log(sigmoid(beta, lamb_.unsqueeze(-1))).sum(-1)
        log_prior_DE = log_const + log_prior
    elif act == "softplus":
        # log_const = beta.shape[-1] * compute_norm_const(softplus, lamb_)
        log_prior = torch.log(softplus(- lamb_.unsqueeze(-1), beta)+1e-8).sum(-1)
        log_prior_DE = log_prior #+log_const
    elif act[:4] == "tanh":
        scale = float(act.split("_")[1])
        shift = torch.tensor(float(act.split("_")[2]))
        log_const = beta.shape[-1] * compute_norm_const(partial(tanh, scale=scale, shift=shift), lamb_)
        log_prior = torch.log( tanh(beta, lamb_.unsqueeze(-1), scale, shift) + eps).sum(-1)
        log_prior_DE = log_prior + log_const
    elif act == "exp10":
        shift = torch.tensor(10.)
        log_prior_DE = torch.log(torch.exp(laplace_prior+shift)-torch.exp(shift)+eps).sum(-1)
    elif act == "exp_2":
        shift = torch.tensor(-2.)
        log_prior_DE = torch.log(torch.exp(laplace_prior+shift)-torch.exp(shift)+eps).sum(-1)
    else:
        raise ValueError("invalid activation name. Choose from 'identity', 'tanh', 'sigmoid', 'softplus'")


    return log_prior_DE

def log_unnorm_posterior(beta, X, y, sigma, lamb, q, act):

    log_likelihood_ = log_likelihood(beta, sigma, X, y)
    # log_prior_beta_ = log_prior_laplace(beta, lamb)
    # log_prior_beta_ = log_prior_q_laplace(beta, lamb, p=p)
    log_prior_beta_ = log_prior_act_laplace(beta=beta, lamb=lamb, q=q, act=act)

    return log_likelihood_ + log_prior_beta_

def log_logistic_posterior(beta, X, y, lamb, q, act):

    log_likelihood_ = log_likelihood_bernoulli(beta, X, y)
    log_prior_beta_ = log_prior_act_laplace(beta=beta, lamb=lamb, q=q, act=act)

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

            # log_lik = log_likelihood(beta=q_samples, sigma=sigma, X=X, y=y)
            log_lik = log_likelihood_bernoulli(beta=q_samples, X=X, y=y)
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

def train_cond_model(model, X, y, sigma, act, q, epochs=2_001, lr=1e-3, sample_size=1, context_size=1_000, lambda_min_exp=-2,
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

            q_tensor = q_samples.new_ones(1)*q
            log_posterior = log_unnorm_posterior(beta=q_samples, X=X, y=y, sigma=sigma, lamb=uniform_lambda, q=q_tensor, act=act)
            # log_posterior = log_logistic_posterior(beta=q_samples, X=X, y=y, lamb=uniform_lambda, q=q_tensor, act=act)
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

def sample_beta_exp(model, X, y, sigma, lambda_min_exp, lambda_max_exp, q, act, context_size=10, sample_size=100, n_iter=100,
                    device='cuda'):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    sample_list, kl_list, lambda_list = [], [], []
    q_tensor = q * X.new_ones(1)
    for _ in tqdm.tqdm(range(n_iter)):
        rand_lambda = torch.rand(context_size).to(device)
        uniform_lambda = (rand_lambda * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
        posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=uniform_lambda)
        sample_list.append(posterior_samples.cpu().detach().numpy())
        lambda_list.append((10 ** uniform_lambda).view(-1).cpu().detach().numpy())
        log_lik = log_unnorm_posterior(beta=posterior_samples, X=X, y=y, sigma=sigma, lamb=uniform_lambda, q=q_tensor, act=act)
        kl_div = log_probs_samples - log_lik
        kl_list.append(kl_div.cpu().detach().numpy())

    sample_list, lambda_list, kl_list = np.concatenate(sample_list, 0), np.concatenate(lambda_list, 0), np.concatenate(
        kl_list, 0)

    lambda_sorted_idx = lambda_list.argsort()
    samples_sorted, lambda_sorted, kl_sorted = sample_list[lambda_sorted_idx], lambda_list[lambda_sorted_idx], kl_list[
        lambda_sorted_idx]

    return samples_sorted, lambda_sorted, kl_sorted

def plot_lines_gt (x_model, y_mean_model, y_l_model, y_r_model, x_gt, y_gt, dim, n_plots, log_scale=True, norm=1, name_file=None, coarse=1):
    n_lines = dim // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == dim:
                    break
                color = clrs[j % n_lines]
                ax.plot(x_model, y_mean_model[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(x_model, y_l_model[:, j], y_r_model[:, j], alpha=0.2, facecolor=color)
                ax.plot(x_gt[::coarse], y_gt[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            plt.xlabel(r"$||\beta||_{%s}$"%norm, fontsize=18)
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if log_scale: plt.xscale('log')
            if name_file is not None:
                plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
            plt.show()

def plot_betas_lambda(samples, lambdas, X_np, y_np, sigma, min_bin, max_bin, n_bins=51, norm=1, a=0.95, n_plots=1, gt='linear_regression', folder_name='./'):

    # compute ground truth solution path. Either linear regression or logistic regression
    if gt == 'linear_regression':
        if norm == 2:
            beta_sklearn = np.array([Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
                                     for alpha in tqdm.tqdm(lambdas * sigma.item()**2)])
        else:
            beta_sklearn = np.array([Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
                                    for alpha in tqdm.tqdm(lambdas * sigma.item()**2 / X_np.shape[0])])
        coarse = 1
    elif gt == 'logistic_regression':
        coarse = lambdas.shape[0] // 50
        beta_sklearn = [LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1 / c).fit(X_np, y_np).coef_
                        for c in tqdm.tqdm(lambdas[::coarse])]
        beta_sklearn = np.array(beta_sklearn).squeeze()
    else:
        raise ValueError("Ground truth path (gt) must be either 'linear_regression' or 'logistic_regression'")

    sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_sklearn[sklearn_sorted_idx]

    # 1) compute solution path for flow samples as a function of lambda
    sample_mean = samples.mean(1)
    l_quant = np.quantile(samples, 1 - a, axis=1)
    r_quant = np.quantile(samples, a, axis=1)

    plot_lines_gt(x_model=lambdas, y_mean_model=sample_mean, y_l_model=l_quant, y_r_model=r_quant, x_gt=lambdas,
                  y_gt=beta_sklearn, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=None)

    # 2) compute solution path for flow samples as a function of norm
    # first compute norm of each sample and then group samples by norm
    all_samples = samples.reshape(-1, X_np.shape[-1])
    all_samples_norms = np.power(np.power(np.abs(all_samples), norm).sum(1), 1 / norm)
    print(f"norms: min={all_samples_norms.min():.1f} max={all_samples_norms.max():.1f}")
    bins = np.linspace(min_bin, max_bin, n_bins)
    bins_midpoint = 0.5 * (bins[1:] + bins[:-1])
    digitized = np.digitize(all_samples_norms, bins)
    n_per_bin, _ = np.histogram(all_samples_norms, bins)
    min_n_per_bin = min(n_per_bin)
    print(f"min samples per bin {min_n_per_bin}")
    bin_means = np.array([all_samples[digitized == i][:min_n_per_bin].mean(0) for i in range(1, len(bins))])
    bin_l_quant = np.array(
        [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.05, axis=0) for i in range(1, len(bins))])
    bin_r_quant = np.array(
        [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.95, axis=0) for i in range(1, len(bins))])

    plot_lines_gt(x_model=bins_midpoint, y_mean_model=bin_means, y_l_model=bin_l_quant, y_r_model=bin_r_quant, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None)

    samples_norm = np.array([all_samples[digitized == i][:min_n_per_bin] for i in range(1, len(bins))])

    # 3) compute solution path for flow samples as a function of norm
    # first compute mean norm of samples with same lambda and then order samples by mean norm

    samples_mean = samples.mean(1)
    samples_l = np.quantile(samples, 1-a, axis=1)
    samples_r = np.quantile(samples, a, axis=1)

    norms = np.power(np.power(np.abs(samples), norm).sum(-1), 1 / norm)
    mean_norms = norms.mean(1)
    norm_idx = mean_norms.argsort()
    mean_norms_sorted = mean_norms[norm_idx]
    samples_mean_sorted = samples_mean[norm_idx]
    samples_l_sorted = samples_l[norm_idx]
    samples_r_sorted = samples_r[norm_idx]

    plot_lines_gt(x_model=mean_norms_sorted, y_mean_model=samples_mean_sorted, y_l_model=samples_l_sorted, y_r_model=samples_r_sorted,
                  x_gt=sklearn_norm, y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None)

    return bins, samples_norm, bin_l_quant, bin_means, bin_r_quant

def plot_betas_norm(samples_sorted, norm_sorted, X_np, y_np, norm=1, gt="linear_regression", a=0.95, n_plots=1, folder_name='./'):
    # compute ground truth solution path. Either linear regression or logistic regression
    if gt == 'linear_regression':
        if norm == 2:
            alphas_ridge = np.logspace(-2, 4, 2000)
            beta_sklearn = np.array(
                [Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_ridge)])
        else:
            alphas_lasso = np.logspace(-4, 2, 2000)
            beta_sklearn = np.array(
                [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_lasso)])
    elif gt == 'logistic_regression':
        # coarse = lambdas.shape[0] // 50
        lambdas = np.logspace(-1, 0.5, 200)
        beta_sklearn = [LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1 / c).fit(X_np, y_np).coef_
                        for c in tqdm.tqdm(lambdas)]
        beta_sklearn = np.array(beta_sklearn).squeeze()
    else:
        raise ValueError("Ground truth path (gt) must be either 'linear_regression' or 'logistic_regression'")

    sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_sklearn[sklearn_sorted_idx]

    l_quant = np.quantile(samples_sorted, 1 - a, axis=1)
    sample_mean = np.mean(samples_sorted, axis=1)
    r_quant = np.quantile(samples_sorted, a, axis=1)

    plot_lines_gt(x_model=norm_sorted, y_mean_model=sample_mean, y_l_model=l_quant, y_r_model=r_quant, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=None, coarse=-1)

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
