import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import time
from datetime import timedelta
import utils_

import tqdm
from enflows.distributions import StandardNormal, DiagonalNormal#, MOG
from enflows.transforms import Sigmoid, ScalarScale, ScalarShift, RandomPermutation, MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
# from enflows.transforms.lipschitz import LipschitzDenseNetBuilder, iResBlock
from enflows.flows.base import Flow
# from enflows.transforms.injective import FixedNorm, ConditionalFixedNorm, ConstrainedAnglesSigmoid, ResidualNetInput, ClampedAngles
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_diabetes_dataset(device='cuda'):
    df = load_diabetes()
    scaler = StandardScaler()
    X_np = scaler.fit_transform(df.data)
    y_np = scaler.fit_transform(df.target.reshape(-1, 1))[:, 0]
    X_tensor = torch.from_numpy(X_np).float().to(device)
    y_tensor = torch.from_numpy(y_np).float().to(device)

    # compute regression parameters
    reg = LinearRegression().fit(X_np, y_np)
    r2_score = reg.score(X_np, y_np)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y_np - X_np @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return X_tensor, y_tensor, X_np, y_np

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_diabetes_dataset(device='cuda'):
    df = load_diabetes()
    scaler = StandardScaler()
    X_np = scaler.fit_transform(df.data)
    y_np = scaler.fit_transform(df.target.reshape(-1, 1))[:, 0]
    X_tensor = torch.from_numpy(X_np).float().to(device)
    y_tensor = torch.from_numpy(y_np).float().to(device)

    # compute regression parameters
    reg = LinearRegression().fit(X_np, y_np)
    r2_score = reg.score(X_np, y_np)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y_np - X_np @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return X_tensor, y_tensor, X_np, y_np

def log_likelihood(beta: torch.Tensor, sigma: torch.Tensor, X: torch.Tensor, y: torch.Tensor, ):

    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * torch.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const


def build_cond_flow_manifold(flow_dim, q, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    base_dist = StandardNormal(shape=[flow_dim - 1])

    # means =  torch.tensor([[1.],[0.],[-1]]).repeat(1,flow_dim-1).to(device)
    # stds = (torch.ones(means.shape[0],flow_dim-1) * 0.3).to(device)
    # base_dist = MOG(means=means, stds=stds)

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

def train_cond_model(model, X, y, sigma, epochs=2_001, lr=1e-3, sample_size=1, context_size=1_000, norm_min=-1,
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

            # log_prior = log_prior_angles(beta=q_samples, sigma=torch.tensor(0.1))
            # log_lik = log_likelihood(beta=q_samples, sigma=sigma, X=X, y=y)
            log_post = log_unnorm_posterior(beta=q_samples, sigma=sigma, X=X, y=y, lamb=torch.tensor(200.))
            # log_lik = uniform_p_norm(beta=q_log_prob)
            # log_posterior = log_prior + log_lik
            # kl_div = torch.mean(q_log_prob - log_posterior / T)
            #
            kl_div = torch.mean(q_log_prob - log_post / T)
            kl_div.backward()

            # print("flow prob", q_log_prob)
            # print(q_log_prob.shape, uniform_norm.shape)
            # print("log lik", uniform_norm[torch.isinf(q_log_prob)])

            # entropy = torch.mean(q_log_prob)
            # entropy.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            # loss.append(torch.mean(q_log_prob_beta - log_lik).cpu().detach().numpy())
            loss.append(torch.mean(q_log_prob - log_lik).cpu().detach().numpy())
            loss_T.append(torch.mean(q_log_prob - log_lik / T).cpu().detach().numpy())
            print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")




    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T

def train_cond_model_manifold(model, X, y, sigma, epochs=2_001, lr=1e-3, sample_size=1, context_size=1_000, norm_min=-1,
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

            # log_prior = log_prior_angles(beta=q_samples, sigma=torch.tensor(0.1))
            log_lik = log_likelihood(beta=q_samples, sigma=sigma, X=X, y=y)
            # log_lik = log_unnorm_posterior(beta=q_samples, sigma=sigma, X=X, y=y, lamb=torch.tensor(200.))
            # log_lik = uniform_p_norm(beta=q_log_prob)
            # log_posterior = log_prior + log_lik
            # kl_div = torch.mean(q_log_prob - log_posterior / T)
            #
            kl_div = torch.mean(q_log_prob - log_lik / T)
            kl_div.backward()

            # print("flow prob", q_log_prob)
            # print(q_log_prob.shape, uniform_norm.shape)
            # print("log lik", uniform_norm[torch.isinf(q_log_prob)])

            # entropy = torch.mean(q_log_prob)
            # entropy.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
            optimizer.step()

            # loss.append(torch.mean(q_log_prob_beta - log_lik).cpu().detach().numpy())
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

    sample_list, norm_list, kl_list = np.concatenate(sample_list, 0), np.concatenate(norm_list, 0), np.concatenate(kl_list, 0)

    norm_sorted_idx = norm_list.argsort()
    samples_sorted, norm_sorted, kl_sorted = sample_list[norm_sorted_idx], norm_list[norm_sorted_idx], kl_list[norm_sorted_idx]

    return samples_sorted, norm_sorted, kl_sorted

def plot_betas_norm(samples_sorted, norm_sorted, X_np, y_np, norm=1, a=0.95, folder_name='./'):
    if norm == 2:
        alphas_ridge = np.logspace(-2, 4, 1000)
        beta_sklearn = np.array(
            [Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in tqdm.tqdm(alphas_ridge)])
    else:
        alphas_lasso = np.logspace(-4, 2, 1000)
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

    clrs = sns.color_palette("husl", X_np.shape[-1])
    fig, ax = plt.subplots(figsize=(14, 14))
    with sns.axes_style("darkgrid"):
        for i in range(sample_mean.shape[-1]):
            color = clrs[i % X_np.shape[-1]]
            ax.plot(norm_sorted, sample_mean[:, i], c=color, alpha=0.7, linewidth=1.5)
            ax.fill_between(norm_sorted, l_quant[:, i], r_quant[:, i], alpha=0.2, facecolor=color)
            ax.plot(sklearn_norm, sklearn_sorted[:, i], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

        # ax.set_xscale('log')
        plt.xlabel(r'$||\beta||_1$', fontsize=18)
        plt.ylabel(r'$\beta$', fontsize=18)
        plt.locator_params(axis='y', nbins=4)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.xscale('log')
        # plt.savefig(f"{folder_name}beta_norm_05_prior_pi_2.png", dpi=200, bbox_inches='tight')
        plt.show()

def main():
    device = 'cuda'
    set_random_seeds(1234)

    # load data
    X_tensor, y_tensor, X_np, y_np = load_diabetes_dataset(device=device)

    # build model
    flow_dim = X_tensor.shape[1]
    q = 1
    flow = utils_.build_cond_flow(flow_dim, n_layers=5, hidden_features=128, device=device)
    # flow = build_cond_flow_manifold(flow_dim, q=q, n_layers=3, context_features=32, hidden_features=128, device=device)

    params_flow = dict(q=q,
                  sigma=torch.tensor(0.7),
                  lr=1e-4,
                  epochs=200,
                  T0=5,
                  Tn=1,
                  iter_per_cool_step=100,
                  norm_min=1,
                  norm_max=2,  # .4
                  sample_size=1,
                  context_size=500,
                  lambda_min_exp=-1,
                  lambda_max_exp=3,
                  device=device)

    flow, loss, loss_T = utils_.train_cond_model(flow, X=X_tensor, y=y_tensor, act="laplace_exact", **params_flow)
    flow.eval()

    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(loss)), loss, label='loss')
    # # plt.plot(range(len(loss)), loss_T, label='loss_T')
    # # plt.yscale("log")
    # plt.legend()
    # plt.show()

    samples_sorted, lambda_sorted, kl_sorted = utils_.sample_beta_exp(flow, X_tensor, y_tensor,
                                                                     sigma=params_flow['sigma'],
                                                                     lambda_min_exp=params_flow['lambda_min_exp'],
                                                                     lambda_max_exp=params_flow['lambda_max_exp'],
                                                                     context_size=2, sample_size=1000, n_iter=25,
                                                                     device='cuda', q=1., act='laplace_exact')
    bins, samples_norm, bin_l_quant, bin_means, bin_r_quant = utils_.plot_betas_lambda(samples_sorted, lambda_sorted,
                                                                                      X_np, y_np,
                                                                                      sigma=params_flow['sigma'],
                                                                                      norm=params_flow['q'], a=0.95, n_plots=1, min_bin=0.1, max_bin=1)


    # samples_sorted, norm_sorted, kl_sorted = sample_beta(flow, X_tensor, y_tensor, sigma=params_flow['sigma'],
    #                                                      norm_min=params_flow['norm_min'], norm_max=params_flow['norm_max'],
    #                                                      context_size=1, sample_size=50, n_iter=250, device='cuda')
    #
    # plot_betas_norm(samples_sorted, norm_sorted, X_np, y_np, norm=params_flow['q'], a=0.95)


if __name__ == "__main__":
    main()