import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy
import numpy as np
import seaborn as sns
import torch
import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression, lasso_path
import pandas as pd

from imf.experiments.utils_manifold import cartesian_to_spherical_torch


def plot_loss (loss):
    plt.figure(figsize=(15,10))
    try:
        plt.plot(range(len(loss)), loss[:,1], label="log-likelihood")
        plt.plot(range(len(loss)), loss[:,2], label="MSE")
        if loss.shape[1]==6:
            plt.plot(range(len(loss)), loss[:,4], linestyle='dashed', label="val log-likelihood")
            plt.plot(range(len(loss)), loss[:,5], linestyle='dashed', label="val MSE")
    except:
        plt.plot(range(len(loss)), loss, label="kl")
    plt.legend()
    plt.show()


def plot_angle_distribution(samples_flow, samples_gt, filename=None, device='cuda'):
    # assert samples_gt.shape == samples_flow.shape

    angles_gt = cart_to_sph_batch(samples_gt)
    angles_flow = cart_to_sph_batch(samples_flow)

    n_dim = samples_gt.shape[-1] - 1
    n_rows = int(np.sqrt(n_dim))
    if n_rows**2 != n_dim: n_rows += 1
    n_cols = n_rows

    fig, axs = plt.subplots(figsize=(14,14), nrows=n_rows, ncols=n_cols)

    for i_r in np.arange(n_rows):
        for i_c in np.arange(n_rows):
            try:
                right_range = np.pi
                if n_rows*i_r+i_c+1 == angles_gt.shape[1]: right_range *= 2
                axs[i_r, i_c].hist(angles_gt[:,n_rows*i_r+i_c], bins=100, alpha=0.5, range=(0, right_range), density=True, label="gt")
                axs[i_r, i_c].hist(angles_flow[:,n_rows*i_r+i_c], bins=100, alpha=0.5, range=(0, right_range), density=True, label="flow")
                if i_r+i_c == 0: axs[i_r, i_c].legend()
            except:
                pass

    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()

def cart_to_sph_batch(samples, batchsize=1000):
    angles = []
    n_iter = samples.shape[0] // batchsize
    if n_iter * batchsize < samples.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * batchsize, (i + 1) * batchsize
        samples_torch = torch.from_numpy(samples[left:right]).float()
        samples_cart = cartesian_to_spherical_torch(samples_torch)
        angles += list(samples_cart[:, 1:].detach().numpy())

    return np.array(angles)

def plot_samples_pca(samples_flow, samples_gt, filename=None):
    sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
    # assert samples_gt.shape == samples_flow.shape

    angles_gt = cart_to_sph_batch(samples_gt)
    angles_flow = cart_to_sph_batch(samples_flow)

    pca = PCA(n_components=2)
    angles_pca_gt = pca.fit_transform(angles_gt)
    angles_pca_flow = pca.transform(angles_flow)

    plt.figure(figsize=(14, 14))
    plt.scatter(angles_pca_gt[:, 0], angles_pca_gt[:, 1], marker='*', alpha=0.2, label='gt')
    plt.scatter(angles_pca_flow[:, 0], angles_pca_flow[:, 1], marker='.', alpha=0.2, label='flow')
    plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


def plot_lines_gt (x_gt, y_gt, dim, n_plots, x_model=None, y_mean_model=None, y_l_model=None, y_r_model=None, log_scale=True, norm=1, name_file=None, coarse=1, true_coeff=None, x_label='norm'):
    n_lines = dim // n_plots
    clrs = sns.color_palette("husl", n_lines)
    sns.set_style("whitegrid")
    sns.set_context('talk')
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(12, 10))

        for j in range(i * n_lines, (i + 1) * n_lines):
            if j == dim:
                break
            color = clrs[j % n_lines]
            if any([i is not None for i in [x_model, y_mean_model, y_l_model, y_r_model]]):
                ax.plot(x_model, y_mean_model[:, j], c=color, alpha=0.9, linewidth=2.5)
                ax.fill_between(x_model, y_l_model[:, j], y_r_model[:, j], alpha=0.2, facecolor=color)
            ax.plot(x_gt[::coarse], y_gt[:, j], linestyle='--', linewidth=2.5, c=color, alpha=0.7)
            if true_coeff is not None:
                if true_coeff[j]!=0:
                    ax.axhline(y=true_coeff[j], xmin=x_gt[::coarse].min(), xmax=x_gt[::coarse].max(), c=color, alpha=0.6, linewidth=2.5, linestyle=':')
        if x_label == "norm":
            plt.xlabel(r"$||\beta||_{%s}$" % norm, fontsize=26)
        elif x_label == "lambda":
            plt.xlabel(r"$\lambda$", fontsize=26)
        else:
            raise ValueError("x_label must be either 'norm' or 'lambda'")
        plt.ylabel(r'$\beta$', fontsize=26)
        plt.locator_params(axis='y', nbins=4)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        if log_scale: plt.xscale('log')
        if name_file is not None:
            plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def plot_betas_lambda(samples, lambdas, X_np, y_np, sigma, gt_only=False, min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', folder_name='./', true_coeff=None, name=None):

    # compute ground truth solution path. Either linear regression or logistic regression
    if gt == 'linear_regression':
        if norm == 2:
            beta_path = np.array([Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
                                  for alpha in tqdm.tqdm(lambdas * sigma.item() ** 2)])
            alphas = lambdas
        else:
            alphas, beta_path, _ = lasso_path(X_np, y_np)
            beta_path = beta_path.T
            alphas = alphas * X_np.shape[0] / sigma **2
        if gt_only:
            plot_lines_gt(x_model=None, y_mean_model=None, y_l_model=None, y_r_model=None, x_gt=alphas, y_gt=beta_path,
                          dim=X_np.shape[-1], n_plots=n_plots, log_scale=True, norm=norm, true_coeff=true_coeff, x_label="lambda")
            norms_sorted, coeff_sorted = sort_path_by_norm(beta_path, norm=norm)
            plot_lines_gt(x_model=None, y_mean_model=None, y_l_model=None, y_r_model=None, x_gt=norms_sorted,
                          y_gt=coeff_sorted,
                          dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, true_coeff=true_coeff)

            #     beta_path = np.array([Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
            #                             for alpha in tqdm.tqdm(lambdas * sigma.item()**2 / X_np.shape[0])])
            # coarse = 1
            return alphas
    elif gt == 'logistic_regression':
        coarse = lambdas.shape[0] // 50
        beta_path = [
            LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1 / c).fit(X_np, y_np).coef_
            for c in tqdm.tqdm(lambdas[::coarse])]
        beta_path = np.array(beta_path).squeeze()
    else:
        raise ValueError("Ground truth path (gt) must be either 'linear_regression' or 'logistic_regression'")

    sklearn_norm = np.power(np.power(np.abs(beta_path), norm).sum(1), 1 / norm)
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = beta_path[sklearn_sorted_idx]

    # 1) compute solution path for flow samples as a function of lambda
    sample_mean = samples.mean(1)
    l_quant = np.quantile(samples, 1 - conf, axis=1)
    r_quant = np.quantile(samples, conf, axis=1)
    plot_lines_gt(x_model=lambdas, y_mean_model=sample_mean, y_l_model=l_quant, y_r_model=r_quant, x_gt=alphas,
                  y_gt=beta_path, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=name, true_coeff=true_coeff, x_label="lambda")

    # 2) compute solution path for flow samples as a function of norm
    # first compute norm of each sample and then group samples by norm
    all_samples = samples.reshape(-1, X_np.shape[-1])
    try:
        bins_midpoint, bin_means, bin_l, bin_r, samples_norm = refine_samples_by_norm(all_samples, norm, min_bin, max_bin, n_bins, conf=conf)
        plot_lines_gt(x_model=bins_midpoint, y_mean_model=bin_means, y_l_model=bin_l, y_r_model=bin_r, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=name+"_n" if name is not None else None, true_coeff=true_coeff)
    except:
        samples_norm, bin_l, bin_means, bin_r = None, None, None, None
        pass

    # 3) compute solution path for flow samples as a function of norm
    # first compute mean norm of samples with same lambda and then order samples by mean norm

    mean_norms, samples_mean, samples_l, samples_r = refine_samples_by_mean_norm(samples, norm, conf)

    plot_lines_gt(x_model=mean_norms, y_mean_model=samples_mean, y_l_model=samples_l, y_r_model=samples_r, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=name+"_nn" if name is not None else None, true_coeff=true_coeff)



    return samples_norm, bin_l, bin_means, bin_r

def plot_betas_lambda_fixed_norm(samples, lambdas, dim, conf=0.95, n_plots=1, true_coeff=None, log_scale=True):

    sample_mean = samples.mean(1)
    l_quant = np.quantile(samples, 1 - conf, axis=1)
    r_quant = np.quantile(samples, conf, axis=1)

    n_lines = dim // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == dim:
                    break
                color = clrs[j % n_lines]
                ax.plot(lambdas, sample_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambdas, l_quant[:, j], r_quant[:, j], alpha=0.2, facecolor=color)
                # secax.plot(alphas, beta_path[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)
                if true_coeff is not None:
                    if true_coeff[j] != 0:
                        print(true_coeff[j])
                        ax.axhline(y=true_coeff[j], c=color, linestyle='dashed')
            plt.xlabel(r"$\lambda$", fontsize=18)
            # secax.set_xlabel('$\lambda_{LASSO}$')
            plt.ylabel(r'$\beta$', fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if log_scale:
                plt.xscale('log')
            # plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
            plt.show()

def plot_cumulative_returns(samples, lambdas, X_np, y_np, prior_name, conf=0.95, n_samples=5, n_plots=1):

    returns = samples @ X_np.T
    returns = np.cumprod(returns, axis=2)
    mean_return = returns.mean(1).T
    l_return = np.quantile(returns, 1 - conf, axis=1).T
    r_return = np.quantile(returns, conf, axis=1).T

    dist_to_gt = np.square(returns - np.cumprod(y_np).ravel()).mean(-1)
    sorted_dist = np.argsort(dist_to_gt)

    n_lines = lambdas.shape[0]
    clrs = sns.color_palette("husl", n_lines)
    with sns.axes_style("whitegrid"):
        for j in range(n_lines):
            fig, ax = plt.subplots(figsize=(14, 14))
            color = clrs[j % n_lines]
            ax.plot(range(X_np.shape[0]), returns[j][sorted_dist[j]][:n_samples].T, alpha=0.5, linewidth=3, label='5 closest samples')
            ax.plot(range(X_np.shape[0]), mean_return[:, j], c=color, linestyle='dashed', alpha=1, linewidth=4, label='average')
            ax.fill_between(range(X_np.shape[0]), l_return[:, j], r_return[:, j], alpha=0.2, facecolor=color, label='95% C.I.')
            ax.plot(range(X_np.shape[0]), np.cumprod(y_np).ravel(), c='k', linestyle='dotted', linewidth=4, label='ref. index')
            plt.locator_params(axis='y', nbins=4)
            plt.locator_params(axis='x', nbins=5)
            plt.xlabel(r"time", fontsize=36)
            plt.ylabel(r'cumulative return', fontsize=36)
            plt.ylim((0,13))
            plt.xticks(fontsize=24)
            plt.yticks(fontsize=24)
            # plt.title(f'Cumulative return for alpha={lambdas[j]:.2f}')
            plt.legend(loc=2, prop={'size': 28})
            plt.savefig(f"{os.getcwd()}/imf/experiments/plots/path_{prior_name}_{lambdas[j]:.2f}.pdf", bbox_inches='tight')
            plt.show()

    sorted_samples = np.array([samples[i][sorted_dist[i]] for i in range(samples.shape[0])])
    plot_sparsity_patterns(samples=sorted_samples[:,:30], prior_name=prior_name, sort=False)

def plot_cumulative_returns_singularly(samples, X_np, y_np, prior_name, conf=0.95, n_plots=1):

    returns = samples @ X_np.T
    returns = np.cumprod(returns, axis=-1)

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.plot(range(X_np.shape[0]), returns.T, alpha=0.7, linewidth=1.5)
        ax.plot(range(X_np.shape[0]), np.cumprod(y_np).ravel(), c='r', linestyle='dashed')
        plt.xlabel(r"time", fontsize=18)
        plt.ylabel(r'stock value', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"{os.getcwd()}/imf/experiments/plots/path_{prior_name}_lines.pdf", bbox_inches='tight')
        plt.show()


def plot_sparsity_patterns(samples, prior_name, sort=True, threshold=None):
    def sort_matrix_by_non_zero(matrix, threshold):
        non_zero_counts = np.sum(matrix > threshold, axis=-1)
        sorted_indices = np.argsort(non_zero_counts)
        sorted_matrix = matrix[sorted_indices]
        return sorted_matrix

    if threshold is None:
        threshold = 1. / samples.shape[-1] # anything more than uniform

    num_matrices = samples.shape[0]
    fig, axes = plt.subplots(num_matrices, 1,)# figsize=(10, 5 * num_matrices))

    # Plot each sorted matrix
    for i, matrix in enumerate(samples):
        if sort:
            sorted_matrix = sort_matrix_by_non_zero(matrix, threshold)
        else:
            sorted_matrix = matrix
        ax = axes[i]
        im = ax.imshow(sorted_matrix.T, vmin=0, vmax=1, cmap="Blues")
        # ax.set_title(f'Matrix {i + 1} (Sorted by Non-zero Entries)')
        # ax.set_xlabel('Columns')
        # ax.set_ylabel('Rows')
        # if i != samples.shape[0] - 1:
        ax.set_xticks([])

    # Add a colorbar to the last subplot
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.15, pad=0.05)

    # plt.tight_layout()
    plt.savefig(f"{os.getcwd()}/imf/experiments/plots/sparsity_{prior_name}.pdf", bbox_inches='tight')
    plt.show()



def plot_sparsity_distr(samples1, samples2, X_np, y_np, norm=True, threshold=None, n_bins=25, folder=None):

    def keys_values(samples, threshold, norm=True):
        non_zero_counts = np.sum(samples > threshold, axis=-1)
        counts_dict = Counter(non_zero_counts)
        keys = np.array(list(counts_dict.keys()))
        values = np.array(list(counts_dict.values()))
        sort_idx = np.argsort(keys)
        values_ = values / values.sum()
        return keys[sort_idx], values_[sort_idx]

    if threshold is None:
        threshold = 1. / samples1.shape[-1] # anything more than uniform

    mse1 = np.sqrt(np.square(samples1 @ X_np.T - y_np).mean(-1))
    flatten_mse1 = mse1.flatten()

    mse2 = np.sqrt(np.square(samples2 @ X_np.T - y_np).mean(-1))
    flatten_mse2 = mse2.flatten()

    min_bin = max(mse1.min(), mse2.min())
    max_bin = min(mse1.max(), mse2.max())

    bin_edges = np.linspace(min_bin, max_bin, num=n_bins)
    idx1 = np.digitize(flatten_mse1, bin_edges)
    idx2 = np.digitize(flatten_mse2, bin_edges)

    n_dim = n_bins//3 - 1
    n_rows = int(np.sqrt(n_dim))
    if n_rows ** 2 != n_dim: n_rows += 1
    n_cols = n_rows

    # reshaped_samples = samples1.reshape(-1, samples1.shape[-1])
    # non_zero_counts = np.sum(reshaped_samples > 0.01, axis=-1)
    # H, xedges, yedges = np.histogram2d(non_zero_counts, 10*(flatten_mse1-flatten_mse1.min())/(flatten_mse1.max() - flatten_mse1.min()))
    # plt.imshow(H, interpolation='nearest', origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    with sns.axes_style("whitegrid"):

        fig, axs = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)
        fig1, axs1 = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)
        fig2, axs2 = plt.subplots(figsize=(14, 14), nrows=n_rows, ncols=n_cols)

        flatten_samples1 = samples1.reshape(-1, samples1.shape[-1])
        flatten_samples2 = samples2.reshape(-1, samples2.shape[-1])

        for i_r in np.arange(n_rows):
            for i_c in np.arange(n_rows):
                try:
                    samples_range1 = flatten_samples1[idx1 == n_rows * i_r + i_c + 1]
                    keys1, values1 = keys_values(samples_range1, norm=norm, threshold=threshold)
                    samples_range2 = flatten_samples2[idx2 == n_rows * i_r + i_c + 1]
                    keys2, values2 = keys_values(samples_range2, norm=norm, threshold=threshold)
                    axs[i_r, i_c].bar(x=keys1, height=values1, alpha=0.5, label='Uniform')
                    axs[i_r, i_c].bar(x=keys2, height=values2, alpha=0.5, label='Dirichlet')
                    if i_r + i_c == 0: axs[i_r, i_c].legend()

                    axs1[i_r, i_c].imshow(samples_range1[:50].T, cmap="Blues")
                    axs2[i_r, i_c].imshow(samples_range2[:50].T, cmap="Oranges")
                except:
                    pass

        of = folder if folder is not None else "./"
        fig.savefig(of + "non_zero_distr.pdf", bbox_inches='tight')
        fig1.figure.savefig(of + "non_zero_uniform.pdf", bbox_inches='tight')
        fig2.figure.savefig(of + "non_zero_dirichlet.pdf", bbox_inches='tight')
        plt.show()


def plot_returns(samples, lambdas, X_np, y_np, conf=0.95, n_plots=1):

    # mean and quantiles of returns
    returns = samples @ X_np.T
    mean_return = returns.mean(1).T
    l_return = np.quantile(returns, 1 - conf, axis=1).T
    r_return = np.quantile(returns, conf, axis=1).T

    n_lines = lambdas.shape[0]
    clrs = sns.color_palette("husl", n_lines)
    fig, ax = plt.subplots(figsize=(14, 14))
    with sns.axes_style("darkgrid"):
        for j in range(n_lines):
            color = clrs[j % n_lines]
            ax.plot(range(X_np.shape[0]), mean_return[:, j], c=color, alpha=0.7, linewidth=1.5)
            ax.fill_between(range(X_np.shape[0]), l_return[:, j], r_return[:, j], alpha=0.2, facecolor=color)
        ax.plot(range(X_np.shape[0]), y_np.ravel(), linestyle='dashed')
        plt.xlabel(r"return", fontsize=18)
        plt.ylabel(r'time', fontsize=18)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
        plt.show()


def refine_samples_by_norm(samples, norm, min_bin, max_bin, n_bins, conf):
    # samples_norms = np.linalg.vector_norm(samples, ord=norm, axis=-1)
    samples_norms = np.power(np.power(np.abs(samples), norm).sum(1), 1 / norm)
    bins = np.linspace(min_bin, max_bin, n_bins)
    bins_midpoint = 0.5 * (bins[1:] + bins[:-1])
    digitized = np.digitize(samples_norms, bins, right=True)
    n_per_bin, _ = np.histogram(samples_norms, bins)
    min_n_per_bin = min(n_per_bin)
    print(f"min samples per bin {min_n_per_bin}")
    bin_means = np.array([samples[digitized == i][:min_n_per_bin].mean(0) for i in range(1, len(bins))])
    bin_l_quant = np.array([np.quantile(samples[digitized == i][:min_n_per_bin], 1-conf, axis=0) for i in range(1, len(bins))])
    bin_r_quant = np.array([np.quantile(samples[digitized == i][:min_n_per_bin], conf, axis=0) for i in range(1, len(bins))])
    samples_norm = np.array([samples[digitized == i][:min_n_per_bin] for i in range(1, len(bins))])

    return bins_midpoint, bin_means, bin_l_quant, bin_r_quant, samples_norm

def refine_samples_by_mean_norm(samples, norm, conf):
    samples_mean = samples.mean(1)
    samples_l = np.quantile(samples, 1 - conf, axis=1)
    samples_r = np.quantile(samples, conf, axis=1)

    mean_norms = np.power(np.power(np.abs(samples_mean), norm).sum(-1), 1 / norm)
    # norms = np.power(np.power(np.abs(samples), norm).sum(-1), 1 / norm)
    # mean_norms = norms.mean(1)
    # norms = np.linalg.vector_norm(samples, ord=norm, axis=-1)

    norm_idx = mean_norms.argsort()
    mean_norms_sorted = mean_norms[norm_idx]
    samples_mean_sorted = samples_mean[norm_idx]
    samples_l_sorted = samples_l[norm_idx]
    samples_r_sorted = samples_r[norm_idx]

    return mean_norms_sorted, samples_mean_sorted, samples_l_sorted, samples_r_sorted

def sort_path_by_norm(coeff, norm):
    norms = np.power(np.power(np.abs(coeff), norm).sum(-1), 1 / norm)
    # norms = np.linalg.vector_norm(coeff, ord=norm, axis=-1)
    norm_idx = norms.argsort()
    norms_sorted = norms[norm_idx]
    coeff_sorted = coeff[norm_idx]

    return norms_sorted, coeff_sorted


def plot_marginal_likelihood (kl_sorted, cond_sorted, args):
    mll = -kl_sorted
    mll_mean = mll.mean(1)
    mll_l = np.quantile(mll, 0.05, axis=1)
    mll_r = np.quantile(mll, 0.95, axis=1)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(cond_sorted, mll_mean)
    ax.fill_between(cond_sorted, mll_l, mll_r, alpha=0.5)

    max_mll = np.argmax(mll_mean)
    opt_cond = cond_sorted[max_mll]
    plt.vlines(opt_cond, ymin=mll_l.min(), ymax=mll_r.max())
    if args.log_cond: plt.xscale('log')
    plt.show()

    return opt_cond


'''
takes numpy array of size nxd and creates a 2d scatter plot
'''
def plot_dirichlet_proj(x, max_num = 5000):
    ones = numpy.ones(x.shape[-1])
    svd = numpy.linalg.svd(np.outer(ones,ones))
    proj = svd.U[:,1:3]  # matrix of size dx2
    xproj = x @ proj

    np.random.shuffle(xproj)
    xproj = xproj[:max_num]

    df = pd.DataFrame(xproj, columns=['x', 'y'])
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x='x', y='y', alpha=0.1)

    plt.show()

def plot_hist(x):
    x = x.flatten()
    plt.bar(np.arange(x.shape[0]), x)

    plt.show()

