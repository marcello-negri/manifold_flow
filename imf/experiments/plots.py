import torch
import meshio
import tqdm
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from functools import partial

from imf.experiments.utils_manifold import rnf_forward_logp, rnf_forward_points
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


def plot_samples(samples, n_samples=10000):
    if samples.shape[-1] == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(samples[:n_samples, 0], samples[:n_samples, 1], samples[:n_samples, 2], marker=".", alpha=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        plt.show()
    else:
        print(f"Skipping 3D plot because d={samples.shape[-1]}")

def plot_icosphere(data, dataset, flow, samples_flow, rnf, samples_rnf, kde=False, device='cuda', args=None, plot_rnf=False):
    # load regular grid in spherical coordinates
    mesh = load_mesh()
    points = mesh.points

    # plot ground truth density
    fig = plt.figure(figsize=(15, 15))
    ax11 = fig.add_subplot(231, projection='3d')
    plot_samples_ax(ax11, data, title="ground truth")
    ax21 = fig.add_subplot(234, projection='3d')
    if kde:
        plot_density(ax21, data, points, title="ground truth")
    else:
        logp_simulator = partial(density_gt, dataset=dataset)
        plot_logp(ax21, logp_function=logp_simulator, points_surface=points, title="gt")

    # plot manifold flow density
    # project point on the learnt surface
    points_torch = torch.from_numpy(points).float().to(device).requires_grad_(True)
    angles = flow.transform_to_noise(points_torch, context=None)
    uniform_surface_flow, logabsdet = flow._transform.inverse(angles, context=None)
    points_surface_flow = uniform_surface_flow.detach().cpu().numpy()

    ax12 = fig.add_subplot(232, projection='3d')
    plot_samples_ax(ax12, samples_flow, title="ours")
    ax22 = fig.add_subplot(235, projection='3d')
    if kde:
        plot_density(ax22, samples_flow, points_surface_flow, title="ours")
    else:
        logp_flow = partial(density_flow, flow=flow)
        plot_logp (ax22, logp_function=logp_flow, points_surface=points_surface_flow, title="ours")
    # plot rectangular normalizing flos density
    # project point on the learnt surface

    if plot_rnf:
        # uniform_surface_rnf, log_prob, u = rnf.forward(points_torch.detach().cpu())
        points_surface_rnf = rnf_forward_points(rnf, points_torch.detach().cpu(), args)
        ax13 = fig.add_subplot(233, projection='3d')
        plot_samples_ax(ax13, samples_rnf, title="rnf")
        ax23 = fig.add_subplot(236, projection='3d')
        if kde:
            plot_density(ax23, samples_rnf, points_surface_rnf, title="rnf")
        else:
            logp_rnf = partial(density_rnf, rnf=rnf, args=args)
            plot_logp(ax23, logp_function=logp_rnf, points_surface=points_surface_rnf, title="rnf")

    plt.show()

def map_colors(p3dc, func, kde=False, cmap='viridis'):
    """
    function taken from: https://stackoverflow.com/questions/63298864/plot-trisurface-with-custom-color-array

    Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize, LogNorm

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = np.array([np.array((x[s], y[s], z[s])).T for s in slices])

    # compute the function in the barycenter
    if kde: values = func(triangles.mean(axis=1).T)
    else: values = func(triangles.mean(axis=1))

    # usual stuff
    norm = Normalize()#vmin=0, vmax=1)
    # norm = LogNorm()#vmin=0, vmax=1)
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)

def plot_density (ax, samples, points_surface, title):
    # Create a kernel density estimate for 3D data
    kde_func = gaussian_kde(samples.T)
    mesh = load_mesh()
    triangles = mesh.cells[0].data

    trisurf = ax.plot_trisurf(points_surface[:, 0], points_surface[:, 1], points_surface[:, 2], triangles=triangles)

    # change to facecolor
    mappable = map_colors(trisurf, kde_func, kde=True, cmap='viridis')
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def density_gt (points, dataset):
    points_torch = torch.from_numpy(points).cuda().float()
    logp = dataset.log_density(points_torch)
    return logp.exp().detach().cpu().numpy()
    # return logp.detach().cpu().numpy()

def density_flow (points, flow, batch_size=1000, device="cuda"):
    logp_flow = []
    n_iter = points.shape[0] // batch_size
    if n_iter * batch_size < points.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * batch_size, (i + 1) * batch_size
        points_torch = torch.from_numpy(points[left:right]).float().to(device).requires_grad_(True)
        angles = flow.transform_to_noise(points_torch, context=None)
        logp_flow_ = flow._distribution.log_prob(angles, context=None)
        uniform_surface_flow, logabsdet = flow._transform.inverse(angles, context=None)
        logp_flow_ = logp_flow_ - logabsdet
        logp_flow += list(logp_flow_.detach().cpu().numpy())

    return np.exp(np.array(logp_flow))
    # return logp_flow.detach().cpu().numpy()

def density_rnf (points, rnf, args):
    logp_rnf = rnf_forward_logp(rnf, torch.from_numpy(points).float(), args)

    return np.exp(logp_rnf)

def plot_logp (ax, logp_function, points_surface, title):
    mesh = load_mesh()
    triangles = mesh.cells[0].data

    trisurf = ax.plot_trisurf(points_surface[:, 0], points_surface[:, 1], points_surface[:, 2], triangles=triangles)

    # change to facecolor
    mappable = map_colors(trisurf, logp_function, cmap='viridis')
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def plot_samples_ax(ax, samples, title):
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], marker=".", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def plot_pairwise_angle_distribution (samples, n_samples=10000):
    d = samples.shape[-1]
    assert d > 2

    # Calculate the angles between pairs of vectors
    thetas = samples[:n_samples] @ samples[:n_samples].T
    thetas = thetas[np.triu_indices_from(thetas, 1)]
    thetas = np.arccos(thetas)

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 8), tight_layout=True)
    ax.set_ylabel('pdf', fontsize=20)
    ax.set_xlabel(r'$\theta_{ij}$', fontsize=20)
    ax.hist(thetas, bins='auto', density=True, label='observed distribution')
    x = np.linspace(0, np.pi, num=1000)
    ax.plot(x, 1 / np.sqrt(np.pi) * sp.special.gamma(d / 2) / sp.special.gamma((d - 1) / 2) * (np.sin(x)) ** (d - 2), color='red', linestyle='--',
            label=r'$\frac{1}{\sqrt{\pi}} \frac{\Gamma(n/2)}{\Gamma\left(\frac{n-1}{2}\right)} \left(\sin(\theta)\right)^{n-2}$')
    plt.legend(loc='upper right', prop={'size': 18}, markerscale=4)
    ax.set_xlim(0, np.pi)

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

    # pca = PCA(n_components=2)
    # samples_pca_gt = pca.fit_transform(samples_gt)
    # samples_pca_flow = pca.transform(samples_flow)
    #
    # plt.figure(figsize=(14,14))
    # plt.scatter(samples_pca_gt[:,0], samples_pca_gt[:,1], marker='*', alpha=0.2, label='gt')
    # plt.scatter(samples_pca_flow[:,0], samples_pca_flow[:,1], marker='.', alpha=0.2, label='flow')
    # plt.legend()
    # plt.show()

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

    # from sklearn.manifold import TSNE
    #
    # # Assuming X is your data matrix
    # samples_emb_gt = TSNE(n_components=2).fit_transform(samples_gt)
    # samples_emb_flow = TSNE(n_components=2).fit_transform(samples_flow)
    #
    # # Plot the clusters
    # plt.scatter(samples_emb_gt[:, 0], samples_emb_gt[:, 1], alpha=0.2)
    # plt.scatter(samples_emb_flow[:, 0], samples_emb_flow[:, 1], alpha=0.2)
    # plt.legend()
    # plt.show()
    #
    # # Assuming X is your data matrix
    # angles_emb_gt = TSNE(n_components=2).fit_transform(angles_gt)
    # angles_emb_flow = TSNE(n_components=2).fit_transform(angles_flow)
    #
    # # Plot the clusters
    # plt.scatter(angles_emb_gt[:, 0], angles_emb_gt[:, 1], alpha=0.2)
    # plt.scatter(angles_emb_flow[:, 0], angles_emb_flow[:, 1], alpha=0.2)
    # plt.legend()
    # plt.show()
    #
    # import umap
    #
    #
    # reducer = umap.UMAP()
    # sample_umap_gt = reducer.fit_transform(samples_gt)
    # sample_umap_flow = reducer.fit_transform(samples_flow)
    #
    # # Plot the clusters
    # plt.scatter(sample_umap_gt[:, 0], sample_umap_gt[:, 1], alpha=0.1)
    # plt.scatter(sample_umap_flow[:, 0], sample_umap_flow[:, 1], alpha=0.1)
    # plt.legend()
    # plt.show()
    #
    # reducer = umap.UMAP()
    # angle_umap_gt = reducer.fit_transform(angles_gt)
    # angle_umap_flow = reducer.fit_transform(angles_flow)
    #
    # # Plot the clusters
    # plt.scatter(angle_umap_gt[:, 0], angle_umap_gt[:, 1], alpha=0.1)
    # plt.scatter(angle_umap_flow[:, 0], angle_umap_flow[:, 1], alpha=0.1)
    # plt.legend()
    # plt.show()


def load_mesh():
    # mesh = meshio.read("icosphere/icoSphereDetail.stl")
    mesh = meshio.read("icosphere/icosphere40.stl")
    mesh.points = mesh.points - mesh.points.mean(0) # center the sphere

    return mesh
# def plot_surface_heatmap (samples, surface_points):
#     assert samples.shape[1] == 3, surface_points.shape[1] == 3
#
#     # Create a kernel density estimate for 3D data
#     kde = gaussian_kde(samples.T)
#
#     # Create a Delaunay triangulation
#     tri = Delaunay(surface_points)
#
#     # Calculate density at each triangle's centroid
#     triangle_density = kde(tri.points.T)
#
#     # Create a 3D scatter plot
#     fig = plt.figure()
#     ax = fig.add_subplot(131, projection='3d')
#     ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5)
#     ax.set_title('3D Scatter Plot')
#
#     # Create a 3D surface plot with color proportional to density
#     ax = fig.add_subplot(132, projection='3d')
#     surface = ax.plot_trisurf(tri.points[:, 0], tri.points[:, 1], tri.points[:, 2], triangles=tri.simplices,
#                               cmap='viridis', linewidth=0.2, antialiaseds=True,
#                               facecolors=plt.cm.viridis(triangle_density))
#     ax.set_title('3D Density on Surface')
#
#     # Add a colorbar
#     fig.colorbar(surface, ax=ax, label='Density')
#
#     ax = fig.add_subplot(133, projection='3d')
#     ax.plot_trisurf(surface_points[:,0], surface_points[:,1], surface_points[:,2], color='b', alpha=0.6)
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#
#     # Show the plot
#     plt.show()
#     breakpoint()
#
# def test_chatgpt (model, samples, device='cuda'):
#     # Create a regular grid in spherical coordinates
#     phi_grid, theta_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50))
#
#     # Convert regular grid to Cartesian coordinates
#     x_grid = np.sin(theta_grid) * np.cos(phi_grid)
#     y_grid = np.sin(theta_grid) * np.sin(phi_grid)
#     z_grid = np.cos(theta_grid)
#
#     mesh_3d_np = np.dstack((x_grid, y_grid, z_grid))
#     mesh_3d = torch.from_numpy(mesh_3d_np).float().to(device)
#     mesh_3d_ = mesh_3d.reshape(-1,3)
#     angles = model.transform_to_noise(mesh_3d_, context=None)
#     uniform_surface, logabsdet = model._transform.inverse(angles, context=None)
#     uniform_surface_np = uniform_surface.reshape(mesh_3d.shape).detach().cpu().numpy()
#
#
#     # plot_surface_heatmap(samples[:5000], uniform_surface_np)
#
#     # Create a kernel density estimate for 3D data
#     kde = gaussian_kde(samples.T)
#
#     # Calculate density at each triangle's centroid
#     triangle_density = kde(uniform_surface_np.reshape(-1,3).T).reshape(uniform_surface_np.shape[:-1])
#
#     # Plot the sphere without mesh using plot_surface
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(uniform_surface_np[...,0], uniform_surface_np[...,1], uniform_surface_np[...,2],
#                     cmap='viridis', linewidth=0.2, antialiaseds=True,
#                     facecolors=plt.cm.viridis(triangle_density))
#
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.show()

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression, lasso_path

def plot_lines_gt (x_gt, y_gt, dim, n_plots, x_model=None, y_mean_model=None, y_l_model=None, y_r_model=None, log_scale=True, norm=1, name_file=None, coarse=1, true_coeff=None, x_label='norm'):
    n_lines = dim // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots(figsize=(14, 14))
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == dim:
                    break
                color = clrs[j % n_lines]
                if any([i is not None for i in [x_model, y_mean_model, y_l_model, y_r_model]]):
                    ax.plot(x_model, y_mean_model[:, j], c=color, alpha=0.7, linewidth=1.5)
                    ax.fill_between(x_model, y_l_model[:, j], y_r_model[:, j], alpha=0.2, facecolor=color)
                ax.plot(x_gt[::coarse], y_gt[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)
                if true_coeff is not None:
                    if true_coeff[j]!=0:
                        ax.axhline(y=true_coeff[j], xmin=x_gt[::coarse].min(), xmax=x_gt[::coarse].max(), c=color, alpha=0.7, linewidth=1.5, linestyle=':')
        if x_label == "norm":
            plt.xlabel(r"$||\beta||_{%s}$" % norm, fontsize=18)
        elif x_label == "lambda":
            plt.xlabel(r"$\lambda$", fontsize=18)
        else:
            raise ValueError("x_label must be either 'norm' or 'lambda'")
        plt.ylabel(r'$\beta$', fontsize=18)
        plt.locator_params(axis='y', nbins=4)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if log_scale: plt.xscale('log')
        if name_file is not None:
            plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
        plt.show()

def plot_betas_lambda(samples, lambdas, X_np, y_np, sigma, gt_only=False, min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', folder_name='./', true_coeff=None):

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
                  y_gt=beta_path, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=None, true_coeff=true_coeff, x_label="lambda")

    # 2) compute solution path for flow samples as a function of norm
    # first compute norm of each sample and then group samples by norm
    all_samples = samples.reshape(-1, X_np.shape[-1])
    try:
        bins_midpoint, bin_means, bin_l, bin_r, samples_norm = refine_samples_by_norm(all_samples, norm, min_bin, max_bin, n_bins, conf=conf)
        plot_lines_gt(x_model=bins_midpoint, y_mean_model=bin_means, y_l_model=bin_l, y_r_model=bin_r, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None, true_coeff=true_coeff)
    except:
        samples_norm, bin_l, bin_means, bin_r = None, None, None, None
        pass

    # 3) compute solution path for flow samples as a function of norm
    # first compute mean norm of samples with same lambda and then order samples by mean norm

    mean_norms, samples_mean, samples_l, samples_r = refine_samples_by_mean_norm(samples, norm, conf)

    plot_lines_gt(x_model=mean_norms, y_mean_model=samples_mean, y_l_model=samples_l, y_r_model=samples_r, x_gt=sklearn_norm,
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None, true_coeff=true_coeff)



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

def plot_cumulative_returns(samples, lambdas, X_np, y_np, conf=0.95, n_plots=1):

    returns = samples @ X_np.T
    returns = np.cumprod(returns, axis=2)
    mean_return = returns.mean(1).T
    l_return = np.quantile(returns, 1 - conf, axis=1).T
    r_return = np.quantile(returns, conf, axis=1).T

    n_lines = lambdas.shape[0]
    clrs = sns.color_palette("husl", n_lines)
    with sns.axes_style("darkgrid"):
        for j in range(n_lines):
            fig, ax = plt.subplots(figsize=(14, 14))
            color = clrs[j % n_lines]
            ax.plot(range(X_np.shape[0]), mean_return[:, j], c=color, alpha=0.7, linewidth=1.5)
            ax.fill_between(range(X_np.shape[0]), l_return[:, j], r_return[:, j], alpha=0.2, facecolor=color)
            ax.plot(range(X_np.shape[0]), np.cumprod(y_np).ravel(), c='r', linestyle='dashed')
            plt.xlabel(r"time", fontsize=18)
            plt.ylabel(r'stock value', fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(f'Cumulative return for alpha={lambdas[j]:.2f}')
            # plt.savefig(f"{name_file}_{j}.pdf", bbox_inches='tight')
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

# def plot_betas_lambda(samples, lambdas, X_np, y_np, sigma, n_bins=51, norm=1, a=0.95, n_plots=1, gt='linear_regression', folder_name='./'):
#
#     # compute ground truth solution path. Either linear regression or logistic regression
#     if gt == 'linear_regression':
#         if norm == 2:
#             beta_sklearn = np.array([Ridge(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
#                                      for alpha in tqdm.tqdm(lambdas * sigma.item()**2)])
#         else:
#             beta_sklearn = np.array([Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_
#                                     for alpha in tqdm.tqdm(lambdas * sigma.item()**2 / X_np.shape[0])])
#         coarse = 1
#     elif gt == 'logistic_regression':
#         coarse = lambdas.shape[0] // 50
#         beta_sklearn = [LogisticRegression(penalty='l1', solver='liblinear', fit_intercept=False, C=1 / c).fit(X_np, y_np).coef_
#                         for c in tqdm.tqdm(lambdas[::coarse])]
#         beta_sklearn = np.array(beta_sklearn).squeeze()
#     else:
#         raise ValueError("Ground truth path (gt) must be either 'linear_regression' or 'logistic_regression'")
#
#     sklearn_norm = np.power(np.power(np.abs(beta_sklearn), norm).sum(1), 1 / norm)
#     sklearn_sorted_idx = sklearn_norm.argsort()
#     sklearn_norm = sklearn_norm[sklearn_sorted_idx]
#     sklearn_sorted = beta_sklearn[sklearn_sorted_idx]
#
#     # 1) compute solution path for flow samples as a function of lambda
#     sample_mean = samples.mean(1)
#     l_quant = np.quantile(samples, 1 - a, axis=1)
#     r_quant = np.quantile(samples, a, axis=1)
#
#     plot_lines_gt(x_model=lambdas, y_mean_model=sample_mean, y_l_model=l_quant, y_r_model=r_quant, x_gt=lambdas,
#                   y_gt=beta_sklearn, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=None)
#
#     # 2) compute solution path for flow samples as a function of norm
#     # first compute norm of each sample and then group samples by norm
#     all_samples = samples.reshape(-1, X_np.shape[-1])
#     all_samples_norms = np.power(np.power(np.abs(all_samples), norm).sum(1), 1 / norm)
#     min_bin, max_bin = all_samples_norms.min(), all_samples_norms.max()
#     bins = np.linspace(min_bin, max_bin, n_bins)
#     bins_midpoint = 0.5 * (bins[1:] + bins[:-1])
#     digitized = np.digitize(all_samples_norms, bins)
#     n_per_bin, _ = np.histogram(all_samples_norms, bins)
#     min_n_per_bin = min(n_per_bin)
#     print(f"min samples per bin {min_n_per_bin}")
#     bin_means = np.array([all_samples[digitized == i][:min_n_per_bin].mean(0) for i in range(1, len(bins))])
#     bin_l_quant = np.array(
#         [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.05, axis=0) for i in range(1, len(bins))])
#     bin_r_quant = np.array(
#         [np.quantile(all_samples[digitized == i][:min_n_per_bin], 0.95, axis=0) for i in range(1, len(bins))])
#
#     plot_lines_gt(x_model=bins_midpoint, y_mean_model=bin_means, y_l_model=bin_l_quant, y_r_model=bin_r_quant, x_gt=sklearn_norm,
#                   y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None)
#
#     samples_norm = np.array([all_samples[digitized == i][:min_n_per_bin] for i in range(1, len(bins))])
#
#     # 3) compute solution path for flow samples as a function of norm
#     # first compute mean norm of samples with same lambda and then order samples by mean norm
#
#     samples_mean = samples.mean(1)
#     samples_l = np.quantile(samples, 1-a, axis=1)
#     samples_r = np.quantile(samples, a, axis=1)
#
#     norms = np.power(np.power(np.abs(samples), norm).sum(-1), 1 / norm)
#     mean_norms = norms.mean(1)
#     norm_idx = mean_norms.argsort()
#     mean_norms_sorted = mean_norms[norm_idx]
#     samples_mean_sorted = samples_mean[norm_idx]
#     samples_l_sorted = samples_l[norm_idx]
#     samples_r_sorted = samples_r[norm_idx]
#
#     plot_lines_gt(x_model=mean_norms_sorted, y_mean_model=samples_mean_sorted, y_l_model=samples_l_sorted, y_r_model=samples_r_sorted,
#                   x_gt=sklearn_norm, y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, log_scale=False, norm=norm, name_file=None)
#
#     return bins, samples_norm, bin_l_quant, bin_means, bin_r_quant


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



def plot_betas_norm(samples_sorted, norm_sorted, X_np, y_np, norm=1, gt="linear_regression", a=0.95, n_plots=1, folder_name='./', true_coeff=None):
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
                  y_gt=sklearn_sorted, dim=X_np.shape[-1], n_plots=n_plots, norm=norm, name_file=None, log_scale=False, coarse=1, true_coeff=true_coeff)


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

def plot_clusters(samples, cond, cond_indices, n_clusters):
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    import tqdm

    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
    }

    for idx in tqdm.tqdm(cond_indices):
        silhouette_coefficients = []
        samples_refined = np.array(samples)
        samples_refined[samples_refined < 1. / samples_refined.shape[-1]] = 0
        for k in range(2, n_clusters):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(samples_refined[idx])
            score = silhouette_score(samples_refined[idx], kmeans.labels_)
            silhouette_coefficients.append(score)

        opt_idx = np.argmax(silhouette_coefficients)
        kmeans = KMeans(n_clusters= opt_idx + 1, **kmeans_kwargs)

        pca = PCA(2)
        df = pca.fit_transform(samples_refined[idx])
        labels = kmeans.fit_predict(df)
        u_labels = np.unique(labels)
        centroids = kmeans.cluster_centers_

        fig, ax = plt.subplots()
        for i in u_labels:
            ax.scatter(df[labels == i, 0], df[labels == i, 1], label=i, alpha=0.2)
            cluster_mean = samples[idx][labels == i].mean(0)
            txt = [f"{value:.2f}" for value in cluster_mean]
            ax.scatter(centroids[i, 0], centroids[i, 1], s=40, color='k')
            ax.annotate(txt, (centroids[i, 0], centroids[i, 1]))

        plt.legend()
        plt.title(f"cond = {cond[idx]:.2e} n_clusters = {opt_idx+1}")
        plt.show()


def plot_samples_3d(samples, s=1, alpha=0.01):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=s, alpha=alpha)
    plt.show()

def plot_angles_3d(samples, args):
    samples = samples.reshape(-1,3)
    samples_cat = cartesian_to_spherical_torch(torch.tensor(samples, device=args.device))
    fig, axs = plt.subplots(1,3, figsize=(14, 14))
    axs[0].hist(samples_cat[:,0].detach().cpu().numpy(),bins=50)
    axs[1].hist(samples_cat[:, 1].detach().cpu().numpy(),bins=50)
    axs[2].hist(samples_cat[:, 2].detach().cpu().numpy(),bins=50)
    plt.show()

def plot_simplex(samples, dim_3, args, shift3to2=False, alpha=0.05, s=2):
    d = samples.shape[-1]
    assert samples.shape[-1] > (3 if dim_3 else 2)
    samples = samples.reshape(-1,d)
    proj = np.ones((d,d))
    proj_svd = np.linalg.svd(proj)
    simp_space = proj_svd.U[:,1:4] if dim_3 else proj_svd.U[:,1:3]
    res = (simp_space.T)[None,] @ samples[...,None]
    res = res[..., 0]

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d' if dim_3 and not shift3to2 else None)

    if dim_3:
        if shift3to2:
            resh = ((proj_svd.U[:,0:1].T)[None,] @ samples[...,None])[...,0]  # this results not in the norm in general, only for the l1 norm of the standard simplex
            ax.scatter(res[:, 0], resh[:, 0], alpha=alpha, s=s)
        else:
            ax.scatter(res[:,0], res[:,1], res[:,2], alpha=alpha, s=s)
    else:
        ax.scatter(res[:,0], res[:,1], alpha=alpha, s=s)

    plt.show()

def to_file(arr, filename):
    if isinstance(arr, np.ndarray):
        df = pd.DataFrame(arr)
    else:
        df = pd.DataFrame(arr.detach().cpu().numpy())

    df.to_csv(filename)
