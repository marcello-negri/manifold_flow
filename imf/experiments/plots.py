import torch
import meshio
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from functools import partial

from imf.experiments.utils_manifold import rnf_forward_logp, rnf_forward_points

def plot_loss (loss):
    plt.figure(figsize=(15,10))
    plt.plot(range(len(loss)), loss[:,0], label="total loss")
    plt.plot(range(len(loss)), loss[:,1], label="log-likelihood")
    plt.plot(range(len(loss)), loss[:,2], label="MSE")
    plt.legend()
    plt.show()

def plot_samples(samples, n_samples=10000):
    if samples.shape[-1] == 3:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(samples[:n_samples, 0], samples[:n_samples, 1], samples[:n_samples, 2], marker=".")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
    else:
        print(f"Skipping 3D plot because d={samples.shape[-1]}")

def plot_icosphere(data, dataset, flow, samples_flow, rnf, samples_rnf, kde=False, device='cuda', args=None, plot_rnf=False):
    # load regular grid in spherical coordinates
    mesh = meshio.read("icoSphereDetail.stl")
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
    points_torch = torch.from_numpy(points).float().to(device)
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
    from matplotlib.colors import Normalize

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = np.array([np.array((x[s], y[s], z[s])).T for s in slices])

    # compute the function in the barycenter
    if kde: values = func(triangles.mean(axis=1).T)
    else: values = func(triangles.mean(axis=1))

    # usual stuff
    norm = Normalize()#vmin=0, vmax=1)
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)

def plot_density (ax, samples, points_surface, title):
    # Create a kernel density estimate for 3D data
    kde_func = gaussian_kde(samples.T)
    mesh = meshio.read("icoSphereDetail.stl")
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
    points_torch = torch.from_numpy(points).float()
    logp = dataset.density(points_torch)
    return logp.exp().detach().cpu().numpy()

def density_flow (points, flow, device="cuda"):
    points_torch = torch.from_numpy(points).float().to(device)
    angles = flow.transform_to_noise(points_torch, context=None)
    logp_flow = flow._distribution.log_prob(angles, context=None)
    uniform_surface_flow, logabsdet = flow._transform.inverse(angles, context=None)
    logp_flow = logp_flow - logabsdet

    return logp_flow.exp().detach().cpu().numpy()

def density_rnf (points, rnf, args):
    logp_rnf = rnf_forward_logp(rnf, torch.from_numpy(points).float(), args)

    return np.exp(logp_rnf)

def plot_logp (ax, logp_function, points_surface, title):
    mesh = meshio.read("icoSphereDetail.stl")
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
