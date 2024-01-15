import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from scipy.spatial import Delaunay
import torch
import meshio
import vtk

def plot_surface_heatmap (samples, surface_points):
    assert samples.shape[1] == 3, surface_points.shape[1] == 3

    # Create a kernel density estimate for 3D data
    kde = gaussian_kde(samples.T)

    # Create a Delaunay triangulation
    tri = Delaunay(surface_points)

    # Calculate density at each triangle's centroid
    triangle_density = kde(tri.points.T)

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5)
    ax.set_title('3D Scatter Plot')

    # Create a 3D surface plot with color proportional to density
    ax = fig.add_subplot(132, projection='3d')
    surface = ax.plot_trisurf(tri.points[:, 0], tri.points[:, 1], tri.points[:, 2], triangles=tri.simplices,
                              cmap='viridis', linewidth=0.2, antialiaseds=True,
                              facecolors=plt.cm.viridis(triangle_density))
    ax.set_title('3D Density on Surface')

    # Add a colorbar
    fig.colorbar(surface, ax=ax, label='Density')

    ax = fig.add_subplot(133, projection='3d')
    ax.plot_trisurf(surface_points[:,0], surface_points[:,1], surface_points[:,2], color='b', alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()
    breakpoint()

def test_chatgpt (model, samples, device='cuda'):
    # Create a regular grid in spherical coordinates
    phi_grid, theta_grid = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50))

    # Convert regular grid to Cartesian coordinates
    x_grid = np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = np.cos(theta_grid)

    mesh_3d_np = np.dstack((x_grid, y_grid, z_grid))
    mesh_3d = torch.from_numpy(mesh_3d_np).float().to(device)
    mesh_3d_ = mesh_3d.reshape(-1,3)
    angles = model.transform_to_noise(mesh_3d_, context=None)
    uniform_surface, logabsdet = model._transform.inverse(angles, context=None)
    uniform_surface_np = uniform_surface.reshape(mesh_3d.shape).detach().cpu().numpy()


    # plot_surface_heatmap(samples[:5000], uniform_surface_np)

    # Create a kernel density estimate for 3D data
    kde = gaussian_kde(samples.T)

    # Calculate density at each triangle's centroid
    triangle_density = kde(uniform_surface_np.reshape(-1,3).T).reshape(uniform_surface_np.shape[:-1])

    # Plot the sphere without mesh using plot_surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(uniform_surface_np[...,0], uniform_surface_np[...,1], uniform_surface_np[...,2],
                    cmap='viridis', linewidth=0.2, antialiaseds=True,
                    facecolors=plt.cm.viridis(triangle_density))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def map_colors(p3dc, func, cmap='viridis'):
    """
Color a tri-mesh according to a function evaluated in each barycentre.

    p3dc: a Poly3DCollection, as returned e.g. by ax.plot_trisurf
    func: a single-valued function of 3 arrays: x, y, z
    cmap: a colormap NAME, as a string

    Returns a ScalarMappable that can be used to instantiate a colorbar.
    """

    from matplotlib.cm import ScalarMappable, get_cmap
    from matplotlib.colors import Normalize
    from numpy import array

    # reconstruct the triangles from internal data
    x, y, z, _ = p3dc._vec
    slices = p3dc._segslices
    triangles = array([array((x[s], y[s], z[s])).T for s in slices])

    # compute the function in the barycentres
    values = func(triangles.mean(axis=1).T)

    # usual stuff
    norm = Normalize(vmin=0, vmax=1)
    colors = get_cmap(cmap)(norm(values))

    # set the face colors of the Poly3DCollection
    p3dc.set_fc(colors)

    # if the caller wants a colorbar, they need this
    return ScalarMappable(cmap=cmap, norm=norm)

def plot_icosphere(data, simulator, flow, samples_flow, rnf, samples_rnf, device='cuda'):
    # load regular grid in spherical coordinates
    mesh = meshio.read("./icoSphereDetail.stl")
    points = mesh.points

    # plot ground truth density
    fig = plt.figure(figsize=(15, 15))
    ax11 = fig.add_subplot(231, projection='3d')
    plot_samples(ax11, data, title="ground truth")
    ax21 = fig.add_subplot(234, projection='3d')
    plot_density(ax21, simulator, data, points, title="ground truth")

    # plot manifold flow density
    # project point on the learnt surface
    points_torch = torch.from_numpy(points).float().to(device)
    angles = flow.transform_to_noise(points_torch, context=None)
    uniform_surface_flow, logabsdet = flow._transform.inverse(angles, context=None)
    points_surface_flow = uniform_surface_flow.detach().cpu().numpy()

    ax12 = fig.add_subplot(232, projection='3d')
    plot_samples(ax12, samples_flow, title="ours")
    ax22 = fig.add_subplot(235, projection='3d')
    plot_density(ax22, simulator, samples_flow, points_surface_flow, title="ours")

    # plot rectangular normalizing flos density
    # project point on the learnt surface

    uniform_surface_rnf, log_prob, u = rnf.forward(points_torch.detach().cpu())
    points_surface_rnf = uniform_surface_rnf.detach().cpu().numpy()
    ax13 = fig.add_subplot(233, projection='3d')
    plot_samples(ax13, samples_rnf, title="rnf")
    ax23 = fig.add_subplot(236, projection='3d')
    plot_density(ax23, simulator, samples_rnf, points_surface_rnf, title="rnf")

    plt.show()

def plot_density (ax, simulator, samples, points_surface, title):
    # Create a kernel density estimate for 3D data
    kde = gaussian_kde(samples.T)
    mesh = meshio.read("./icoSphereDetail.stl")
    triangles = mesh.cells[0].data

    trisurf = ax.plot_trisurf(points_surface[:, 0], points_surface[:, 1], points_surface[:, 2], triangles=triangles)

    # change to facecolor
    mappable = map_colors(trisurf, kde, 'viridis')
    plt.colorbar(mappable, shrink=0.67, aspect=16.7)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

def plot_samples(ax, samples, title):
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], marker=".")
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)