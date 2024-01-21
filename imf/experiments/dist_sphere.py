# code taken from https://github.com/CUAI/Neural-Manifold-Ordinary-Differential-Equations

import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import utils_manifold

import n_sphere
import abc
import torch
import torch.autograd.functional as AF
import numpy as np

# EPS = {torch.float32: 1e-4, torch.float64: 1e-8}
#
# class Manifold(metaclass=abc.ABCMeta):
#
#     @abc.abstractmethod
#     def zero(self, *shape):
#         pass
#
#     @abc.abstractmethod
#     def zero_like(self, x):
#         pass
#
#     @abc.abstractmethod
#     def zero_vec(self, *shape):
#         pass
#
#     @abc.abstractmethod
#     def zero_vec_like(self, x):
#         pass
#
#     @abc.abstractmethod
#     def inner(self, x, u, v, keepdim=False):
#         pass
#
#     def norm(self, x, u, squared=False, keepdim=False):
#         norm_sq = self.inner(x, u, u, keepdim)
#         norm_sq.data.clamp_(EPS[u.dtype])
#         return norm_sq if squared else norm_sq.sqrt()
#
#     @abc.abstractmethod
#     def proju(self, x, u):
#         pass
#
#     def proju0(self, u):
#         return self.proju(self.zero_like(u), u)
#
#     @abc.abstractmethod
#     def projx(self, x):
#         pass
#
#     def egrad2rgrad(self, x, u):
#         return self.proju(x, u)
#
#     @abc.abstractmethod
#     def exp(self, x, u):
#         pass
#
#     def exp0(self, u):
#         return self.exp(self.zero_like(u), u)
#
#     @abc.abstractmethod
#     def log(self, x, y):
#         pass
#
#     def log0(self, y):
#         return self.log(self.zero_like(y), y)
#
#     def dist(self, x, y, squared=False, keepdim=False):
#         return self.norm(x, self.log(x, y), squared, keepdim)
#
#     def pdist(self, x, squared=False):
#         assert x.ndim == 2
#         n = x.shape[0]
#         m = torch.triu_indices(n, n, 1, device=x.device)
#         return self.dist(x[m[0]], x[m[1]], squared=squared, keepdim=False)
#
#     def transp(self, x, y, u):
#         return self.proju(y, u)
#
#     def transpfrom0(self, x, u):
#         return self.transp(self.zero_like(x), x, u)
#
#     def transpto0(self, x, u):
#         return self.transp(x, self.zero_like(x), u)
#
#     def mobius_addition(self, x, y):
#         return self.exp(x, self.transp(self.zero_like(x), x, self.log0(y)))
#
#     @abc.abstractmethod
#     def sh_to_dim(self, shape):
#         pass
#
#     @abc.abstractmethod
#     def dim_to_sh(self, dim):
#         pass
#
#     @abc.abstractmethod
#     def squeeze_tangent(self, x):
#         pass
#
#     @abc.abstractmethod
#     def unsqueeze_tangent(self, x):
#         pass
#
#     @abc.abstractmethod
#     def rand(self, *shape):
#         pass
#
#     @abc.abstractmethod
#     def randvec(self, x, norm=1):
#         pass
#
#     @abc.abstractmethod
#     def __str__(self):
#         pass
#
#     def logdetexp(self, x, u):
#         # very expensive rip
#         if len(u.shape) == 1:
#             return torch.det(AF.jacobian(lambda v: self.exp(x, v), u))
#         else:
#             jacobians = [AF.jacobian(lambda v: self.exp(x[i], v), u[i]) for
#                          i in range(u.shape[0])]
#             return torch.det(torch.stack(jacobians))
#
#     def logdetlog(self, x, y):
#         return -self.logdetexp(x, self.log(x, y))
#
#     def logdetexp0(self, u):
#         return self.logdetexp(self.zero_like(u), u)
#
#     def logdetlog0(self, y):
#         return self.logdetlog(self.zero_like(y), y)
#
# class Sphere(Manifold):
#
#     def __init__(self):
#         super(Sphere, self).__init__()
#
#     def zero(self, *shape, out=None):
#         x = torch.zeros(*shape, out=out)
#         x[..., 0] = -1
#         return x
#
#     def zero_vec(self, *shape, out=None):
#         return torch.zeros(*shape, out=out)
#
#     def zero_like(self, x):
#         y = torch.zeros_like(x)
#         y[..., 0] = -1
#         return y
#
#     def zero_vec_like(self, x):
#         return torch.zeros_like(x)
#
#     def inner(self, x, u, v, keepdim=False):
#         return (u * v).sum(dim=-1, keepdim=keepdim)
#
#     def proju(self, x, u, inplace=False):
#         return u.addcmul(-self.inner(None, x, u, keepdim=True), x)
#
#     def projx(self, x, inplace=False):
#         return x.div(self.norm(None, x, keepdim=True))
#
#     def exp(self, x, u):
#         norm_u = u.norm(dim=-1, keepdim=True)
#         return x * torch.cos(norm_u) + u * sindiv(norm_u)
#
#     def retr(self, x, u):
#         return self.projx(x + u)
#
#     def log(self, x, y):
#         xy = (x * y).sum(dim=-1, keepdim=True)
#         xy.data.clamp_(min=-1 + 1e-6, max=1 - 1e-6)
#         val = torch.acos(xy)
#         return divsin(val) * (y - xy * x)
#
#     def jacoblog(self, x, y):
#         z = (x * y).sum(dim=-1, keepdim=True)
#         z.data.clamp_(min=-1 + 1e-4, max=1 - 1e-4)
#
#         firstterm = firstjacscalar(z.unsqueeze(-1)) * (y - z * x).unsqueeze(-1) * x.unsqueeze(-2)
#         secondterm = divsin(torch.acos(z).unsqueeze(-1)) * (
#                     torch.eye(x.shape[-1]).to(x).unsqueeze(0) - x.unsqueeze(-1) * x.unsqueeze(-2))
#         return firstterm + secondterm
#
#     def dist(self, x, y, squared=False, keepdim=False):
#         inner = self.inner(None, x, y, keepdim=keepdim)
#         inner.data.clamp_(min=-1 + EPS[x.dtype] ** 2, max=1 - EPS[x.dtype] ** 2)
#         sq_dist = torch.acos(inner)
#         sq_dist.data.clamp_(min=EPS[x.dtype])
#
#         return sq_dist.pow(2) if squared else sq_dist
#
#     def rand(self, *shape, out=None, ir=1e-2):
#         x = self.zero(*shape, out=out)
#         u = self.randvec(x, norm=ir)
#         return self.retr(x, u)
#
#     def rand_uniform(self, *shape, out=None):
#         return self.projx(
#             torch.randn(*shape, out=out), inplace=True)
#
#     def rand_ball(self, *shape, out=None):
#         xs_unif = self.rand_uniform(*shape, out=out)
#         rs = torch.rand(*shape[0]).pow_(1 / (self.dim + 1))
#         # rs = rs.reshape(*shape, *((1, ) * len(self.shape)))
#         xs_ball = xs_unif.mul_(rs)
#         return xs_ball
#
#     def randvec(self, x, norm=1):
#         u = torch.randn(x.shape, out=torch.empty_like(x))
#         u = self.proju(x, u, inplace=True)  # "transport" ``u`` to ``x``
#         u.div_(u.norm(dim=-1, keepdim=True)).mul_(norm)  # normalize
#         return u
#
#     def transp(self, x, y, u):
#         yu = torch.sum(y * u, dim=-1, keepdim=True)
#         xy = torch.sum(x * y, dim=-1, keepdim=True)
#         return u - yu / (1 + xy) * (x + y)
#
#     def __str__(self):
#         return "Sphere"
#
#     def sh_to_dim(self, sh):
#         if hasattr(sh, '__iter__'):
#             return sh[-1] - 1
#         else:
#             return sh - 1
#
#     def dim_to_sh(self, dim):
#         if hasattr(dim, '__iter__'):
#             return dim[-1] + 1
#         else:
#             return dim + 1
#
#     def squeeze_tangent(self, x):
#         return x[..., 1:]
#
#     def unsqueeze_tangent(self, x):
#         return torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
#
#     def logdetexp(self, x, u):
#         norm_u = u.norm(dim=-1)
#         val = torch.abs(sindiv(norm_u)).log()
#         return (u.shape[-1] - 2) * val
#
#
# class WrappedNormal(torch.distributions.Distribution):
#     arg_constraints = {
#                 'loc': torch.distributions.constraints.real_vector,
#                 'scale': torch.distributions.constraints.positive
#                 }
#
#     support = torch.distributions.constraints.real
#     has_rsample = True
#
#     def __init__(self, manifold, loc, scale, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.manifold = manifold
#         self.loc = loc
#         self.scale = scale
#         self.dev = self.loc.device
#         self.normal = torch.distributions.Normal(self.manifold.squeeze_tangent(
#             self.manifold.zero_vec_like(loc)), scale)
#
#     @property
#     def mean(self):
#         return self.loc
#
#     @property
#     def stddev(self):
#         return self.scale
#
#     def rsample(self, shape=torch.Size()):
#         # v ~ N(0, \Sigma)
#         v = self.normal.rsample(shape)
#         # u = PT_{mu_0 -> mu}([0, v_tilde])
#         # z = exp_{mu}(u)
#         u = self.manifold.transp(self.manifold.zero_like(self.loc), self.loc,
#                                 self.manifold.unsqueeze_tangent(v))
#         z = self.manifold.exp(self.loc, u)
#         return z
#
#     def log_prob(self, z):
#         # log(z) = log p(v) - log det [(\partial / \partial v) proj_{\mu}(v)]
#         u = self.manifold.log(self.loc, z)
#         v = self.manifold.transp(self.loc, self.manifold.zero_like(self.loc), u)
#         v = self.manifold.squeeze_tangent(v)
#         n_logprob = self.normal.log_prob(v).sum(dim=-1)
#         logdet = self.manifold.logdetexp(self.loc, u)
#         assert n_logprob.shape == logdet.shape
#         log_prob = n_logprob - logdet
#         return log_prob
#
#     def rsample_log_prob(self, shape=torch.Size()):
#         z = self.rsample(shape)
#         return z, self.log_prob(z)
#
#
# ## Utility methods for transformations
#
# def xyz_to_spherical(xyz):
#     # assume points live on hypersphere
#     x = xyz[:, 0]
#     y = xyz[:, 1]
#     z = xyz[:, 2]
#     lonlat = np.empty((xyz.shape[0], 2))
#     phi = np.arctan2(y, x)
#     phi[y < 0] = phi[y < 0] + 2 * np.pi
#     lonlat[:, 0] = phi
#     lonlat[:, 1] = np.arctan2(np.sqrt(x ** 2 + y ** 2), z)
#     return lonlat
#
#
# def spherical_to_xyz(lonlat):
#     # lonlat[:,0] is azimuth phi in [0,2pi]
#     # lonlat[:,1] is inclination theta in [0,pi]
#
#     phi = lonlat[:, 0]
#     theta = lonlat[:, 1]
#     x = torch.sin(theta) * torch.cos(phi)
#     y = torch.sin(theta) * torch.sin(phi)
#     z = torch.cos(theta)
#
#     xyz = torch.stack([x, y, z], dim=1)
#     detjac = -torch.sin(theta)
#     return xyz, detjac
#
#
# def plot_sphere_density(lonlat, probs, npts, uniform=False):
#     # lon in [0,2pi], lat in [0,pi]
#     fig = plt.figure(figsize=(3, 2), dpi=200)
#     proj = ccrs.Mollweide()
#     ax = fig.add_subplot(111, projection=proj)
#     lon, lat = lonlat[:, 0], lonlat[:, 1]
#     lon = lon.cpu().numpy().reshape(npts, npts)
#     lat = lat.cpu().numpy().reshape(npts, npts)
#     lon -= np.pi
#     lat -= np.pi / 2
#     probs = probs.cpu().numpy().reshape(npts, npts)
#     if not uniform:
#         ax.pcolormesh(lon * 180 / np.pi, lat * 180 / np.pi, probs,
#                       transform=ccrs.PlateCarree(), cmap='magma')
#     else:  # uniform color
#         colormap = plt.cm.get_cmap('magma')
#         norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
#         probs[probs > 0] = .5
#         ax.pcolormesh(lon * 180 / np.pi, lat * 180 / np.pi, probs,
#                       transform=ccrs.PlateCarree(), cmap='magma',
#                       norm=norm)
#
#     plt.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_global()
#     plt.show()
#
#
# ## Data fetching
#
# def data_gen_sphere(dataset, n_samples):
#     if dataset == '1wrapped':
#         center = -torch.ones(3)
#         center = Sphere().projx(center)
#         loc = center.repeat(n_samples, 1)
#         scale = torch.ones(n_samples, 2) * torch.tensor([.3, .3])
#         distr = WrappedNormal(Sphere(), loc, scale)
#         samples = distr.rsample()
#
#     elif dataset == '4wrapped':
#         one = torch.ones(3)
#         oned = torch.ones(3)
#         oned[2] = -1
#         centers = [one, -one, oned, -oned]
#         centers = [Sphere().projx(center) for center in centers]
#         n = n_samples // len(centers)
#         scales = [torch.ones(n, 2) * torch.tensor([.3, .3]) for _ in range(len(centers))]
#         distrs = []
#         for i in range(len(centers)):
#             loc = centers[i].repeat(n, 1)
#             distrs.append(WrappedNormal(Sphere(), loc, scales[i]))
#         samples = distrs[0].rsample()
#         for i in range(1, len(distrs)):
#             samples = torch.cat([samples, distrs[i].rsample()], dim=0)
#
#     elif dataset == 'bigcheckerboard':
#         s = np.pi / 2 - .2  # long side length
#         offsets = [(0, 0), (s, s / 2), (s, -s / 2), (0, -s), (-s, s / 2), (-s, -s / 2), (-2 * s, 0), (-2 * s, -s)]
#         offsets = torch.tensor([o for o in offsets])
#
#         # (x,y) ~ uniform([pi,pi + s] times [pi/2, pi/2 + s/2])
#         x1 = torch.rand(n_samples) * s + np.pi
#         x2 = torch.rand(n_samples) * s / 2 + np.pi / 2
#
#         samples = torch.stack([x1, x2], dim=1)
#         off = offsets[torch.randint(len(offsets), size=(n_samples,))]
#
#         samples += off
#
#         samples, _ = spherical_to_xyz(samples)
#
#     return samples
#
#
# ## Visualize GT or model distributions
#
# def plot_distr(distr=None, res_npts=500, save_fig=True, model=None, device=None, base_distr=None,
#                namestr='sphere_model'):
#     on_mani, lonlat, log_detjac = make_grid_sphere(res_npts)
#
#     if distr == '1wrapped':
#         probs = true_1wrapped_probs(lonlat)
#     elif distr == '4wrapped':
#         probs = true_4wrapped_probs(lonlat, res_npts)
#     elif distr == 'bigcheckerboard':
#         probs = true_bigcheckerboard_probs(lonlat)
#     elif distr == 'model':
#         probs = model_probs(model, on_mani, log_detjac, device, base_distr)
#
#     plot_sphere_density(lonlat, probs, res_npts, uniform=True if distr == 'bigcheckerboard' else False)
#
#     if save_fig:
#         print(f'Saved to: {namestr}.png')
#         plt.savefig(f'{namestr}.png')
#
#
# def true_1wrapped_probs(lonlat):
#     xyz, _ = spherical_to_xyz(lonlat)
#
#     center = -torch.ones(3)
#     center = Sphere().projx(center)
#     loc = center.repeat(xyz.shape[0], 1)
#     scale = torch.ones(xyz.shape[0], 2) * torch.tensor([.3, .3])
#     distr = WrappedNormal(Sphere(), loc, scale)
#
#     probs = torch.exp(distr.log_prob(xyz))
#     return probs
#
#
# def true_4wrapped_probs(lonlat, npts):
#     xyz, _ = spherical_to_xyz(lonlat)
#
#     one = torch.ones(3)
#     oned = torch.ones(3)
#     oned[2] = -1
#     centers = [one, -one, oned, -oned]
#     centers = [Sphere().projx(center) for center in centers]
#     n = npts * npts
#     scale = torch.tensor([.3, .3])
#
#     scales = [torch.ones(n, 2) * scale for _ in range(len(centers))]
#     distrs = []
#     for i in range(len(centers)):
#         loc = centers[i].repeat(n, 1)
#         distrs.append(WrappedNormal(Sphere(), loc, scales[i]))
#
#     probs = torch.exp(distrs[0].log_prob(xyz))
#     for i in range(1, len(distrs)):
#         probs += torch.exp(distrs[i].log_prob(xyz))
#     probs /= len(distrs)
#     return probs
#
#
# def true_bigcheckerboard_probs(lonlat):
#     s = np.pi / 2 - .2  # long side length
#
#     def in_board(z, s):
#         # z is lonlat
#         lon = z[0]
#         lat = z[1]
#         if np.pi <= lon < np.pi + s or np.pi - 2 * s <= lon < np.pi - s:
#             return np.pi / 2 <= lat < np.pi / 2 + s / 2 or np.pi / 2 - s <= lat < np.pi / 2 - s / 2
#         elif np.pi - 2 * s <= lon < np.pi + 2 * s:
#             return np.pi / 2 + s / 2 <= lat < np.pi / 2 + s or np.pi / 2 - s / 2 <= lat < np.pi / 2
#         else:
#             return 0
#
#     probs = torch.zeros(lonlat.shape[0])
#     for i in range(lonlat.shape[0]):
#         probs[i] = in_board(lonlat[i, :], s)
#
#     probs /= torch.sum(probs)
#     return probs
#
#
# def model_probs(model, on_mani, log_detjac, device, base_distr):
#     if device:
#         on_mani = on_mani.to(device)
#         log_detjac = log_detjac.to(device)
#
#     z, logprob = model(on_mani)
#
#     val = base_distr.log_prob(z)
#     val += logprob.detach()
#     val += log_detjac
#     probs = torch.exp(val)
#     return probs.detach()
#
#
# def make_grid_sphere(npts):
#     lon = torch.linspace(0, 2 * np.pi, npts)
#     lat = torch.linspace(0, np.pi, npts)
#     Lon, Lat = torch.meshgrid((lon, lat))
#     lonlat = torch.stack([Lon.flatten(), Lat.flatten()], dim=1)
#     xyz, detjac = spherical_to_xyz(lonlat)
#     log_detjac = torch.log(torch.abs(detjac))
#     on_mani = xyz
#     return on_mani, lonlat, log_detjac


def generate_sphere_checkerboard(filename_train, filename_test, n_samples=1000, n_theta=6, n_phi=6):
    # samples_train = data_gen_sphere("bigcheckerboard", n_samples=n_samples)
    # samples_train_np = samples_train.detach().cpu().numpy()s
    samples_train_np = generate_checkerboard_points(n_samples=n_samples, n_theta=n_theta, n_phi=n_phi)
    np.save(filename_train, samples_train_np)

    # samples_test = data_gen_sphere("bigcheckerboard", n_samples=n_samples)
    # samples_test_np = samples_test.detach().cpu().numpy()
    samples_test_np = generate_checkerboard_points(n_samples=n_samples, n_theta=n_theta, n_phi=n_phi)
    np.save(filename_test, samples_test_np)

    return samples_train_np, samples_test_np


def evaluate_sphere_checkerboard(test_data, n_theta=6, n_phi=6):
    logp_gt = -10 * np.ones(test_data.shape[0])
    mask = checkerboard_mask(test_data, n_theta, n_phi, numpy=False)
    logp_gt[mask] = - np.log(2 * np.pi)

    # lonlat = xyz_to_spherical(test_data)
    # logp_gt = true_bigcheckerboard_probs(torch.from_numpy(lonlat).float())
    # logp_gt.detach().cpu().numpy()

    return logp_gt

def checkerboard_mask(points_cart, n_theta, n_phi, numpy=True):
    assert n_theta > 1 and n_phi > 1
    assert n_phi % 2 == 0
    delta_theta = np.pi / n_theta
    delta_phi = 2 * np.pi / n_phi
    idx_rotation = [1,2,0]
    if numpy:
        points_cart_torch = torch.from_numpy(points_cart[:,idx_rotation]).float()
    else:
        points_cart_torch = points_cart[:,idx_rotation]

    points_sph = utils_manifold.cartesian_to_spherical_torch(points_cart_torch[:,idx_rotation]).detach().numpy()

    mask = (points_sph[:,1] // delta_theta + points_sph[:,2] // delta_phi) % 2 == 0

    return mask

def generate_checkerboard_points(n_samples, n_theta, n_phi):
    points = np.random.randn(n_samples * 10, 3)
    points /= np.linalg.norm(points, axis=1).reshape(-1, 1)
    mask = checkerboard_mask(points, n_theta, n_phi)
    points = points[mask]

    assert points.shape[0] > n_samples

    return points[:n_samples]




