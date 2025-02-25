import os
import torch
import numpy as np
import scipy as sp

from imf.experiments.utils_manifold import cartesian_to_spherical_torch, spherical_to_cartesian_torch
from imf.experiments.vonmises_fisher import vMF, MixvMF

from rpy2 import robjects


class Dataset():
    def __init__(self, args):
        self.args = args
        self.dataset_suffix = f"_d{self.args.datadim:d}_n{self.args.n_samples_dataset:d}_e{self.args.epsilon:.2f}"
        self.dataset_folder = self.args.data_folder + f"/{self.args.dataset}"

        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)

    def log_density(self, points):
        raise NotImplementedError

    def sample(self, n_samples):
        raise NotImplementedError

    def load_samples(self, split, overwrite=False):
        filename = f"{self.dataset_folder}/x_{split}{self.dataset_suffix}.npy"
        if os.path.isfile(filename) and not overwrite:
            samples = np.load(filename)
        else:
            samples = self.sample(self.args.n_samples_dataset).detach().cpu().numpy()
            noise_train = np.random.normal(loc=0.0, scale=1.0 * self.args.epsilon, size=samples.shape)
            samples = samples + noise_train
            np.save(filename, samples)

        return samples


class VonMisesFisher(Dataset):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_kappa()
        self.initialize_mu()
        self.initialize_vMF()
        self.dataset_suffix += f"_k{self.args.kappa:.2f}"

    def initialize_kappa(self):
        self.kappa = torch.tensor(self.args.kappa, device=self.args.device)

    def initialize_mu(self):
        if self.args.mu is None:
            mu = torch.zeros(self.args.datadim, device=self.args.device)
            mu[-1] = 1.0
            self.mu = mu / self.norm(input=mu, dim=0)
        else:
            mu = torch.ones(self.args.datadim, device=self.args.device)
            mu = mu / self.norm(input=mu, dim=0)
            self.mu = mu * self.args.mu

    def initialize_vMF(self):
        self.vMF = vMF(x_dim=self.mu.shape[0], device=self.args.device)
        self.vMF.set_params(mu=self.mu, kappa=self.kappa)

    def norm(self, input, p=2, dim=0, eps=1e-12):
        return input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)

    def sample(self, n_samples):
        samples = self.vMF.sample(N=n_samples, rsf=1)
        samples = self.check_tuple(samples)

        return samples

    def log_density(self, points):
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float().to(self.args.device)

        logp = self.vMF.forward(points)
        logp = self.check_tuple(logp)

        return logp

    def check_tuple(self, obj):
        # the child class VonMisesFisherMixture returns a tuple (obj, obj per component)
        if isinstance(obj, tuple):
            obj = obj[0]

        return obj


class VonMisesFisherMixture(VonMisesFisher):
    def __init__(self, args):
        super().__init__(args)

        self.dataset_suffix += f"_n{self.args.n_mix:d}"

    def initialize_kappa(self):
        self.kappa = torch.ones(self.args.n_mix, device=self.args.device) * self.args.kappa_mix
        self.alpha = torch.ones(self.args.n_mix, device=self.args.device) * self.args.alpha_mix

    def initialize_mu(self):
        mu = torch.randn([self.args.n_mix, self.args.datadim], device=self.args.device)
        mu /= torch.norm(mu, dim=1).reshape(-1, 1)
        self.mu = mu

    def initialize_vMF(self):
        self.vMF = MixvMF(x_dim=self.args.datadim, order=self.args.n_mix).to(device=self.args.device)
        self.vMF.set_params(alpha=self.alpha, mus=self.mu, kappas=self.kappa)


class VonMisesFisherMixtureSpiral(VonMisesFisherMixture):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_suffix += f"_t{self.args.n_turns_spiral:d}"

    def initialize_mu(self):
        self.mu = self.generate_spiral_on_sphere()

    def generate_spiral_on_sphere(self):
        # Generate parametric values
        theta = torch.linspace(0, 2 * np.pi * self.args.n_turns_spiral, self.args.n_mix, device=self.args.device)
        phi = torch.linspace(0, np.pi, self.args.n_mix, device=self.args.device)

        # Parametric equations for a spherical spiral
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)


class Uniform(Dataset):
    def __init__(self, args):
        super().__init__(args)

        self.compute_log_surface()

    def sample(self, n_samples):
        samples = torch.randn([n_samples, self.args.datadim], device=self.args.device)
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)

        return samples * self.args.radius

    def log_density(self, points):
        logp = torch.ones(points.shape[0], device=self.args.device)
        norm_const = - self.log_surface_area

        return logp * norm_const

    def compute_log_surface(self):
        d = self.args.datadim
        r = self.args.radius

        log_const_1 = np.log(2) + 0.5 * d * np.log(np.pi)
        log_const_2 = (d - 1) * np.log(r)
        log_const_3 = - sp.special.loggamma(0.5 * d)

        self.log_surface_area = log_const_1 + log_const_2 + log_const_3


class UniformTorus(Dataset):
    def __init__(self, args):
        super().__init__(args)
        assert args.R > args.r
        self.r = args.r
        self.R = args.R
        self.compute_log_surface()

    def sample(self, n_samples):

        enough = False
        samples = torch.tensor([], device=self.args.device)
        while not enough:
            u = torch.rand(n_samples, device=self.args.device)
            v = torch.rand(n_samples, device=self.args.device)
            w = torch.rand(n_samples, device=self.args.device)

            theta = 2 * torch.pi * u
            phi = 2 * torch.pi * v
            threshold = (self.R + self.r * torch.cos(theta) / (self.R + self.r))
            mask = (w < threshold)
            theta = theta[mask]
            phi = phi[mask]
            x = (self.R + self.r * torch.cos(theta)) * torch.cos(phi)
            y = (self.R + self.r * torch.cos(theta)) * torch.sin(phi)
            z = self.r * torch.sin(theta)

            new_samples = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
            samples = torch.cat((samples, new_samples), dim=0)

            if samples.shape[0] >= n_samples: enough = True

        return samples[:n_samples]

    def log_density(self, points):
        logp = torch.ones(points.shape[0], device=self.args.device)
        norm_const = - self.log_surface_area

        return logp * norm_const

    def compute_log_surface(self):
        surface = (2 * torch.pi) ** 2 * self.R * self.r
        self.log_surface_area = np.log(surface)

class HyperSurface(Dataset):
    def __init__(self, args):
        super().__init__(args)

    def hypersurface(self, points):
        raise NotImplementedError

    def sample(self, n_samples):
        xy = 2 * torch.rand((n_samples, 2)) - 1
        # xy = torch.randn((n_samples, 2))
        z = self.hypersurface(xy)
        samples = torch.cat((xy, z.unsqueeze(1)), dim=1)

        return samples

class HyperSurface1(HyperSurface):
    def __init__(self, args):
        super().__init__(args)

    def hypersurface(self, points):
        x = points[:,0]
        y = points[:,1]
        return - 0.5 * x**2 + 0.5 * y**3

class HyperSurface2(HyperSurface):
    def __init__(self, args):
        super().__init__(args)

    def hypersurface(self, points):
        x = points[:,0]
        y = points[:,1]
        return x**2 + x * y + y**2 - 1

class HyperSurface3(HyperSurface):
    def __init__(self, args):
        super().__init__(args)

    def hypersurface(self, points):
        x = points[:,0]
        y = points[:,1]
        return x**2 - y**2 - 0.5 * x


class UniformCheckerboard(Uniform):
    def __init__(self, args):
        super().__init__(args)

        assert self.args.datadim == 3
        self.dataset_suffix += f"_nr{self.args.n_theta:d}_nc{self.args.n_phi:d}"
        self.check_normalization_constant()
        self._compute_normalization_const()

    def check_normalization_constant(self):
        norm_consts = []
        range_samples = np.logspace(0, 8,20).astype("int")
        for n in range_samples:
            self._compute_normalization_const(n_samples=n)
            norm_consts.append(self.log_norm_const.item())
        import matplotlib.pyplot as plt
        plt.plot(range_samples, norm_consts)
        plt.xscale("log")
        plt.show()

    def _compute_normalization_const(self, n_samples=100_000):
        # should only be called once at initialization
        self.log_norm_const = 0.
        uniform_samples = self.sample_uniform(n_samples)
        log_uniform_samples = self.log_density(uniform_samples)
        self.log_norm_const = torch.logsumexp(log_uniform_samples, dim=-1) - np.log(n_samples) + self.log_surface_area

    def sample(self, n_samples):
        # samples 3 times as much points to make sure at least 1/3 fall within the mask
        samples = torch.randn([n_samples * 3, self.args.datadim], device=self.args.device)
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)
        mask = self.checkerboard_mask(samples).detach().cpu().numpy()
        samples = samples[mask]

        assert samples.shape[0] > n_samples

        return samples[:n_samples]

    def sample_uniform(self, n_samples):
        samples = torch.randn([n_samples, self.args.datadim], device=self.args.device)
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)

        return samples * self.args.radius

    def log_density(self, points):
        assert self.args.n_theta > 1 and self.args.n_phi > 1
        assert self.args.n_phi % 2 == 0

        points_sph = cartesian_to_spherical_torch(points)
        logp = 2 * torch.sin(self.args.n_theta * points_sph[:, 1]) * torch.sin(self.args.n_theta * points_sph[:, 2])

        return logp - self.log_norm_const

    def compute_log_surface(self):
        # density is defined on the mask only, which covers half of the surface area --> 2pi
        self.log_surface_area = np.log(4*np.pi)

    def smooth_checkerboard_mask(self, points_cart):
        assert self.args.n_theta > 1 and self.args.n_phi > 1
        assert self.args.n_phi % 2 == 0

        # idx_rotation = [2, 0, 1]
        idx_rotation = [0, 1, 2]

        rotated_points = points_cart[:, idx_rotation]
        if isinstance(rotated_points, np.ndarray):
            rotated_points = torch.from_numpy(rotated_points).to(points_cart.device).float()

        points_sph = cartesian_to_spherical_torch(rotated_points)
        mask = 2 * torch.sin(self.args.n_theta * points_sph[:, 1]) * torch.sin(self.args.n_theta * points_sph[:, 2])
        return mask

    def checkerboard_mask(self, points_cart):
        assert self.args.n_theta > 1 and self.args.n_phi > 1
        assert self.args.n_phi % 2 == 0

        delta_theta = np.pi / self.args.n_theta
        delta_phi = 2 * np.pi / self.args.n_phi
        idx_rotation = [2, 0, 1]

        rotated_points = points_cart[:, idx_rotation]
        if isinstance(rotated_points, np.ndarray):
            rotated_points = torch.from_numpy(rotated_points).to(points_cart.device).float()

        points_sph = cartesian_to_spherical_torch(rotated_points)
        mask = (points_sph[:, 1] // delta_theta + points_sph[:, 2] // delta_phi) % 2 == 0
        return mask

class LpUniform(Uniform):
    '''
    Note: the surface of a Lp unit ball is not known analytically because it requires a complicated integral
          (see https://en.wikipedia.org/wiki/Volume_of_an_n-ball#Relation_with_surface_area_2),
          so log_density returns a constant value, which is however not the correct one
    '''
    def __init__(self, args):
        super().__init__(args)

        self.alpha = self.args.beta ** (1./self.args.beta)
        robjects.r.source("./utils.R")
        self.sample_gen_norm = robjects.r['sample_gen_norm']
        self.compute_log_surface()

    def sample(self, n_samples):
        samples = self.sample_gen_norm(x=self.args.datadim * n_samples, alpha=self.alpha, beta=self.args.beta, mu=0)
        samples = np.array(samples)
        # breakpoint()
        samples = samples.reshape((n_samples, self.args.datadim))
        samples_norm = self.lp_norm(samples, p=self.args.beta) * self.args.radius


        return torch.from_numpy(samples_norm).float().to(device=self.args.device)

    def lp_norm(self, arr, p):
        norm_stable = sp.linalg.norm(arr, ord=p, axis=1, keepdims=True)
        # norm = np.sum(np.power(np.abs(arr), p), 1)
        # norm = np.power(norm, 1 / p).reshape(-1, 1)
        # breakpoint()
        return arr / norm_stable

    def project(self, samples):
        samples_torch = torch.from_numpy(samples).float()
        samples_spherical = cartesian_to_spherical_torch(samples_torch)
        theta = samples_spherical[:, 1:]
        r = self.r_given_theta(theta)
        theta_r = torch.cat([r, theta], dim=1)
        outputs = spherical_to_cartesian_torch(theta_r)

        return outputs.detach().cpu().numpy()

    def r_given_theta(self, theta):
        assert theta.shape[1] >= 2
        eps = 1e-10

        r_theta = torch.cat((torch.ones_like(theta[:,:1]), theta), dim=1)
        cartesian = spherical_to_cartesian_torch(r_theta)
        p_norm = torch.linalg.vector_norm(cartesian, ord=self.args.beta, dim=1)
        r = 1 / (p_norm + eps)

        return r.unsqueeze(-1)

    def compute_log_surface(self):
        # NOTE: this is exact only for beta=1
        if self.args.beta == 1:
            d = self.args.datadim - 1
            r = self.args.radius
            log_area_simplex = 0.5 * np.log(d + 1) - sp.special.loggamma(d+1)
            self.log_surface_area = (d + 1) * np.log(2.) + log_area_simplex + d * np.log(r)
        else:
            self.log_surface_area = 1.


class DeformedSphereDataset1(Uniform):
    def __init__(self, args):
        super().__init__(args)

        assert self.args.datadim == 3
        self.dataset_suffix += f"_dataset1_manifold_type{self.args.manifold_type:d}"

        self.check_normalization_constant()
        self._compute_normalization_const()

    def check_normalization_constant(self):
        norm_consts = []
        range_samples = np.logspace(0, 8,20).astype("int")
        for n in range_samples:
            self._compute_normalization_const(n_samples=n)
            norm_consts.append(self.log_norm_const.item())
        import matplotlib.pyplot as plt
        plt.plot(range_samples, norm_consts)
        plt.xscale("log")
        plt.show()

    def _compute_normalization_const(self, n_samples=100_000):
        # should only be called once at initialization
        self.log_norm_const = 0.
        uniform_samples = self.sample_uniform(n_samples)
        log_uniform_samples = self.log_density(uniform_samples)
        self.log_norm_const = torch.logsumexp(log_uniform_samples, dim=-1) - np.log(n_samples) + self.log_surface_area

    def sample_uniform(self, n_samples):
        samples = torch.randn([n_samples, self.args.datadim], device=self.args.device)
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)

        return samples * self.args.radius

    def log_density(self, points):
        # idx_rotation = [2, 0, 1]
        points_sph = cartesian_to_spherical_torch(points)#[:, idx_rotation])

        # logp = 2 * torch.sin(3*points_sph[:, 1]) * torch.sin(3*points_sph[:, 2])

        # logp = torch.sin(points_sph[:, 1]) * torch.sin(points_sph[:, 2])
        if self.args.manifold_type == 2:
            logp = 0.5*torch.sin(3*points_sph[:, 2]) - self.log_norm_const# * torch.sin(6*points_sph[:, 2])
        if self.args.manifold_type == 4:
            logp = torch.exp(torch.sin(4*points_sph[:, 1]) * torch.sin(4*points_sph[:, 2]))
            # logp = torch.sin(4*points_sph[:, 1]) * torch.sin(4*points_sph[:, 2])
        if self.args.manifold_type == 0:
            logp = 3 * torch.sin(points_sph[:, 1]) * torch.sin(points_sph[:, 2])
        # logp = torch.exp(torch.sin(4*points_sph[:, 1]) * torch.sin(4*points_sph[:, 2]))
        # logp = torch.ones(points.shape[0], device=self.args.device)
        return logp
        # return logp

    def r_given_theta(self, angles):
        assert angles.shape[-1] == 2
        theta = angles[:,:1]
        phi = angles[:,1:2]
        r = theta.new_ones(theta.shape[0], 1) * self.args.radius
        # r = self.network(theta)
        match self.args.manifold_type:
            case 0:
                # ellipsoid
                a, b, c, = 1, 0.9, 0.8
                r = 1. / torch.sqrt(
                    ((torch.cos(phi) / a) ** 2 + (torch.sin(phi) / b) ** 2) * torch.sin(theta) ** 2 + (
                                torch.cos(theta) / c) ** 2)
            case 1:
                # heart-like
                r = 1 + 0.05 * torch.sin(5 * theta) ** 2 + 0.2 * torch.sin(theta) * torch.abs(torch.sin(phi)) * torch.cos(phi)
            case 2:
                # weird oscillations
                r = 0.6 + 0.05 * torch.sin(5 * theta) ** 2 + 0.05 * torch.sin(3 * theta) * torch.sin(2 * phi)
            case 3:
                # ridge squash
                focus_equator = 5
                num_ridges = 6
                focus_ridges = 3
                r = 0.6 + 0.5 * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) + 0.3 * torch.sin(
                    num_ridges * phi) * torch.exp(
                    - focus_ridges * (1.0 - torch.sin(theta)))
            case 4:
                # spirals
                spiral_speed = 3
                num_spirals = 3  # this times 2
                spiral_power = 2
                focus_equator = 2
                amplitude = 0.15
                r = 1.0 + amplitude * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) * torch.sin(
                    num_spirals * phi + spiral_speed * theta) ** spiral_power
            case 5:
                # spiral spikes
                spiral_speed = 5
                num_spirals = 4  # this times 2
                spiral_power = 2
                focus_equator = 3
                spike_separation = 7  # this number of spikes per spiral. should be odd
                spike_power = 1.0
                amplitude = 0.4
                r = 1.0 + amplitude * torch.exp(- focus_equator * (1.0 - torch.sin(theta))) * torch.abs(
                    torch.sin(spike_separation * theta)) ** spike_power * torch.sin(
                    num_spirals * phi + spiral_speed * theta) ** spiral_power

        return r

def create_dataset(args):
    if args.dataset == 'uniform':
        dataset = Uniform(args)
    elif args.dataset == 'uniform_checkerboard':
        dataset = UniformCheckerboard(args)
    elif args.dataset == 'uniform_torus':
        dataset = UniformTorus(args)
    elif args.dataset == "hypersurface1":
        dataset = HyperSurface1(args)
    elif args.dataset == "hypersurface2":
        dataset = HyperSurface2(args)
    elif args.dataset == "hypersurface3":
        dataset = HyperSurface3(args)
    elif args.dataset == 'lp_uniform':
        dataset = LpUniform(args)
    elif args.dataset == 'vonmises_fisher_mixture':
        dataset = VonMisesFisherMixture(args)
    elif args.dataset == 'vonmises_fisher_mixture_spiral':
        dataset = VonMisesFisherMixtureSpiral(args)
    elif args.dataset == 'vonmises_fisher':
        dataset = VonMisesFisher(args)
    elif args.dataset == 'deformed_sphere1':
        dataset = DeformedSphereDataset1(args)
    else:
        raise ValueError('Dataset {} not recognized'.format(args.dataset))

    return dataset

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

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


# def generate_regression_dataset(n_samples, n_features, noise_std=0.1):
#     # Generate random feature matrix X and regression parameters beta
#     X = np.random.randn(n_samples, n_features)
#     true_beta = np.random.randn(n_features)
#
#     # Generate target variable y with noise
#     y = np.dot(X, true_beta) + noise_std * np.random.randn(n_samples)
#
#     return X, y, true_beta


def generate_regression_dataset(n_samples, n_features, n_non_zero, noise_std):
    assert n_features > n_non_zero

    # Generate non-zero coefficients randomly
    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    coefficients[non_zero_indices] = np.random.normal(0, 1, n_non_zero)  # Random non-zero coefficients

    # Generate data matrix X from a Gaussian distribution with covariance matrix sampled from a Wishart distribution
    scale_matrix = np.eye(n_features)  # Identity matrix as the scale matrix
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    # Sample data matrix X from a multivariate Gaussian distribution with zero mean and covariance matrix
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)

    # Generate response variable y
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std**2, n_samples)  # Linear regression model with Gaussian noise

    # compute regression parameters
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y - X @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return X, y, coefficients

def generate_regression_dataset_positive_coeff(n_samples, n_features, n_non_zero, noise_std):
    assert n_features > n_non_zero

    # Generate non-zero coefficients randomly
    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    random_coeff = np.random.rand(n_non_zero)
    coefficients[non_zero_indices] = random_coeff / random_coeff.sum()  # Random non-zero coefficients

    # Generate data matrix X from a Gaussian distribution with covariance matrix sampled from a Wishart distribution
    scale_matrix = np.eye(n_features)  # Identity matrix as the scale matrix
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    # Sample data matrix X from a multivariate Gaussian distribution with zero mean and covariance matrix
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)
    X -= X.mean(0)
    X /= X.std(0)

    # Generate response variable y
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std**2, n_samples)  # Linear regression model with Gaussian noise

    # compute regression parameters
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y - X @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return X, y, coefficients

