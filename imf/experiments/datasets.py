import os
import torch
import numpy as np
import scipy as sp
import rpy2.robjects as robjects

from imf.experiments.utils_manifold import cartesian_to_spherical_torch
from imf.experiments.vonmises_fisher import vMF, MixvMF


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
        self.vMF = vMF(x_dim=self.mu.shape[0])
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


class UniformCheckerboard(Uniform):
    def __init__(self, args):
        super().__init__(args)

        assert self.args.datadim == 3
        self.dataset_suffix += f"_nr{self.args.n_theta:d}_nc{self.args.n_phi:d}"

    def sample(self, n_samples):
        # samples 3 times as much points to make sure at least 1/3 fall within the mask
        samples = torch.randn([n_samples * 3, self.args.datadim], device=self.args.device)
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)
        mask = self.checkerboard_mask(samples)
        samples = samples[mask]

        assert samples.shape[0] > n_samples

        return samples[:n_samples]

    def log_density(self, points):
        logp = torch.ones(points.shape[0], device=self.args.device) * -10
        mask = self.checkerboard_mask(points)
        logp[mask] = -self.log_surface_area

        return logp

    def compute_log_surface(self):
        # density is defined on the mask only, which covers half of the surface area --> 2pi
        self.log_surface_area = np.log(2*np.pi)

    def checkerboard_mask(self, points_cart):
        assert self.args.n_theta > 1 and self.args.n_phi > 1
        assert self.args.n_phi % 2 == 0

        delta_theta = np.pi / self.args.n_theta
        delta_phi = 2 * np.pi / self.args.n_phi
        idx_rotation = [2, 0, 1]

        rotated_points = points_cart[:, idx_rotation]
        if isinstance(rotated_points, np.ndarray):
            rotated_points = torch.from_numpy(rotated_points).float()

        points_sph = cartesian_to_spherical_torch(rotated_points).detach().numpy()
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


def create_dataset(args):
    if args.dataset == 'uniform':
        dataset = Uniform(args)
    elif args.dataset == 'uniform_checkerboard':
        dataset = UniformCheckerboard(args)
    elif args.dataset == 'lp_uniform':
        dataset = LpUniform(args)
    elif args.dataset == 'vonmises_fisher_mixture':
        dataset = VonMisesFisherMixture(args)
    elif args.dataset == 'vonmises_fisher_mixture_spiral':
        dataset = VonMisesFisherMixtureSpiral(args)
    elif args.dataset == 'vonmises_fisher':
        dataset = VonMisesFisher(args)
    else:
        raise ValueError('Dataset {} not recognized'.format(args.dataset))

    return dataset