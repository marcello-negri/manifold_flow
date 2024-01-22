import os
import torch
import numpy as np

from imf.experiments.utils_manifold import cartesian_to_spherical_torch
from imf.experiments.vonmises_fisher import vMF, MixvMF


class Dataset():
    def __init__(self, args):
        self.args = args
        self.dataset_suffix = f"_e{self.args.epsilon:.2f}"
        self.dataset_folder = self.args.data_folder + f"/{self.args.dataset}"

        if not os.path.exists(self.dataset_folder):
            os.makedirs(self.dataset_folder)

    def log_density(self, points):
        raise NotImplementedError

    def sample(self, n_samples):
        raise NotImplementedError

    def load_samples(self, overwrite=False):
        self.set_random_seed()

        train_filename = f"{self.dataset_folder}/x_train{self.dataset_suffix}.npy"
        if os.path.isfile(train_filename) and not overwrite:
            samples_train = np.load(train_filename)
        else:
            samples_train = self.sample(self.args.n_samples_dataset).detach().numpy()
            noise_train = np.random.normal(loc=0.0, scale=1.0 * self.args.epsilon, size=self.args.n_samples_dataset)
            samples_train = samples_train + noise_train.reshape(-1,1)
            np.save(train_filename, samples_train)

        test_filename = f"{self.dataset_folder}/x_test{self.dataset_suffix}.npy"
        if os.path.isfile(test_filename) and not overwrite:
            samples_test = np.load(test_filename)
        else:
            samples_test = self.sample(self.args.n_samples_dataset).detach().numpy()
            noise_test = np.random.normal(loc=0.0, scale=1.0 * self.args.epsilon, size=self.args.n_samples_dataset)
            samples_test = samples_test + noise_test.reshape(-1,1)
            np.save(test_filename, samples_test)

        return samples_train, samples_test

    def set_random_seed(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)


class VonMisesFisher(Dataset):
    def __init__(self, args):
        super().__init__(args)

        self.initialize_kappa()
        self.initialize_mu()
        self.initialize_vMF()
        self.dataset_suffix += f"_k{self.args.kappa:.2f}"

    def initialize_kappa(self):
        self.kappa = torch.tensor(self.args.kappa)

    def initialize_mu(self):
        if self.args.mu is None:
            mu = torch.zeros(self.args.datadim)
            mu[-1] = 1.0
            self.mu = mu / self.norm(input=mu, dim=0)
        else:
            mu = torch.ones(self.args.datadim)
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

    def lop_density(self, points):
        if isinstance(points, np.ndarray):
            points = torch.from_numpy(points).float()

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
        self.kappa = torch.ones(self.args.n_mix) * self.args.kappa_mix
        self.alpha = torch.ones(self.args.n_mix) * self.args.alpha_mix

    def initialize_mu(self):
        mu = torch.randn([self.args.n_mix, self.args.datadim])
        mu /= torch.norm(mu, dim=1).reshape(-1, 1)
        self.mu = mu

    def initialize_vMF(self):
        self.vMF = MixvMF(x_dim=self.args.datadim, order=self.args.n_mix)
        self.vMF.set_params(alpha=self.alpha, mus=self.mu, kappas=self.kappa)


class VonMisesFisherMixtureSpiral(VonMisesFisherMixture):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_suffix += f"_t{self.args.n_turns_spiral:d}"

    def initialize_mu(self):
        self.mu = self.generate_spiral_on_sphere()

    def generate_spiral_on_sphere(self):
        # Generate parametric values
        theta = torch.linspace(0, 2 * np.pi * self.args.n_turns_spiral, self.args.n_mix)
        phi = torch.linspace(0, np.pi, self.args.n_mix)

        # Parametric equations for a spherical spiral
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        return torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)


class Uniform(Dataset):
    def __init__(self, args):
        super().__init__(args)

        self.surface_prob = 4 * np.pi

    def sample(self, n_samples):
        samples = torch.randn([n_samples, self.args.datadim])
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)

        return samples

    def log_density(self, points):
        logp = torch.ones(points.shape[0])
        norm_const = - np.log(self.surface_prob)

        return logp * norm_const


class UniformCheckerboard(Uniform):
    def __init__(self, args):
        super().__init__(args)

        assert self.args.datadim == 3

        self.surface_prob = 2 * np.pi
        self.dataset_suffix += f"_nr{self.args.n_theta:d}_nc{self.args.n_phi:d}"

    def sample(self, n_samples):
        # samples 3 times as much points to make sure at least 1/3 fall within the mask
        samples = torch.randn([n_samples * 3, self.args.datadim])
        samples /= torch.norm(samples, dim=1).reshape(-1, 1)
        mask = self.checkerboard_mask(samples)
        samples = samples[mask]

        assert samples.shape[0] > n_samples

        return samples[:n_samples]

    def log_density(self, points):
        logp = torch.ones(points.shape[0]) * -10
        mask = self.checkerboard_mask(points)
        logp[mask] = -np.log(self.surface_prob)

        return logp

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


def create_dataset(args):
    if args.dataset == 'uniform':
        dataset = Uniform(args)
    elif args.dataset == 'uniform_checkerboard':
        dataset = UniformCheckerboard(args)
    elif args.dataset == 'vonmises_fisher_mixture':
        dataset = VonMisesFisherMixture(args)
    elif args.dataset == 'vonmises_fisher_mixture_spiral':
        dataset = VonMisesFisherMixtureSpiral(args)
    elif args.dataset == 'vonmises_fisher':
        dataset = VonMisesFisher(args)
    else:
        raise ValueError('Dataset {} not recognized'.format(args.dataset))

    return dataset