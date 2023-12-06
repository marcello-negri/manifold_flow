import numpy as np
import torch
import time
import scipy as sp
from datetime import timedelta
import matplotlib.pyplot as plt

import tqdm
from enflows.distributions import StandardNormal, MOG, Uniform
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.flows.base import Flow
from enflows.transforms.injective import ConstrainedAnglesSigmoid, ClampedAngles, LearnableManifoldFlow, SphereFlow


def spherical_to_cartesian_torch(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)

    assert arr.shape[1] >= 2
    r = arr[:, :1]
    angles = arr[:, 1:]

    sin_prods = torch.cumprod(torch.sin(angles), dim=1)
    x1 = r * torch.cos(angles[:, :1])
    xs = r * sin_prods[:, :-1] * torch.cos(angles[:, 1:])
    xn = r * sin_prods[:, -1:]

    return torch.cat((x1, xs, xn), dim=1)


def cartesian_to_spherical_torch(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    eps = 1e-5
    assert arr.shape[-1] >= 2
    radius = torch.linalg.norm(arr, dim=-1)
    flipped_cumsum = torch.cumsum(torch.flip(arr ** 2, dims=(-1,)), dim=-1)
    sqrt_sums = torch.flip(torch.sqrt(flipped_cumsum + eps), dims=(-1,))[..., :-1]
    angles = torch.acos(arr[..., :-1] / (sqrt_sums + eps))
    last_angle = ((arr[..., -1] >= 0).float() * angles[..., -1] + \
                  (arr[..., -1] < 0).float() * (2 * np.pi - angles[..., -1]))

    return torch.cat((radius.unsqueeze(-1), angles[..., :-1], last_angle.unsqueeze(-1)), dim=-1)

def logabsdet_sph_to_car(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    eps = 1e-8
    n = arr.shape[1]
    r = arr[:, -1]
    angles = arr[:, :-2]
    sin_angles = torch.sin(angles)
    sin_exp = torch.arange(n - 2, 0, -1).to(arr.device)

    logabsdet_r = (n - 1) * torch.log(r + eps)
    logabsdet_sin = torch.sum(sin_exp * torch.log(torch.abs(sin_angles) + eps), dim=1)

    return logabsdet_r + logabsdet_sin


def build_flow_manifold(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    base_dist = StandardNormal(shape=[flow_dim - 1])

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim-1))

        transformation_layers.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
                                             n_sigmoids=30)
            )
        )

        # transformation_layers.append(
        # InverseTransform(
        #        Sigmoid()
        #    )
        # )

        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim - 1)
            )
        )

    # transformation_layers.append(
    #    InverseTransform(
    #            CompositeTransform([
    #                ScalarScale(scale=2, trainable=False)])#,
    #                #ScalarShift(shift=-1, trainable=False)])
    #        )
    # )

    # transformation_layers.append(
    #   InverseTransform(
    #       Sigmoid()
    #   )
    # )

    transformation_layers.append(
        InverseTransform(
            ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True)
        )
    )

    # transformation_layers.append(
    #     InverseTransform(
    #         ClampedAngles(eps=1e-5)
    #     )
    # )

    transformation_layers.append(
        InverseTransform(
            LearnableManifoldFlow(n=flow_dim - 1)
            # SphereFlow(n=flow_dim - 1)
        )
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(device)

    return flow


def train_model(model, data, epochs=2_001, lr=1e-3, r=1., context=None, device="cuda", **kwargs):
    # optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':log_sigma, 'lr':1e-2}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    loss = []
    try:
        start_time = time.monotonic()
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            data_base = model.transform_to_noise(data, context)
            log_prob = model._distribution.log_prob(data_base, context)
            data_manifold, logabsdet = model._transform.inverse(data_base, context)
            log_prob = log_prob - logabsdet

            # q_samples, q_log_prob = model.sample_and_log_prob(num_samples=sample_size)
            if torch.any(torch.isnan(data_base)): breakpoint()
            if torch.any(torch.isnan(data_manifold)): breakpoint()

            # log_lik = uniform_p_norm(beta=q_log_prob_beta)
            # kl_div = torch.mean(q_log_prob_beta - log_lik)
            # kl_div.backward()

            # assert not torch.any(torch.isnan(q_log_prob))

            # log_prior = log_prior_angles(q_samples, torch.tensor(0.5))
            # assert not torch.any(torch.isnan(log_prior))
            log_likelihood = -torch.mean(log_prob)
            mse_loss = torch.norm(data_manifold - data, dim=1).mean()
            # kl_div = torch.mean(q_log_prob)
            total_loss = log_likelihood + 100 * mse_loss
            total_loss.backward()
            # log_likelihood.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .001)
            optimizer.step()

            # loss.append(torch.mean(q_log_prob_beta - log_lik).cpu().detach().numpy())
            # loss.append(torch.mean(q_log_prob).cpu().detach().numpy())
            log_lik_ = log_likelihood.cpu().detach().numpy()
            mse_ = mse_loss.cpu().detach().numpy()
            loss_ = total_loss.cpu().detach().numpy()
            loss.append([loss_, log_lik_, mse_])

            print(f"Training loss at step {epoch}: {loss_:.5f} (neg logL: {log_lik_:.4f}, MSE: {mse_:.5f})")
            # if epoch + 1 == epochs:
            #     samples, log_probs = generate_samples(model)
            #     mse_log_probs = evaluate_uniform_on_sphere(d=data.shape[-1], r=r, samples=samples, loglik=log_probs)
            #     print(f"MSE: {mse_log_probs:.2f}")
            #     plot_pairwise_angle_distribution(samples)
            #     plot_samples(samples)

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, np.array(loss)


def plot_loss (loss):
    plt.figure(figsize=(15,10))
    plt.plot(range(len(loss)), loss[:,0], label="total loss")
    plt.plot(range(len(loss)), loss[:,1], label="log-likelihood")
    plt.plot(range(len(loss)), loss[:,2], label="MSE")
    plt.legend()
    plt.show()

# def generate_spherical_with_noise(n, d, r, std, device="cuda"):
#     phi = torch.rand(n, 1) * 2 * np.pi
#     radius = torch.ones(n, 1) * r
#     if d == 1:
#         angles = torch.cat((radius, phi), dim=1)
#     elif d > 1:
#         theta = torch.rand(n, d - 2) * np.pi
#         angles = torch.cat((radius, theta, phi), dim=1)
#     else:
#         raise ValueError("d must be strictly greater than 1")
#
#     x = spherical_to_cartesian_torch(angles)
#     noise = torch.normal(torch.zeros_like(x), torch.ones_like(x) * std)
#
#     data = (x + noise).to(device)
#     data_np = data.detach().cpu().numpy()
#     fig = plt.figure(figsize=(15, 15))
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], marker="*")
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
#     plt.show()
#
#     return data, data_np

def generate_spherical_with_noise(n, d, r, std, device="cuda"):
    assert d > 1
    samples = torch.torch.randn(n,d)
    samples /= samples.norm(dim=1).unsqueeze(-1)
    # phi = torch.rand(n, 1) * 2 * np.pi
    # radius = torch.ones(n, 1) * r
    # if d == 1:
    #     angles = torch.cat((radius, phi), dim=1)
    # elif d > 1:
    #     theta = torch.rand(n, d - 2) * np.pi
    #     angles = torch.cat((radius, theta, phi), dim=1)
    # else:
    #     raise ValueError("d must be strictly greater than 1")

    noise = torch.normal(torch.zeros_like(samples), torch.ones_like(samples) * std)

    data = (samples + noise).to(device)
    data_np = data.detach().cpu().numpy()
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data_np[:, 0], data_np[:, 1], data_np[:, 2], marker="*")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

    return data, data_np

def generate_samples (model, sample_size=100, n_iter=500):
    samples, log_probs = [], []
    for _ in tqdm.tqdm(range(n_iter)):
        posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)
        samples.append(posterior_samples.cpu().detach().numpy())
        log_probs.append(log_probs_samples.cpu().detach().numpy())

    samples = np.concatenate(samples, 0)
    log_probs = np.concatenate(log_probs, 0)

    return samples, log_probs

def evaluate_uniform_on_sphere(d, r, samples, loglik):
    log_const_1 = np.log(2) + 0.5 * d * np.log(np.pi)
    log_const_2 = (d - 1) * np.log(r)
    log_const_3 = - sp.special.loggamma(0.5 * d)

    gt = - (log_const_1 + log_const_2 + log_const_3)

    cart_samples = torch.from_numpy(samples)
    sph_samples = cartesian_to_spherical_torch(cart_samples)
    logabsdet = logabsdet_sph_to_car(sph_samples).detach().cpu().numpy()
    print(gt)
    print(loglik)
    print(logabsdet)
    return np.sqrt(np.mean(np.square(gt - loglik)))


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

