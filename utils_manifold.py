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
from enflows.transforms.injective import ConstrainedAnglesSigmoid, ClampedAngles, LearnableManifoldFlow


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
            mse_loss = 5 * torch.norm(data_manifold - data, dim=1).mean()
            # kl_div = torch.mean(q_log_prob)
            total_loss = log_likelihood + mse_loss
            total_loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), .001)
            optimizer.step()

            # loss.append(torch.mean(q_log_prob_beta - log_lik).cpu().detach().numpy())
            # loss.append(torch.mean(q_log_prob).cpu().detach().numpy())
            log_lik_ = log_likelihood.cpu().detach().numpy()
            mse_ = mse_loss.cpu().detach().numpy()
            loss_ = total_loss.cpu().detach().numpy()
            loss.append([loss_, log_lik_, mse_])

            print(f"Training loss at step {epoch}: {loss_:.4f}")
            print(f"negative log-likelihood: {log_lik_:.4f} \t MSE: {mse_:.4f}")
            if epoch % 100 == 0:
                samples, log_probs = generate_samples(model)
                mse_log_probs = evaluate_uniform_on_sphere(data.shape[-1], r, log_probs)
                print(f"MSE: {mse_log_probs:.2f}")

                samples_cart, _ = model.sample_and_log_prob(num_samples=1000)
                samples_cart_np = samples_cart.detach().cpu().numpy()
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(projection='3d')
                ax.scatter(samples_cart_np[:, 0], samples_cart_np[:, 1], samples_cart_np[:, 2], marker=".")
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')
                plt.show()


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

def generate_spherical_with_noise(n, d, r, std, device="cuda"):
    phi = torch.rand(n, 1) * 2 * np.pi
    radius = torch.ones(n, 1) * r
    if d == 1:
        angles = torch.cat((radius, phi), dim=1)
    elif d > 1:
        theta = torch.rand(n, d - 2) * np.pi
        angles = torch.cat((radius, theta, phi), dim=1)
    else:
        raise ValueError("d must be strictly greater than 1")

    x = spherical_to_cartesian_torch(angles)
    noise = torch.normal(torch.zeros_like(x), torch.ones_like(x) * std)

    data = (x + noise).to(device)
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

def evaluate_uniform_on_sphere(d, r, samples_loglik):
    log_const_1 = np.log(2) + 0.5 * d * np.log(np.pi)
    log_const_2 = (d - 1) * np.log(r)
    log_const_3 = - sp.special.loggamma(0.5 * d)

    gt = - (log_const_1 + log_const_2 + log_const_3)

    return np.sqrt(np.mean(np.square(gt-samples_loglik)))
