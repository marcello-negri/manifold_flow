import numpy as np
import torch
import time
import os
import tqdm
from datetime import timedelta
from torch.utils.data import Dataset

from normflows.flows.neural_spline.wrapper import CircularAutoregressiveRationalQuadraticSpline, CircularCoupledRationalQuadraticSpline

from enflows.transforms import Sigmoid
from enflows.distributions import StandardNormal, Uniform
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.permutations import RandomPermutation
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.nn.nets import Sin
from enflows.flows.base import Flow
from enflows.transforms.lipschitz import LipschitzDenseNetBuilder, iResBlock
from enflows.transforms.injective import ConstrainedAnglesSigmoid, ClampedAngles, ClampedTheta, LearnableManifoldFlow, SphereFlow, PeriodicElementwiseTransform, ScaleLastDim
from enflows.transforms.linear import ScalarScale, ScalarShift
from enflows.transforms.orthogonal import HouseholderSequence
from enflows.transforms.svd import SVDLinear


import logging
logger = logging.getLogger(__name__)


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
    # torch_pi = torch.tensor(np.pi).to(device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_pi, high=torch_pi)
    # base_dist = normalizing_flow.Uniform(shape=[flow_dim - 1], low=torch.tensor(0.), high=torch.tensor(np.pi))

    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=1, activation_function=Sin(w0=0.1), lip_coeff=.97,)

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim-1))

        # transformation_layers.append(
        #     InverseTransform(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
        #                                      n_sigmoids=30, dropout_probability=0.2)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         HouseholderSequence(features=flow_dim - 1, num_transforms=1)
        #     )
        # )

        transformation_layers.append(
            InverseTransform(
                iResBlock(densenet_builder.build_network(), brute_force=True)
            )
        )
        #
        # transformation_layers.append(
        #     InverseTransform(
        #         ActNorm(features=flow_dim - 1)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=flow_dim - 1, hidden_features=hidden_features,
        #                                                                 num_blocks=3, num_bins=20, tail_bound=np.pi, tails="circular")
        #     )
        # )
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        # transformation_layers.append(
        #     InverseTransform(
        #         CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1, num_hidden_channels=hidden_features,
        #                                                       num_blocks=2, num_bins=8, tail_bound=np.pi,
        #                                                       ind_circ=[i for i in range(flow_dim - 1)])
        #     )
        # )
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        # transformation_layers.append(
        #     InverseTransform(
        #         CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                               num_hidden_channels=hidden_features,
        #                                               num_blocks=2, num_bins=5, tail_bound=np.pi,
        #                                               ind_circ=[i for i in range(flow_dim - 1)])
        #     )
        # )

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
    #                ScalarShift(shift=np.pi, trainable=False),
    #                ScalarScale(scale=0.5, trainable=False)])
    #        )
    # )

    # transformation_layers.append(
    #     InverseTransform(
    #         ScaleLastDim()
    #     )
    # )

    # transformation_layers.append(
    #    InverseTransform(
    #            CompositeTransform([
    #                ScalarShift(shift=np.pi, trainable=False),
    #                ScalarScale(scale=0.5, trainable=False)])
    #        )
    # )

    # transformation_layers.append(
    #     InverseTransform(
    #         ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True)
    #     )
    # )

    transformation_layers.append(
       InverseTransform(
               CompositeTransform([
                   Sigmoid(),
                   ScalarScale(scale=2*np.pi, trainable=False),
                   ScalarShift(shift=-np.pi, trainable=False)])
           )
    )

    # transformation_layers.append(
    #    InverseTransform(
    #            CompositeTransform([
    #                ScalarScale(scale=2, trainable=False),
    #                ScalarShift(shift=-np.pi, trainable=False)])
    #        )
    # )
    #
    for i in range(1):
        transformation_layers.append(
            InverseTransform(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              num_blocks=1, num_bins=10, tail_bound=np.pi,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
            )
        )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScalarShift(shift=np.pi, trainable=False),
                ScalarScale(scale=0.5, trainable=False)])
        )
    )

    transformation_layers.append(
        InverseTransform(
            ScaleLastDim()
        )
    )

    # transformation_layers.append(
    #     InverseTransform(
    #         ClampedTheta(eps=1e-3)
    #     )
    # )

    transformation_layers.append(
        InverseTransform(
            # LearnableManifoldFlow(n=flow_dim - 1, max_radius=5, bruteforce=False)
            SphereFlow(n=flow_dim - 1, r=1., bruteforce=False)
        )
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(device)

    return flow


def train_model(model, data, batch_size=1_000, epochs=2_001, lr=1e-3, r=1., context=None, alternating=False, dataset="spherical_uniform", epsilon=0.1, device="cuda", **kwargs):
    # optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':log_sigma, 'lr':1e-2}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    loss = []
    try:
        start_time = time.monotonic()
        model.train()
        for epoch in range(epochs):
            for i, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                data_base = model.transform_to_noise(batch_data, context)
                log_prob = model._distribution.log_prob(data_base, context)
                data_manifold, logabsdet = model._transform.inverse(data_base, context)
                log_prob = log_prob - logabsdet
                # log_prob = model.log_prob(data, context=None)

                # q_samples, q_log_prob = model.sample_and_log_prob(num_samples=sample_size)
                if torch.any(torch.isnan(data_base)): breakpoint()
                if torch.any(torch.isnan(data_manifold)): breakpoint()

                beta = 100
                if alternating:
                    mse_loss = beta * torch.norm(data_manifold - batch_data, dim=1).mean()
                    mse_loss.backward(retain_graph=True)
                    optimizer.step()
                    log_likelihood = -torch.mean(log_prob)
                    log_likelihood.backward()
                    optimizer.step()
                    total_loss = log_likelihood + mse_loss
                else:
                    mse_loss = torch.norm(data_manifold - batch_data, dim=1).mean()
                    log_likelihood = -torch.mean(log_prob)
                    total_loss = log_likelihood + beta * mse_loss
                    total_loss.backward()
                    optimizer.step()
                # kl_div = torch.mean(q_log_prob)
                # total_loss = log_likelihood + 100 * mse_loss
                # total_loss.backward()
                # log_likelihood.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), .001)


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

    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"{model_dir}/manifold_flow_{dataset}_{epsilon:.2f}_e{epochs}")
    torch.save(model.state_dict(), f"{model_dir}/manifold_flow_{dataset}_{epsilon:.2f}_e{epochs}")

    return model, np.array(loss)


def generate_samples (model, sample_size=100, n_iter=1000):
    samples, log_probs = [], []
    for _ in tqdm.tqdm(range(n_iter)):
        posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)
        samples.append(posterior_samples.cpu().detach().numpy())
        log_probs.append(log_probs_samples.cpu().detach().numpy())

    samples = np.concatenate(samples, 0)
    log_probs = np.concatenate(log_probs, 0)

    return samples, log_probs

def evaluate_flow_rnf(test_data, simulator, flow, rnf, device='cuda'):
    test_data_torch = torch.from_numpy(test_data).float().to(device)

    # ground truth
    logp_gt = simulator.log_density(test_data)

    # proposed flow
    angles = flow.transform_to_noise(test_data_torch, context=None)
    logp_flow = flow._distribution.log_prob(angles, context=None)
    proj_data_flow, logabsdet = flow._transform.inverse(angles, context=None)
    logp_flow = (logp_flow - logabsdet).detach().cpu().numpy()

    # rnf
    proj_data_rnf, logp_rnf, _ = rnf.forward(test_data_torch.detach().cpu())
    logp_rnf = logp_rnf.detach().cpu().numpy()

    MSE_flow = np.mean(np.square(logp_flow - logp_gt))
    MSE_rnf = np.mean(np.square(logp_rnf - logp_gt))

    dist_flow = simulator.distance_from_manifold(proj_data_flow.detach().cpu().numpy())
    dist_rnf = simulator.distance_from_manifold(proj_data_rnf.detach().cpu().numpy())

    MSE_dist_flow = np.mean(np.square(dist_flow))
    MSE_dist_rnf = np.mean(np.square(dist_rnf))

    return MSE_flow, MSE_rnf, MSE_dist_flow, MSE_dist_rnf


def evaluate_samples(test_data, flow, rnf, args, device='cuda'):
    if args.dataset == 'vonmises_fisher':
        logp_gt = von_mises_fisher.evaluate_mises_fisher(test_data, args)
    elif args.dataset == 'sphere_checkerboard':
        logp_gt = dist_sphere.evaluate_sphere_checkerboard(test_data)

    test_data_torch = torch.from_numpy(test_data).float().to(device)

    # proposed flow
    angles = flow.transform_to_noise(test_data_torch, context=None)
    logp_flow = flow._distribution.log_prob(angles, context=None)
    proj_data_flow, logabsdet = flow._transform.inverse(angles, context=None)
    logp_flow = (logp_flow - logabsdet).detach().cpu().numpy()

    # rnf
    logp_rnf = rnf_forward_logp(rnf, test_data_torch, args)

    MSE_flow = np.mean(np.square(logp_flow - logp_gt))
    MSE_rnf = np.mean(np.square(logp_rnf - logp_gt))


    return MSE_flow, MSE_rnf


def rnf_forward_logp (rnf, data, args):
    logp = []
    n_iter = data.shape[0] // args.batchsize
    if n_iter * args.batchsize < data.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * args.batchsize, (i + 1) * args.batchsize
        data_proj_, logp_, _ = rnf.forward(data[left:right].detach().cpu())
        logp += list(logp_.detach().cpu().numpy())
    return logp

def rnf_forward_points (rnf, data, args):
    data_proj = []
    n_iter = data.shape[0] // args.batchsize
    if n_iter * args.batchsize < data.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * args.batchsize, (i + 1) * args.batchsize
        data_proj_, logp_, _ = rnf.forward(data[left:right].detach().cpu())
        data_proj += list(data_proj_.detach().cpu().numpy())
    return np.array(data_proj)

