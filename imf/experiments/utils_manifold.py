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
from enflows.transforms.injective import ConstrainedAnglesSigmoid, ClampedAngles, ClampedTheta, LearnableManifoldFlow, SphereFlow, LpManifoldFlow, ScaleLastDim
from enflows.transforms.linear import ScalarScale, ScalarShift
from enflows.transforms.orthogonal import HouseholderSequence
from enflows.transforms.svd import SVDLinear


import logging
logger = logging.getLogger(__name__)

def define_model_name(args, dataset):
    args.model_name = (f"./models/imf_{args.dataset}_{args.architecture}_lm{args.learn_manifold}_{args.logabs_jacobian}"
                       f"{dataset.dataset_suffix}_epochs{args.epochs}_seed{args.seed}")

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




def build_flow_reverse(args, clamp_theta=False):
    params = dict(flow_dim=args.datadim, n_layers=args.n_layers, hidden_features=args.n_hidden_features, device=args.device)
    if args.architecture == 'circular':
        base_dist, transformation_layers = build_flow_manifold_circular_(**params)
    elif args.architecture == 'unbounded':
        base_dist, transformation_layers = build_flow_manifold_unbounded(**params)
    elif args.architecture == 'unbounded_circular':
        base_dist, transformation_layers = build_flow_manifold_unbounded_circular(**params)
    elif args.architecture == 'ambient':
        base_dist, transformation_layers = build_flow_manifold_ambient(**params)
        transformation_layers = transformation_layers[::-1]
        transform = CompositeTransform(transformation_layers)
        flow = Flow(transform, base_dist).to(args.device)
        return flow
    else:
        raise ValueError(f'type {type} is not supported')

    if clamp_theta:
        transformation_layers.append(
            InverseTransform(
                ClampedTheta(eps=1e-3)
            )
        )

    if args.learn_manifold:
        manifold_mapping = LearnableManifoldFlow(n=args.datadim - 1, max_radius=2.,
                                                 logabs_jacobian=args.logabs_jacobian, num_hutchinson_samples=4)
    else:
        manifold_mapping = LpManifoldFlow(norm=1., p=1.)
        manifold_mapping = SphereFlow(n=args.datadim - 1, r=1.,
                                      logabs_jacobian=args.logabs_jacobian, num_hutchinson_samples=4)

    transformation_layers.append(
        InverseTransform(
            manifold_mapping
        )
    )

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow

def build_flow_manifold_unbounded_circular(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim - 1])
    torch_one = torch.ones(1).to(device)
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim-1))

        # transformation_layers.append(
        #     InverseTransform(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
        #                                      n_sigmoids=30)
        #     )
        # )

        transformation_layers.append(
            InverseTransform(
                iResBlock(densenet_builder.build_network(), brute_force=True)
            )
        )

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
                   ScalarScale(scale=2, trainable=False),
                   ScalarShift(shift=-1, trainable=False)])
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
    for i in range(10):
        transformation_layers.append(
            InverseTransform(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              num_blocks=1, num_bins=10, tail_bound=1,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
            )
        )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScalarShift(shift=1., trainable=False),
                ScalarScale(scale=0.5*np.pi, trainable=False)])
        )
    )

    transformation_layers.append(
        InverseTransform(
            ScaleLastDim()
        )
    )

    return base_dist, transformation_layers

def build_flow_manifold_unbounded(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim-1))

        # transformation_layers.append(
        #     InverseTransform(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
        #                                      n_sigmoids=30)
        #     )
        # )

        transformation_layers.append(
            InverseTransform(
                iResBlock(densenet_builder.build_network(), brute_force=True)
            )
        )

        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim - 1)
            )
        )

    transformation_layers.append(
        InverseTransform(
            ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True)
        )
    )

    return base_dist, transformation_layers

def build_flow_manifold_ambient(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))

        # transformation_layers.append(
        #     InverseTransform(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        #     )
        # )

        # transformation_layers.append(
        #     InverseTransform(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
        #                                      n_sigmoids=30)
        #     )
        # )

        transformation_layers.append(
            InverseTransform(
                iResBlock(densenet_builder.build_network(), brute_force=True)
            )
        )

        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )

    return base_dist, transformation_layers


def build_flow_manifold_ambient_(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))

        # transformation_layers.append(
        #     InverseTransform(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        #     )
        # )

        transformation_layers.append(
            # InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features, num_blocks=3,
                                             n_sigmoids=30)
            # )
        )

        # transformation_layers.append(
        #     InverseTransform(
        #         iResBlock(densenet_builder.build_network(), brute_force=True)
        #     )
        #
        # )

        transformation_layers.append(
            ActNorm(features=flow_dim)
        )

    return base_dist, transformation_layers


def build_flow_manifold_circular(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    # torch_pi = torch.tensor(np.pi).to(device)
    torch_one = torch.ones(1).to(device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_pi, high=torch_pi)
    base_dist = Uniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one)

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        ScaleLastDim(scale=0.5)
    )

    transformation_layers.append(
        CompositeTransform([
            # ScalarShift(shift=np.pi, trainable=False),
            ScalarScale(scale= 2./np.pi, trainable=False),
            ScalarShift(shift=-1., trainable=False),
            # ScalarScale(scale=0.5, trainable=False)])
        ])
    )

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        transformation_layers.append(
            InverseTransform(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              # num_blocks=2, num_bins=30, tail_bound=np.pi,
                                                              num_blocks=3, num_bins=10, tail_bound=1,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
            )
        )
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        transformation_layers.append(
            InverseTransform(
                CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                      num_hidden_channels=hidden_features,
                                                      num_blocks=2, num_bins=5, tail_bound=1,
                                                      ind_circ=[i for i in range(flow_dim - 1)])
            )
        )

    return base_dist, transformation_layers

def build_flow_manifold_circular_(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    # torch_pi = torch.tensor(np.pi).to(device)
    torch_one = torch.ones(1).to(device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_pi, high=torch_pi)
    base_dist = Uniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one)

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        # transformation_layers.append(
        #     # InverseTransform(
        #         CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                                       num_hidden_channels=hidden_features,
        #                                                       # num_blocks=2, num_bins=30, tail_bound=np.pi,
        #                                                       num_blocks=3, num_bins=10, tail_bound=1,
        #                                                       ind_circ=[i for i in range(flow_dim - 1)])
        #     # )
        #
        # )
        transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        transformation_layers.append(
            # InverseTransform(
                CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                      num_hidden_channels=hidden_features,
                                                      num_blocks=3, num_bins=10, tail_bound=np.pi,
                                                      ind_circ=[i for i in range(flow_dim - 1)])
            # )
        )



    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScalarShift(shift=1., trainable=False),
                ScalarScale(scale=0.5 * np.pi, trainable=False),
            ])
        )

    )

    transformation_layers.append(
        InverseTransform(
            ScaleLastDim(scale=2)
        )
    )

    return base_dist, transformation_layers


def train_model_forward(model, data, args, batch_size=800, context=None, alternating=False, early_stopping=False, device="cuda", **kwargs):
    # optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':log_sigma, 'lr':1e-2}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    if early_stopping:
        idx = data.shape[0]//5
        val_data = data[:idx].requires_grad_(True)
        dataloader = torch.utils.data.DataLoader(data[idx:], batch_size=batch_size, shuffle=True)
    else:
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    loss = []
    try:
        start_time = time.monotonic()
        model.train()
        for epoch in range(args.epochs):
            for i, batch_data in enumerate(dataloader):
                optimizer.zero_grad()
                # add small noise to the dataset to prevent overfitting
                # batch_data = batch_data + torch.randn(batch_data.shape, device=device) * 0.05
                batch_data.requires_grad_(True)
                # project data on the manifold
                thetas, _ = model._transform._transforms[0].forward(batch_data, context=context)
                # thetas, _ = model._transform._transforms[0].forward(batch_data, context=context)
                data_manifold, _ = model._transform._transforms[0].inverse(thetas, context=context)
                # data_manifold, _ = model._transform._transforms[0].inverse(thetas, context=context)
                # breakpoint()
                # compute log prob
                # data_base_ = model.transform_to_noise(batch_data, context)
                # log_prob_ = model._distribution.log_prob(data_base_, context)
                # data_manifold_, logabsdet_ = model._transform.inverse(data_base_, context)
                # log_prob_ = log_prob_ - logabsdet_

                log_prob = model.log_prob(data_manifold, context=context)

                # q_samples, q_log_prob = model.sample_and_log_prob(num_samples=sample_size)
                if torch.any(torch.isnan(log_prob)): breakpoint()
                if torch.any(torch.isnan(data_manifold)): breakpoint()

                # breakpoint()

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
                    # total_loss = beta * mse_loss
                    total_loss.backward()
                    optimizer.step()
                    if early_stopping:
                        val_thetas, _ = model._transform._transforms[0].forward(val_data, context=context)
                        val_data_manifold, _ = model._transform._transforms[0].inverse(val_thetas, context=context)
                        val_log_prob = model.log_prob(val_data_manifold, context=context)

                        val_mse_loss = torch.norm(val_data_manifold - val_data, dim=1).mean()
                        val_log_likelihood = -torch.mean(val_log_prob)
                        val_total_loss = val_log_likelihood + beta * val_mse_loss

                        log_lik_v = val_log_likelihood.cpu().detach().numpy()
                        mse_v = val_mse_loss.cpu().detach().numpy()
                        loss_v = val_total_loss.cpu().detach().numpy()

                log_lik_ = log_likelihood.cpu().detach().numpy()
                mse_ = mse_loss.cpu().detach().numpy()
                loss_ = total_loss.cpu().detach().numpy()
                if early_stopping:
                    loss.append([loss_, log_lik_, mse_, loss_v, log_lik_v, mse_v])
                    print(f"Train loss {epoch}: {loss_:.5f} (NLL: {log_lik_:.4f}, MSE: {mse_:.5f})")
                    print(f"Val loss {epoch}: {loss_v:.5f} (NLL: {log_lik_v:.4f}, MSE: {mse_v:.5f})")
                else:
                    loss.append([loss_, log_lik_, mse_])
                    print(f"Train loss {epoch}: {loss_:.5f} (NLL: {log_lik_:.4f}, MSE: {mse_:.5f})")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    torch.save(model.state_dict(), args.model_name)
    f = open(args.model_name+".txt", "w")
    f.write(str(time_diff))
    f.close()

    return model, np.array(loss)

def train_model_reverse(model,args, batch_size=1000, context=None, **kwargs):
    # optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':log_sigma, 'lr':1e-2}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    loss = []
    try:
        start_time = time.monotonic()
        model.train()
        for epoch in range(args.epochs):
            # T = cooling_function(epoch // (args.epochs / num_iter))

            optimizer.zero_grad()
            samples, logprob_flow = model.sample_and_log_prob(num_samples=batch_size, context=None)
            # logprob_target = -torch.ones_like(logprob_flow)*np.log(np.sqrt(3)*4) # uniform on lp manifold

            kl_div = torch.mean(logprob_flow)# - logprob_target)
            kl_div.backward()
            optimizer.step()

            kl_div_ = kl_div.cpu().detach().numpy()
            loss.append(kl_div_)
            # print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")
            print(f"Training loss at step {epoch}: {loss[-1]:.3f}")
            print(f"logprob_flow: {logprob_flow.mean().cpu().detach().numpy():.3f} ")
                  # f"logprob_target: {logprob_target.mean().cpu().detach().numpy():.3f}")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    torch.save(model.state_dict(), args.model_name)
    f = open(args.model_name+".txt", "w")
    f.write(str(time_diff))
    f.close()

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

def evaluate_flow(points, flow, dataset, batch_size=1000, device='cuda'):
    points_torch = torch.from_numpy(points).float().to(device)

    logp_flow, logp_gt, radius = [], [], []
    n_iter = points_torch.shape[0] // batch_size
    if n_iter * batch_size < points_torch.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * batch_size, (i + 1) * batch_size
        points_torch = torch.from_numpy(points[left:right]).float().to(device)
        logp_gt_ = dataset.log_density(points[left:right])
        logp_gt += list(logp_gt_.detach().cpu().numpy())

        angles = flow.transform_to_noise(points_torch, context=None)
        logp_flow_ = flow._distribution.log_prob(angles, context=None)
        uniform_surface_flow, logabsdet = flow._transform.inverse(angles, context=None)
        logp_flow_ = logp_flow_ - logabsdet
        logp_flow += list(logp_flow_.detach().cpu().numpy())

        learnt_r = cartesian_to_spherical_torch(uniform_surface_flow)[:,0]
        radius = list(learnt_r.detach().cpu().numpy())

    logp_flow = np.array(logp_flow)
    logp_gt = np.array(logp_gt)
    radius = np.array(radius)

    MSE_logp = np.sqrt(np.mean(np.square(logp_flow-logp_gt)))
    MSE_dist = np.sqrt(np.mean(np.square(radius-1.)))

    return MSE_logp, MSE_dist

def evaluate_samples(dataset, test_data, flow, rnf, args):
    # ground truth
    logp_gt = dataset.log_density(test_data).detach().cpu().numpy()

    # proposed flow
    angles = flow.transform_to_noise(test_data, context=None)
    logp_flow = flow._distribution.log_prob(angles, context=None)
    proj_data_flow, logabsdet = flow._transform.inverse(angles, context=None)
    logp_flow = (logp_flow - logabsdet).detach().cpu().numpy()

    # rnf
    logp_rnf = rnf_forward_logp(rnf, test_data, args)

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
    return np.array(logp)


def rnf_forward_points (rnf, data, args):
    data_proj = []
    n_iter = data.shape[0] // args.batchsize
    if n_iter * args.batchsize < data.shape[0]: n_iter += 1
    for i in tqdm.tqdm(range(n_iter)):
        left, right = i * args.batchsize, (i + 1) * args.batchsize
        data_proj_, logp_, _ = rnf.forward(data[left:right].detach().cpu())
        data_proj += list(data_proj_.detach().cpu().numpy())
    return np.array(data_proj)

