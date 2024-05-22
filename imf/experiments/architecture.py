import torch
from enflows.distributions import StandardNormal
from enflows.distributions.uniform import UniformSimplex
from enflows.flows.base import Flow
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.injective import (ScaleLastDim, ResidualNetInput, CondLpManifoldFlow, PositiveL1ManifoldFlow)
from enflows.transforms.linear import ScalarScale, ScalarShift
from enflows.transforms.normalization import ActNorm
from enflows.transforms.permutations import RandomPermutation
from normflows.flows.neural_spline.wrapper import CircularAutoregressiveRationalQuadraticSpline


def build_cond_flow_reverse(args, clamp_theta=False):
    params = dict(flow_dim=args.datadim, n_layers=args.n_layers, hidden_features=args.n_hidden_features, context_features=args.n_context_features, device=args.device)
    if args.architecture == 'ambient':
        base_dist, transformation_layers = build_cond_flow_ambient_rvs(**params)
        transformation_layers = transformation_layers[::-1]
        transform = CompositeTransform(transformation_layers)
        embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=256,
                                         num_blocks=3, activation=torch.nn.functional.relu)
        flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)
        return flow
    else:
        raise ValueError(f'type {type} is not supported')


def build_cond_flow_ambient_rvs(flow_dim, n_layers=3, hidden_features=256, context_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim])
    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))
        transformation_layers.append(InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features,
                                             context_features=context_features, num_blocks=3, n_sigmoids=30))
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )

    return base_dist, transformation_layers


def build_circular_cond_flow_l1_manifold(args, star_like=False):
    base_dist = UniformSimplex(shape=[args.datadim - 1], extend_star_like=star_like)

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                                ScaleLastDim(scale=0.5 if star_like else 1.0),
                                ScalarScale(scale=4 / torch.pi if not star_like else 2 / torch.pi, trainable=False, eps=0),
                                ScalarShift(shift=-1, trainable=False)
                                ])
        )
    )

    for _ in range(args.n_layers):
        transformation_layers.append(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=args.datadim - 1,
                                                      num_hidden_channels=args.n_hidden_features,
                                                      num_context_channels=args.n_context_features,
                                                      num_blocks=3, num_bins=8, tail_bound=1,
                                                      ind_circ=[i for i in range(args.datadim - 1)]
                                                      )
        )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([ScalarScale(scale=1 - 1e-4, trainable=False, eps=0),
                                ScalarShift(shift=1., trainable=False),
                                ScalarScale(scale=0.25 * torch.pi if not star_like else 0.5 * torch.pi, trainable=False, eps=0),
                                ScaleLastDim(scale=2.0 if star_like else 1.0),
                                ])
        )
    )

    if not star_like:
        transformation_layers.append(PositiveL1ManifoldFlow(logabs_jacobian=args.logabs_jacobian))
    else:
        transformation_layers.append(CondLpManifoldFlow(logabs_jacobian=args.logabs_jacobian, p=1, norm=args.norm))

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=64,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)

    return flow