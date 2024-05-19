import numpy as np
import torch
import warnings

from normflows.flows.neural_spline.wrapper import CircularAutoregressiveRationalQuadraticSpline, CircularCoupledRationalQuadraticSpline
from normflows.flows.neural_spline.wrapper import AutoregressiveRationalQuadraticSpline, CoupledRationalQuadraticSpline
from normflows.distributions.base import UniformGaussian

from enflows.transforms import Sigmoid, Tanh
from enflows.distributions import StandardNormal, Uniform
from enflows.distributions.normal import MOG
from enflows.nn.nets.resnet import ConvResidualNet, ResidualNet
from enflows.distributions.uniform import UniformSphere, MultimodalUniform, UniformSimplex
from enflows.transforms import MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.permutations import RandomPermutation
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.nn.nets import Sin
from enflows.flows.base import Flow
from enflows.transforms.lipschitz import LipschitzDenseNetBuilder, iResBlock
from enflows.transforms.injective import (ConstrainedAnglesSigmoid, ClampedTheta, ClampedThetaPositive, LearnableManifoldFlow,
                                          SphereFlow, LpManifoldFlow, ScaleLastDim, ResidualNetInput, CondLpManifoldFlow, PositiveL1ManifoldFlow)
from enflows.transforms.linear import ScalarScale, ScalarShift
from enflows.transforms.svd import SVDLinear


def build_flow_forward(args, clamp_theta=False):
    transformation_layers = []

    if args.learn_manifold:
        manifold_mapping = LearnableManifoldFlow(n=args.datadim - 1, max_radius=2., logabs_jacobian=args.logabs_jacobian)
    else:
        if args.dataset == "lp_uniform":
            manifold_mapping = LpManifoldFlow(norm=1, p=args.beta, logabs_jacobian=args.logabs_jacobian)
        else:
            manifold_mapping = SphereFlow(n=args.datadim - 1, r=1., logabs_jacobian=args.logabs_jacobian)

    transformation_layers.append(manifold_mapping)

    if clamp_theta:
        transformation_layers.append(ClampedTheta(eps=1e-3))

    params = dict(flow_dim=args.datadim, n_layers=args.n_layers, hidden_features=args.n_hidden_features, device=args.device)
    if args.architecture == 'circular':
        base_dist, _transformation_layers = build_flow_circular_fwd(**params)
    elif args.architecture == 'unbounded':
        base_dist, _transformation_layers = build_flow_unbounded_fwd(**params)
    elif args.architecture == 'unbounded_circular':
        base_dist, _transformation_layers = build_flow_unbounded_circular_fwd(**params)
    elif args.architecture == 'ambient':
        base_dist, _transformation_layers = build_flow_ambient_fwd(**params)
        transform = CompositeTransform(_transformation_layers)
        flow = Flow(transform, base_dist).to(args.device)
        return flow
    else:
        raise ValueError(f'type {type} is not supported')

    transformation_layers += _transformation_layers
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow

RADIUS_CONST = 1.64

def build_flow_reverse(args, clamp_theta=False):
    params = dict(flow_dim=args.datadim, n_layers=args.n_layers, hidden_features=args.n_hidden_features, device=args.device)
    if args.architecture == 'circular':
        base_dist, transformation_layers = build_flow_circular_rvs(**params)
    elif args.architecture == 'unbounded':
        base_dist, transformation_layers = build_flow_unbounded_rvs(**params)
    elif args.architecture == 'unbounded_circular':
        base_dist, transformation_layers = build_flow_unbounded_circular_rvs(**params)
    elif args.architecture == 'ambient':
        base_dist, transformation_layers = build_flow_ambient_rvs(**params)
        # transformation_layers = transformation_layers[::-1]
        transform = CompositeTransform(transformation_layers)
        flow = Flow(transform, base_dist).to(args.device)
        return flow
    else:
        raise ValueError(f'type {type} is not supported')

    if clamp_theta:
        transformation_layers.append(ClampedTheta(eps=1e-3))

    if args.learn_manifold:
        manifold_mapping = LearnableManifoldFlow(n=args.datadim - 1, max_radius=2., logabs_jacobian=args.logabs_jacobian)
    else:
        if args.dataset == "lp_uniform":
            if isinstance(base_dist, UniformSphere):
                manifold_mapping = LpManifoldFlow(norm=1., p=args.beta, logabs_jacobian=args.logabs_jacobian)
                radius = np.array(manifold_mapping.r_given_theta(base_dist.sample(1000)))
                norm = RADIUS_CONST / radius.mean()
                manifold_mapping.norm = norm
            else:
                warnings.warn("The logabsdet of the Jacobian for the LpManifold can become unstable "
                              "it the base distribution is not UniformSphere()")
                manifold_mapping = LpManifoldFlow(norm=1., p=args.beta, logabs_jacobian=args.logabs_jacobian)
        else:
            # manifold_mapping = SphereFlow(n=args.datadim - 1, r=RADIUS_CONST, logabs_jacobian=args.logabs_jacobian)
            manifold_mapping = SphereFlow(n=args.datadim - 1, r=1, logabs_jacobian=args.logabs_jacobian)

    transformation_layers.append(manifold_mapping)

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # combine into a flow
    flow = Flow(transform, base_dist).to(args.device)

    return flow

def build_flow_circular_fwd(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    torch_one = torch.ones(1, device=device)
    base_dist = Uniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one)

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        ScaleLastDim(scale=0.5)
    )

    transformation_layers.append(
        CompositeTransform([
            ScalarScale(scale= 2./np.pi, trainable=False),
            ScalarShift(shift=-1., trainable=False),
        ])
    )

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        # transformation_layers.append(
        #     InverseTransform(
        #         CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                                       num_hidden_channels=hidden_features,
        #                                                       num_blocks=3, num_bins=10, tail_bound=1,
        #                                                       ind_circ=[i for i in range(flow_dim - 1)])
        #     )
        # )
        transformation_layers.append(
            InverseTransform(
                CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                      num_hidden_channels=hidden_features,
                                                      num_blocks=3, num_bins=10, tail_bound=1,
                                                      ind_circ=[i for i in range(flow_dim - 1)])
            )
        )

    return base_dist, transformation_layers

def build_flow_circular_rvs(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    torch_one = torch.ones(1, device=device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one)
    # base_dist = MultimodalUniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one, n_modes=2)
    base_dist = UniformSphere(shape=[flow_dim - 1])
    # n_modes = 20
    # base_dist = MOG(means=torch.rand((n_modes, flow_dim-1), device=device)*2-1,
    #                 stds=torch.ones((n_modes, flow_dim-1), device=device)*0.05, low=-1, high=1)

    # base_dist = StandardNormal(shape=[flow_dim-1])
    # base_dist = UniformGaussian(ndim=flow_dim - 1, ind=-1)

    # Define an invertible transformation
    transformation_layers = []
    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScaleLastDim(scale=0.5),
                ScalarScale(scale=2/np.pi, trainable=False),
                ScalarShift(shift=-1, trainable=False),
            ])
        )
    )

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        transformation_layers.append(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              num_blocks=3, num_bins=8, tail_bound=1,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
        )
        # transformation_layers.append(
        #     # InverseTransform(
        #         CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                                num_hidden_channels=hidden_features,
        #                                                num_blocks=3, num_bins=8, tail_bound=1,
        #                                                ind_circ=[i for i in range(flow_dim - 1)])
        #     # )
        #
        # )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScalarShift(shift=1., trainable=False),
                ScalarScale(scale=0.5 * np.pi, trainable=False),
                ScaleLastDim(scale=2)
            ])
        )
    )

    return base_dist, transformation_layers


def build_flow_ambient_fwd(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim, densenet_depth=3, activation_function=Sin(w0=10), lip_coeff=.97,)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))
        transformation_layers.append(ActNorm(features=flow_dim))

        # transformation_layers.append(SVDLinear(features= flow_dim - 1, num_householder=4))

        # transformation_layers.append(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features, num_blocks=3, n_sigmoids=30)
        # )

        transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))

    return base_dist, transformation_layers


def build_flow_unbounded_fwd(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    transformation_layers.append(ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True))

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim-1))
        transformation_layers.append(ActNorm(features=flow_dim-1))
        # transformation_layers.append(SVDLinear(features= flow_dim - 1, num_householder=4))

        transformation_layers.append(
          MaskedSumOfSigmoidsTransform(features=flow_dim-1, hidden_features=hidden_features, num_blocks=3, n_sigmoids=30)
        )

        # transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))

    return base_dist, transformation_layers

def build_flow_unbounded_rvs(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim - 1])
    # base_dist = UniformSphere(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim-1))
        # transformation_layers.append(InverseTransform(SVDLinear(features= flow_dim - 1, num_householder=4)))

        # transformation_layers.append(InverseTransform(iResBlock(densenet_builder.build_network(), brute_force=True)))
        # transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))
        transformation_layers.append(InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3,
                                             n_sigmoids=30))
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim - 1)
            )
        )


    transformation_layers.append(
            ConstrainedAnglesSigmoid(temperature=1, learn_temperature=True)
    )

    return base_dist, transformation_layers


def build_flow_unbounded_circular_fwd(flow_dim, n_layers=3, hidden_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim-1, densenet_depth=3, activation_function=Sin(w0=1), lip_coeff=.97,)

    transformation_layers.append(ScaleLastDim(scale=0.5))
    transformation_layers.append(
        CompositeTransform([ScalarScale(scale=2. / np.pi, trainable=False),
                            ScalarShift(shift=-1., trainable=False)])
    )

    for i in range(10):
        transformation_layers.append(
            InverseTransform(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              num_blocks=3, num_bins=10, tail_bound=1,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
            )
        )
        # transformation_layers.append(
        #     InverseTransform(
        #         CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                               num_hidden_channels=hidden_features,
        #                                               num_blocks=3, num_bins=10, tail_bound=1,
        #                                               ind_circ=[i for i in range(flow_dim - 1)])
        #     )
        # )

    transformation_layers.append(
        CompositeTransform([ScalarShift(shift=1, trainable=False),
                            ScalarScale(scale=0.5, trainable=False),
                            InverseTransform(Sigmoid())])
    )

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim-1))
        transformation_layers.append(ActNorm(features=flow_dim - 1))
        # transformation_layers.append(
        #         SVDLinear(features= flow_dim - 1, num_householder=4)
        # )

        # transformation_layers.append(
        #         MaskedSumOfSigmoidsTransform(features=flow_dim - 1, hidden_features=hidden_features, num_blocks=3, n_sigmoids=30)
        # )

        transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))


    return base_dist, transformation_layers

def build_cond_flow_reverse(args, clamp_theta=False):
    params = dict(flow_dim=args.datadim, n_layers=args.n_layers, hidden_features=args.n_hidden_features, context_features=args.n_context_features, device=args.device)
    if args.architecture == 'circular':
        base_dist, transformation_layers = build_cond_flow_circular_rvs(**params)
    elif args.architecture == 'unbounded':
        base_dist, transformation_layers = build_flow_unbounded_rvs(**params)
    elif args.architecture == 'unbounded_circular':
        base_dist, transformation_layers = build_flow_unbounded_circular_rvs(**params)
    elif args.architecture == 'ambient':
        base_dist, transformation_layers = build_cond_flow_ambient_rvs(**params)
        transformation_layers = transformation_layers[::-1]
        transform = CompositeTransform(transformation_layers)
        embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=256,
                                         num_blocks=3, activation=torch.nn.functional.relu)
        flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)
        return flow
    else:
        raise ValueError(f'type {type} is not supported')

    if clamp_theta:
        transformation_layers.append(ClampedTheta(eps=1e-3))

    manifold_mapping = CondLpManifoldFlow(norm=1., p=args.beta, logabs_jacobian=args.logabs_jacobian)

    transformation_layers.append(manifold_mapping)

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=256,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)

    return flow

def build_cond_flow_circular_rvs(flow_dim, n_layers=3, hidden_features=256, context_features=16, device='cuda'):
    # torch_one = torch.ones(1, device=device)
    # base_dist = Uniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one)
    # base_dist = MultimodalUniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one, n_modes=2)
    base_dist = UniformSphere(shape=[flow_dim - 1])
    # n_modes = 20
    # base_dist = MOG(means=torch.rand((n_modes, flow_dim-1), device=device)*2-1,
    #                 stds=torch.ones((n_modes, flow_dim-1), device=device)*0.05, low=-1, high=1)

    # base_dist = StandardNormal(shape=[flow_dim-1])
    # base_dist = UniformGaussian(ndim=flow_dim - 1, ind=-1)

    # Define an invertible transformation
    transformation_layers = []
    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScaleLastDim(scale=0.5),
                ScalarScale(scale=2/np.pi, trainable=False),
                ScalarShift(shift=-1, trainable=False),
            ])
        )
    )

    for _ in range(n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
        transformation_layers.append(
                CircularAutoregressiveRationalQuadraticSpline(num_input_channels=flow_dim - 1,
                                                              num_hidden_channels=hidden_features,
                                                              num_context_channels=context_features,
                                                              num_blocks=3, num_bins=8, tail_bound=1,
                                                              ind_circ=[i for i in range(flow_dim - 1)])
        )
        # transformation_layers.append(
        #     # InverseTransform(
        #         CircularCoupledRationalQuadraticSpline(num_input_channels=flow_dim - 1,
        #                                                num_hidden_channels=hidden_features,
        #                                                num_context_channels = context_features,
        #                                                num_blocks=3, num_bins=8, tail_bound=1,
        #                                                ind_circ=[i for i in range(flow_dim - 1)])
        #     # )
        #
        # )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([
                ScalarShift(shift=1., trainable=False),
                ScalarScale(scale=0.5 * np.pi, trainable=False),
                ScaleLastDim(scale=2)
            ])
        )
    )

    return base_dist, transformation_layers


def build_cond_flow_ambient_rvs(flow_dim, n_layers=3, hidden_features=256, context_features=256, device='cuda'):
    base_dist = StandardNormal(shape=[flow_dim])
    # base_dist = UniformSphere(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    # densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim, densenet_depth=3,
    #                                             context_features=context_features, activation_function=Sin(w0=1), lip_coeff=.97)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=flow_dim))
        # transformation_layers.append(InverseTransform(SVDLinear(features= flow_dim - 1, num_householder=4)))

        # transformation_layers.append(InverseTransform(iResBlock(densenet_builder.build_network(), brute_force=False)))
        # transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))
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


def build_cond_flow_l1_manifold(args, clamp_theta=False):
    base_dist = StandardNormal(shape=[args.datadim - 1])
    # base_dist = UniformSphere(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    # densenet_builder = LipschitzDenseNetBuilder(input_channels=args.datadim - 1, densenet_depth=3,
    #                                             context_features=args.n_context_features, activation_function=Sin(w0=1), lip_coeff=.97)

    for _ in range(args.n_layers):
        transformation_layers.append(RandomPermutation(features=args.datadim - 1))
        # transformation_layers.append(InverseTransform(SVDLinear(features= flow_dim - 1, num_householder=4)))
        # transformation_layers.append(InverseTransform(iResBlock(densenet_builder.build_network(), brute_force=True)))
        # transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))
        transformation_layers.append(InverseTransform(
            MaskedSumOfSigmoidsTransform(features=args.datadim - 1, hidden_features=args.n_hidden_features,
                                         context_features=args.n_context_features, num_blocks=3, n_sigmoids=30))
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=args.datadim - 1)
            )
        )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([Sigmoid(eps=1e-4),
                                ScalarScale(scale=0.5 * torch.pi, trainable=False, eps=0)
                                # ScalarShift(shift=1, trainable=False),
                                # ScalarScale(scale=np.pi, trainable=False),
                                # ScalarScale(scale=0.25, trainable=False)
                                # Sigmoid(temperature=1, learn_temperature=True),

            ])
        )
    )

    # transformation_layers.append(LpManifoldFlow(norm=1., p=1., logabs_jacobian=args.logabs_jacobian))
    transformation_layers.append(PositiveL1ManifoldFlow(logabs_jacobian=args.logabs_jacobian))

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=256,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)

    return flow


def build_simple_cond_flow_l1_manifold(args, n_layers, n_hidden_features, n_context_features,  clamp_theta=False):
    base_dist = StandardNormal(shape=[args.datadim - 1])
    # base_dist = UniformSphere(shape=[flow_dim - 1])
    # Define an invertible transformation
    transformation_layers = []

    # densenet_builder = LipschitzDenseNetBuilder(input_channels=args.datadim - 1, densenet_depth=3,
    #                                             context_features=n_context_features, activation_function=Sin(w0=1), lip_coeff=.97)

    for _ in range(n_layers):
        transformation_layers.append(RandomPermutation(features=args.datadim - 1))
        # transformation_layers.append(InverseTransform(SVDLinear(features= flow_dim - 1, num_householder=4)))
        # transformation_layers.append(InverseTransform(iResBlock(densenet_builder.build_network(), brute_force=True)))
        # transformation_layers.append(iResBlock(densenet_builder.build_network(), brute_force=True))
        transformation_layers.append(InverseTransform(
            MaskedSumOfSigmoidsTransform(features=args.datadim - 1, hidden_features=n_hidden_features,
                                         context_features=n_context_features, num_blocks=3, n_sigmoids=30))
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=args.datadim - 1)
            )
        )

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([Sigmoid(eps=1e-8),
                                ScalarScale(scale=0.5 * torch.pi - 1e-6, trainable=False, eps=1e-8)
                                # ScalarShift(shift=1, trainable=False),
                                # ScalarScale(scale=np.pi, trainable=False),
                                # ScalarScale(scale=0.25, trainable=False)
                                # Sigmoid(temperature=1, learn_temperature=True),
            ])
        )
    )

    #transformation_layers.append(PositiveL1ManifoldFlow(logabs_jacobian=args.logabs_jacobian))
    transformation_layers.append(LpManifoldFlow(logabs_jacobian=args.logabs_jacobian, p=1, norm=args.norm))

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=n_context_features, hidden_features=64,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)

    return flow

def build_circular_cond_flow_l1_manifold(args):
    # torch_one = torch.ones(1, device=args.device)
    # base_dist = Uniform(shape=[args.datadim - 1], low=torch_one * 0, high=torch_one * 0.5 * torch.pi)
    # base_dist = MultimodalUniform(shape=[flow_dim - 1], low=-torch_one, high=torch_one, n_modes=2)
    # base_dist = UniformSphere(shape=[args.datadim - 1], all_positive=True)
    base_dist = UniformSimplex(shape=[args.datadim - 1], extend_star_like=False)
    # n_modes = 20
    # base_dist = MOG(means=torch.rand((n_modes, flow_dim-1), device=device)*2-1,
    #                 stds=torch.ones((n_modes, flow_dim-1), device=device)*0.05, low=-1, high=1)

    # base_dist = StandardNormal(shape=[flow_dim-1])
    # base_dist = UniformGaussian(ndim=flow_dim - 1, ind=-1)

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        InverseTransform(
            CompositeTransform([ScalarScale(scale=4 / torch.pi, trainable=False, eps=0),
                                ScalarShift(shift=-1, trainable=False)
                                ])
        )
    )

    for _ in range(args.n_layers):
        # transformation_layers.append(RandomPermutation(features=flow_dim - 1))
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
                                ScalarScale(scale=0.25 * torch.pi, trainable=False, eps=0)
                                ])
        )
    )

    #transformation_layers.append(InverseTransform(ClampedThetaPositive(eps=1e-10)))

    transformation_layers.append(PositiveL1ManifoldFlow(logabs_jacobian=args.logabs_jacobian))

    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNetInput(in_features=1, out_features=args.n_context_features, hidden_features=64,
                                     num_blocks=3, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(args.device)

    return flow