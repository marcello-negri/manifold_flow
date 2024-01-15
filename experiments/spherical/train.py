import numpy as np
import torch
import os
from utils_manifold import (build_flow_manifold, train_model, evaluate_flow_rnf, plot_loss, generate_samples,
                            evaluate_uniform_on_sphere, plot_samples, plot_pairwise_angle_distribution)
import argparse
from plot import test_chatgpt, plot_icosphere

from rnf.experiments.datasets import load_simulator
from rnf.experiments.utils import create_filename
from rnf.experiments.architectures import create_model
from rnf.experiments.evaluate import sample_from_model
from rnf.experiments.architectures.create_model import ALGORITHMS
from rnf.experiments.train import train_model as train_model_rnf

import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dataset", type=str, default="spherical_gaussian", choices=["spherical_gaussian", "spherical_uniform"])
parser.add_argument("--train_flow", action="store_true", help="train proposed manifold flow")
parser.add_argument("--train_rnf", action="store_true", help="train rectangular normalizing flow")
parser.add_argument('--epochs_flow', metavar='e', type=int, default=1_000,
                    help='number of epochs')
parser.add_argument('--seed', metavar='s', type=int, default=1234,
                    help='random seed')
parser.add_argument('--datadim', metavar='d', type=int, default=3,
                    help='number of dimensions')
parser.add_argument('--n', metavar='n', type=int, default=1000,
                    help='number of training samples')
parser.add_argument('--r', metavar='r', type=float, default=1.,
                    help='radius of d-dimensional sphere')
parser.add_argument('--epsilon', metavar='epsilon', type=float, default=0.1,
                    help='std of the isotropic noise in the data')
parser.add_argument("--overwrite", action="store_true", help="Retrain and overwrite existing saved trained flow model")



# Model details
parser.add_argument("--modelname", type=str, default=None, help="Model name. Algorithm, latent dimension, dataset, and run are prefixed automatically.")
parser.add_argument("--algorithm", type=str, default="flow", choices=ALGORITHMS, help="Model: flow (AF), mf (FOM, M-flow), emf (Me-flow), pie (PIE), gamf (M-flow-OT), pae (PAE)...",)
parser.add_argument("-i", type=int, default=0, help="Run number")
parser.add_argument("--modellatentdim", type=int, default=2, help="Model manifold dimensionality")
parser.add_argument("--specified", action="store_true", help="Prescribe manifold chart: FOM instead of M-flow")
parser.add_argument("--outertransform", type=str, default="rq-coupling",
                    help="Scalar base trf. for f: {affine | quadratic | rq}-{coupling | autoregressive}")
parser.add_argument("--innertransform", type=str, default="rq-coupling",
                    help="Scalar base trf. for h: {affine | quadratic | rq}-{coupling | autoregressive}")
parser.add_argument("--lineartransform", type=str, default="permutation",
                    help="Scalar linear trf: linear | permutation")
parser.add_argument("--outerlayers", type=int, default=5,
                    help="Number of transformations in f (not counting linear transformations)")
parser.add_argument("--innerlayers", type=int, default=5,
                    help="Number of transformations in h (not counting linear transformations)")
parser.add_argument("--conditionalouter", action="store_true",
                    help="If dataset is conditional, use this to make f conditional (otherwise only h is conditional)")
parser.add_argument("--dropout", type=float, default=0.0, help="Use dropout")
parser.add_argument("--pieepsilon", type=float, default=0.01, help="PIE epsilon term")
parser.add_argument("--pieclip", type=float, default=None, help="Clip v in p(v), in multiples of epsilon")
parser.add_argument("--encoderblocks", type=int, default=5, help="Number of blocks in Me-flow / PAE encoder")
parser.add_argument("--encoderhidden", type=int, default=100, help="Number of hidden units in Me-flow / PAE encoder")
parser.add_argument("--splinerange", default=3.0, type=float, help="Spline boundaries")
parser.add_argument("--splinebins", default=8, type=int, help="Number of spline bins")
parser.add_argument("--levels", type=int, default=3,
                    help="Number of levels in multi-scale architectures for image data (for outer transformation f)")
parser.add_argument("--actnorm", action="store_true", help="Use actnorm in convolutional architecture")
parser.add_argument("--batchnorm", action="store_true", help="Use batchnorm in ResNets")
parser.add_argument("--linlayers", type=int, default=2,
                    help="Number of linear layers before the projection for M-flow and PIE on image data")
parser.add_argument("--linchannelfactor", type=int, default=2,
                    help="Determines number of channels in linear trfs before the projection for M-flow and PIE on image data")
parser.add_argument("--intermediatensf", action="store_true",
                    help="Use NSF rather than linear layers before projecting (for M-flows and PIE on image data)")
parser.add_argument("--decoderblocks", type=int, default=5, help="Number of blocks in PAE encoder")
parser.add_argument("--decoderhidden", type=int, default=100, help="Number of hidden units in PAE encoder")

# Training
parser.add_argument("--alternate", action="store_true", help="Use alternating M/D training algorithm")
parser.add_argument("--sequential", action="store_true", help="Use sequential M/D training algorithm")
parser.add_argument("--load", type=str, default=None,
                    help="Model name to load rather than training from scratch, run is affixed automatically")
parser.add_argument("--startepoch", type=int, default=0,
                    help="Sets the first trained epoch for resuming partial training")
parser.add_argument("--samplesize", type=int, default=None, help="If not None, number of samples used for training")
parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
parser.add_argument("--subsets", type=int, default=1, help="Number of subsets per epoch in an alternating training")
parser.add_argument("--batchsize", type=int, default=100, help="Batch size for everything except OT training")
parser.add_argument("--genbatchsize", type=int, default=1000, help="Batch size for OT training")
parser.add_argument("--lr", type=float, default=1.0e-3, help="Initial learning rate")
parser.add_argument("--msefactor", type=float, default=1000.0, help="Reco error multiplier in loss")
parser.add_argument("--addnllfactor", type=float, default=0.1,
                    help="Negative log likelihood multiplier in loss for M-flow-S training")
parser.add_argument("--nllfactor", type=float, default=1.0,
                    help="Negative log likelihood multiplier in loss (except for M-flow-S training)")
parser.add_argument("--sinkhornfactor", type=float, default=10.0, help="Sinkhorn divergence multiplier in loss")
parser.add_argument("--weightdecay", type=float, default=1.0e-4, help="Weight decay")
parser.add_argument("--clip", type=float, default=1.0, help="Gradient norm clipping parameter")
parser.add_argument("--nopretraining", action="store_true", help="Skip pretraining in M-flow-S training")
parser.add_argument("--noposttraining", action="store_true", help="Skip posttraining in M-flow-S training")
parser.add_argument("--validationsplit", type=float, default=0.25,
                    help="Fraction of train data used for early stopping")
parser.add_argument("--scandal", type=float, default=None,
                    help="Activates SCANDAL training and sets prefactor of score MSE in loss")
parser.add_argument("--l1", action="store_true", help="Use smooth L1 loss rather than L2 (MSE) for reco error")
parser.add_argument("--uvl2reg", type=float, default=None,
                    help="Add L2 regularization term on the latent variables after the outer flow (M-flow-M/D only)")
parser.add_argument("--resume", type=int, default=None,
                    help="Resume training at a given epoch (overwrites --load and --startepoch)")

# Other settings
parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
parser.add_argument("--dir", type=str, default="/home/negri0001/Documents/Marcello/cond_flows/rnf/rnf",
                    help="Base directory of repo")
parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")

# evaluate
parser.add_argument("--evaluate", type=int, default=1000, help="Number of test samples to be evaluated")
parser.add_argument("--generate", type=int, default=100000, help="Number of samples to be generated from model")
parser.add_argument("--trueparam", type=int, default=None, help="Index of true parameter point for inference tasks")


args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    device = 'cuda'

    dir_name = "plots/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # load training data
    folder_name = f"/home/negri0001/Documents/Marcello/cond_flows/rnf/rnf/experiments/data/samples/"
    train_data_np = np.load(folder_name+f"{args.dataset}/x_train_{args.epsilon:.2f}.npy")
    test_data_np = np.load(folder_name+f"{args.dataset}/x_test_{args.epsilon:.2f}.npy")
    train_data = torch.from_numpy(train_data_np).float().to(device)

    # build flow
    flow = build_flow_manifold(flow_dim=args.datadim, n_layers=3, hidden_features=256, device=device)
    params = dict(lr=5e-3, epochs=args.epochs_flow, device=device, r=args.r, epsilon=args.epsilon)

    # train flow
    flow_name = f"./models/manifold_flow_{args.dataset}_{args.epsilon:.2f}_e{args.epochs_flow}"
    if not os.path.isfile(flow_name) or args.overwrite:
        flow, loss = train_model(model=flow, data=train_data, dataset=args.dataset, **params)
        plot_loss(loss)
    else:
        build_flow_manifold(flow_dim=args.datadim, n_layers=3, hidden_features=256, device=device)
        flow.load_state_dict(torch.load(flow_name))
        flow.eval()

    # evaluate learnt distribution
    samples_flow, log_probs_flow = generate_samples(flow, sample_size=100, n_iter=100)
    mse_log_probs = evaluate_uniform_on_sphere(d=args.datadim, r=args.r, samples=samples_flow, loglik=log_probs_flow)
    print(f"MSE: {mse_log_probs:.2f}")

    # load mflow model
    args.truelatentdim = args.datadim - 1
    simulator = load_simulator(args)
    rnf = create_model(args, simulator=simulator)
    try:
        rnf.load_state_dict(torch.load(create_filename("model", None, args), map_location=torch.device("cpu")))
    except:
        # Train and save
        dataset = simulator.load_dataset(train=True, dataset_dir=create_filename("dataset", None, args),
                                         limit_samplesize=args.samplesize, joint_score=args.scandal is not None,
                                         epsilon=args.epsilon)
        learning_curves = train_model_rnf(args, dataset, rnf, simulator)

        # Save
        logger.info("Saving model")
        torch.save(rnf.state_dict(), create_filename("model", None, args))
        np.save(create_filename("learning_curve", None, args), learning_curves)
        logger.info("All done! Have a nice day!")

    rnf.eval()
    samples_rnf = sample_from_model(args, rnf, simulator)
    # _, log_probs_rnf, _ = rnf.forward(samples_rnf)

    plot_icosphere(data=train_data_np, simulator=simulator, flow=flow, samples_flow=samples_flow, rnf=rnf, samples_rnf=samples_rnf, device='cuda')
    MSE_flow, MSE_rnf, MSE_dist_flow, MSE_dist_rnf = evaluate_flow_rnf(test_data_np, simulator, flow, rnf, device='cuda')
    print(f"MSE_flow: {MSE_flow:.3f} and MSE_rnf: {MSE_rnf:.3f}")
    print(f"MSE_dist_flow: {MSE_dist_flow:.5f} and MSE_dist_rnf: {MSE_dist_rnf:.5f}")

    MSE_flow, MSE_rnf, MSE_dist_flow, MSE_dist_rnf = evaluate_flow_rnf(train_data_np, simulator, flow, rnf, device='cuda')
    print(f"MSE_flow: {MSE_flow:.3f} and MSE_rnf: {MSE_rnf:.3f}")
    print(f"MSE_dist_flow: {MSE_dist_flow:.5f} and MSE_dist_rnf: {MSE_dist_rnf:.5f}")

    breakpoint()
    # compare samples
    plot_samples(samples_flow)

    test_chatgpt(samples=samples_flow, model=flow)
    plot_icosphere(samples=samples_flow, model=flow)
    plot_pairwise_angle_distribution(samples_flow)


    # simulator = SphericalGaussianSimulator(d-1, d, epsilon=0.01)
    #
    # args.dataset = "spherical_gaussian"
    # args.i = 0
    # args.trueparam = None
    # args.dir = "/home/negri0001/Documents/Marcello/cond_flows/manifold_flow"
    # args.modelname = "manifold_flow"
    # evaluate_model_samples(args, simulator, samples)





if __name__ == "__main__":
    main()