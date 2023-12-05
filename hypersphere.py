import numpy as np
import torch
import os
from utils_manifold import build_flow_manifold, train_model, generate_spherical_with_noise, plot_loss, generate_samples, evaluate_uniform_on_sphere
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--epochs', metavar='e', type=int, default=1_000,
                    help='number of epochs')
parser.add_argument('--seed', metavar='s', type=int, default=1234,
                    help='random seed')
parser.add_argument('--d', metavar='d', type=int, default=3,
                    help='number of dimensions')
parser.add_argument('--n', metavar='n', type=int, default=1000,
                    help='number of training samples')
parser.add_argument('--r', metavar='r', type=float, default=1.,
                    help='radius of d-dimensional sphere')
parser.add_argument('--std', metavar='std', type=float, default=0.2,
                    help='std of the isotropic noise in the data')
args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)
    device = 'cuda'

    dir_name = "./plots/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # generate training data
    data, data_np = generate_spherical_with_noise(n=args.n, d=args.d, r=args.r, std=args.std)

    # build flow
    flow = build_flow_manifold(flow_dim=args.d, n_layers=3, hidden_features=256, device=device)

    params = dict(lr=1e-3, epochs=args.epochs, device=device, r=args.r)

    # train flow
    flow, loss = train_model(model=flow, data=data, **params)
    plot_loss(loss)

    # evaluate learnt distribution
    samples, log_probs = generate_samples(flow)
    mse_log_probs = evaluate_uniform_on_sphere(args.d, args.r, log_probs)
    print(f"MSE: {mse_log_probs:.2f}")

if __name__ == "__main__":
    main()