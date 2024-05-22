import logging
import time
from datetime import timedelta

import numpy as np
import torch
import tqdm

logger = logging.getLogger(__name__)

def spherical_to_cartesian_torch(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)

    assert arr.shape[1] > 2
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
    assert arr.shape[-1] > 2
    radius = torch.linalg.norm(arr, dim=-1)
    flipped_cumsum = torch.cumsum(torch.flip(arr ** 2, dims=(-1,)), dim=-1)
    sqrt_sums = torch.flip(torch.sqrt(flipped_cumsum + eps), dims=(-1,))[..., :-1]
    angles = torch.acos(arr[..., :-1] / (sqrt_sums + eps))
    last_angle = ((arr[..., -1] >= 0).float() * angles[..., -1] + \
                  (arr[..., -1] < 0).float() * (2 * np.pi - angles[..., -1]))

    return torch.cat((radius.unsqueeze(-1), angles[..., :-1], last_angle.unsqueeze(-1)), dim=-1)


def gen_cooling_schedule(T0, Tn, num_iter, scheme):
    def cooling_schedule(t):
        if t < num_iter:
            k = t / num_iter
            if scheme == 'exp_mult':
                alpha = Tn / T0
                return T0 * (alpha ** k)
            #elif scheme == 'log_mult':
            #    return T0 / (1 + alpha * math.log(1 + k))
            elif scheme == 'lin_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * k)
            elif scheme == 'quad_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * (k ** 2))
        else:
            return Tn
    return cooling_schedule



def train_regression_cond(model, log_unnorm_posterior, args, manifold, tn, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # set up cooling schedule
    num_iter = args.epochs // args.iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=args.T0, Tn=tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss, loss_T = [], []
    try:
        start_time = time.monotonic()
        for epoch in range(args.epochs):
            T = cooling_function(epoch // (args.epochs / num_iter))
            optimizer.zero_grad()

            rand_cond = torch.rand(args.n_context_samples, device=args.device)
            # alpha = min(epoch/(args.epochs-200), 1.)
            # uniform_cond = (args.cond_max - rand_cond * alpha * (args.cond_max-args.cond_min)).view(-1, 1)
            # print(uniform_cond.min().item(), uniform_cond.max().item())

            uniform_cond = (rand_cond * (args.cond_max - args.cond_min) + args.cond_min).view(-1, 1)

            samples, log_prob = model.sample_and_log_prob(num_samples=args.n_samples, context=uniform_cond)
            if torch.any(torch.isnan(samples)): breakpoint()

            if manifold:
                log_posterior = log_unnorm_posterior(beta=samples)
            else:
                log_posterior = log_unnorm_posterior(beta=samples, cond=uniform_cond)

            if isinstance(log_posterior, tuple):
                _, log_prior, log_like = log_posterior
                # k = 4.0  # >1 reduces the impact on the prior. <1 increases impact on prior. <0 for no impact
                k =0.1  # >1 reduces the impact on the prior. <1 increases impact on prior. <0 for no impact
                if T > 1:
                    Tp = (T+k-1)/k
                else:
                    Tp = T
                log_posterior_noT = log_prior+log_like
                log_posterior = log_prior / Tp + log_like / T
            else:
                log_posterior_noT = log_posterior
                log_posterior = log_posterior / T
            kl_div = torch.mean(log_prob - log_posterior)
            kl_div.backward()

            optimizer.step()

            loss.append(torch.mean(log_prob - log_posterior_noT).cpu().detach().numpy())
            loss_T.append(torch.mean(log_prob - log_posterior / T).cpu().detach().numpy())
            if epoch % 10 == 0:
                print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")

    except KeyboardInterrupt:
        print("interrupted..")

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T


def generate_samples(model, args, n_lambdas=0, cond=False, log_unnorm_posterior=None, manifold=True, context_size=10, sample_size=100, n_iter=1000):
    it = 0
    samples_list, log_probs_list, kl_list, cond_list = [], [], [], []
    if n_lambdas != 0:
        full_lambdas = torch.linspace(args.cond_min, args.cond_max, n_lambdas, device=args.device).view(n_iter, -1, 1)
    for _ in tqdm.tqdm(range(n_iter)):
        # it = it + 1
        if cond and log_unnorm_posterior is not None:
            if n_lambdas == 0:
                rand_cond = torch.rand(context_size, device=args.device)
                uniform_cond = (rand_cond * (args.cond_max - args.cond_min) + args.cond_min).view(-1, 1)
            else:
                uniform_cond = full_lambdas[it]
            # if (args.cond_max < 1 and args.cond_min < 1):
            #     factor = it / n_iter
            #     uniform_cond = factor * uniform_cond + (1-factor)*1.0
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=uniform_cond)
            if manifold:
                log_lik = log_unnorm_posterior(beta=posterior_samples)
            else:
                log_lik = log_unnorm_posterior(beta=posterior_samples, cond=uniform_cond)
            if isinstance(log_lik, tuple):
                log_lik = log_lik[0]
            kl_div = log_probs_samples - log_lik
            kl_list.append(kl_div.detach().cpu().numpy())
            if args.log_cond: uniform_cond = 10 ** uniform_cond
            cond_list.append(uniform_cond.view(-1).cpu().detach().numpy())
        else:
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)

        samples_list.append(posterior_samples.detach().cpu().numpy())
        log_probs_list.append(log_probs_samples.detach().cpu().numpy())
        it = it + 1

    samples_list = np.concatenate(samples_list, 0)
    log_probs_list = np.concatenate(log_probs_list, 0)

    if cond and log_unnorm_posterior is not None:
        cond_list = np.concatenate(cond_list, 0)
        kl_list = np.concatenate(kl_list, 0)
        cond_sorted_idx = cond_list.argsort()
        return samples_list[cond_sorted_idx], cond_list[cond_sorted_idx], kl_list[cond_sorted_idx]
    else:
        return samples_list, log_probs_list

