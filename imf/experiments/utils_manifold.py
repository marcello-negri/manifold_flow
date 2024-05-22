import numpy as np
import torch
import time
import os
import tqdm
from datetime import timedelta
from torch.utils.data import Dataset

import logging
logger = logging.getLogger(__name__)

def define_model_name(args, dataset):
    args.model_name = (f"./models/imf_{args.dataset}_{args.architecture}_lm{args.learn_manifold}_{args.logabs_jacobian}"
                       f"{dataset.dataset_suffix}_epochs{args.epochs}_seed{args.seed}")

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



def train_model_forward(model, data, args, batch_size=1, context=None, alternating=False, early_stopping=False, device="cuda", **kwargs):
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

def train_model_reverse(model, args, dataset, train_data_np, batch_size=100_000, context=None, **kwargs):
    from imf.experiments.plots import plot_angle_distribution, plot_samples_pca

    # optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':log_sigma, 'lr':1e-2}], lr=lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    num_iter = args.epochs // args.iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=args.T0, Tn=args.Tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss = []
    try:
        start_time = time.monotonic()
        model.train()
        for epoch in range(args.epochs):
            T = cooling_function(epoch // (args.epochs / num_iter))

            optimizer.zero_grad()
            samples, logprob_flow = model.sample_and_log_prob(num_samples=batch_size, context=None)
            logprob_target = dataset.log_density(samples) # uniform on lp manifold

            kl_div = torch.mean(logprob_flow - logprob_target/T)
            kl_div.backward()


            if epoch > 0:
                T_old = cooling_function((epoch - 1) // (args.epochs / num_iter))
                if T != T_old:
                    samples_np = samples.detach().cpu().numpy()
                    plot_angle_distribution(samples_flow=samples_np, samples_gt=train_data_np, device=args.device,
                                            filename=f"./plots/{args.model_name.split('/')[-1]}_angles_T{T:.2f}.png")
                    plot_samples_pca(samples_np, train_data_np,
                                     filename=f"./plots/{args.model_name.split('/')[-1]}_angles_pca_T{T:.2f}.png")

            # torch.nn.utils.clip_grad_value_(model.parameters(), 1e-4)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-4)
            optimizer.step()

            kl_div_ = kl_div.cpu().detach().numpy()
            loss.append(kl_div_)
            # print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")
            print(f"Training loss at step {epoch} (T={T:.2f}): {loss[-1]:.3f}")
            # print(f"logprob_flow: {logprob_flow.mean().cpu().detach().numpy():.3f} ")
                  # f"logprob_target: {logprob_target.mean().cpu().detach().numpy():.3f}")
            # if epoch % 20 ==0:
            #     scheduler.step()

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


# def generate_samples (model, sample_size=100, n_iter=1000):
#     samples, log_probs = [], []
#     for _ in tqdm.tqdm(range(n_iter)):
#         posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size)
#         samples.append(posterior_samples.cpu().detach().numpy())
#         log_probs.append(log_probs_samples.cpu().detach().numpy())
#     samples = np.concatenate(samples, 0)
#     log_probs = np.concatenate(log_probs, 0)
#
#     return samples, log_probs

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

