import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from imf.experiments.architecture import build_circular_cond_flow_l1_manifold, build_cond_flow_reverse
from imf.experiments.datasets import generate_regression_dataset
from imf.experiments.plots import plot_betas_lambda
from imf.experiments.utils_manifold import train_regression_cond, generate_samples

parser = argparse.ArgumentParser(description='Process some integers.')

# TRAIN PARAMETERS
parser.add_argument("--device", type=str, default="cuda", help='device for training the model')
parser.add_argument('--epochs', metavar='e', type=int, default=2000, help='number of epochs')
parser.add_argument('--lr', metavar='lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', metavar='s', type=int, default=1234, help='random seed')
parser.add_argument("--overwrite", action="store_true", help="re-train and overwrite flow model")
parser.add_argument('--T0', metavar='T0', type=float, default=2., help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1., help='final temperature')
parser.add_argument('--iter_per_cool_step', metavar='ics', type=int, default=20, help='iterations per cooling step in simulated annealing')
parser.add_argument('--cond_min', metavar='cmin', type=float, default=-2, help='minimum value of conditional variable')
parser.add_argument('--cond_max', metavar='cmax', type=float, default=2, help='minimum value of conditional variable')
parser.add_argument("--log_cond", action="store_true", help="samples conditional values logarithmically")

parser.add_argument("--n_context_samples", metavar='ncs', type=int, default=200, help='number of context samples. Tot samples = n_context_samples x n_samples')
parser.add_argument("--n_samples", metavar='ns', type=int, default=5, help='number of samples per context value. Tot samples = n_context_samples x n_samples')
parser.add_argument('--beta', metavar='be', type=float, default=1.0, help='p of the lp norm')


# MODEL PARAMETERS
parser.add_argument("--n_layers", metavar='nl', type=int, default=5, help='number of layers in the flow model')
parser.add_argument("--n_hidden_features", metavar='nf', type=int, default=128, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--n_context_features", metavar='nf', type=int, default=256, help='number of hidden features in the embedding space of the flow model')
parser.add_argument("--logabs_jacobian", type=str, default="analytical_lu", choices=["analytical_sm", "analytical_lu", "cholesky"])
parser.add_argument("--architecture", type=str, default="circular", choices=["circular", "ambient", "unbounded", "unbounded_circular"])
parser.add_argument("--learn_manifold", action="store_true", help="learn the manifold together with the density")
parser.add_argument("--kl_div", type=str, default="forward", choices=["forward", "reverse"])

args = parser.parse_args()

def to_file(arr, filename):
    if isinstance(arr, np.ndarray):
        df = pd.DataFrame(arr)
    else:
        df = pd.DataFrame(arr.detach().cpu().numpy())
    df.to_csv(filename)


def set_random_seeds(seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_likelihood(beta, sigma, X, y):
    eps = 1e-7
    log_lk = - 0.5 * (y - beta @ X.T).square().sum(-1) / (sigma**2 + eps)
    log_lk_const = - X.shape[0] * np.log((sigma + eps) * np.sqrt(2. * np.pi))

    return log_lk + log_lk_const


def llaplace(beta, lambdas):
    return - lambdas * beta.abs().sum(-1)


def root_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)
    spllap = - torch.sqrt(-llap+1)
    return spllap


def square_offset(beta, lamb_):
    llap = llaplace(beta, lamb_)
    spllap = - torch.pow(-llap+1.0, 4)
    return spllap


def log_prior_act_laplace(beta, lamb, act, args):
    if args.log_cond: lamb_ = 10 ** lamb
    else: lamb_ = lamb

    if act == "laplace_exact":
        log_const = beta.shape[-1] * torch.log(0.5 * lamb_)
        log_prior = llaplace(beta, lamb_)
        log_prior_DE = log_prior + log_const
    elif act == "root_offset":
        log_prior = root_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    elif act == "square_offset":
        log_prior = square_offset(beta, lamb_)
        log_prior_DE = log_prior  # + 0.0 # implement constant
    else:
        raise ValueError("invalid activation name")

    return log_prior_DE


def log_unnorm_posterior(beta, cond, X, y, sigma, act, use_l, use_p):
    log_likelihood_ = log_likelihood(beta, sigma, X, y) if use_l else X.new_zeros((1))
    log_prior_beta_ = log_prior_act_laplace(beta=beta, lamb=cond, act=act, args=args) if use_p else X.new_zeros((1))

    return log_likelihood_ + log_prior_beta_


def main():
    seed = 666  # some bad luck never hurt anyone
    set_random_seeds(seed)

    if visualize_all:
        path_print(0)
        group_print(1)
        exit(0)

    args.log_cond = to_use_log_cond
    match monotonic_act:
        case "square_offset":
            args.cond_min = -3.5
            args.cond_max = 0.2
        case "root_offset":
            args.cond_min = -2.5
            args.cond_max = 2.7
        case "laplace_exact":
            if on_manifold:
                args.cond_min = 0.1#-1.
                args.cond_max = 32.0#np.log10(30)
            else:
                args.cond_min = -2.5
                args.cond_max = 1.2
    Tn = 0.01  # end temperature
    # we define a very straightforward regression problem. can be changed here
    ratio_nonzero = 1
    args.datadim = 5
    # very little data is given. this makes the differences between the subjective priors more visible
    n_samples = 7
    nn_manifold = on_manifold

    args.use_likelihood = True
    args.use_prior = True

    chosen_norm = 5.0
    use_map_norm_matching = True
    map_its = int(0.5 * args.epochs)

    # the sigma parameter should be tuned depending on the noise in the dataset
    sigma_regr = 4.0
    X_np, y_np, true_beta = generate_regression_dataset(n_samples=n_samples, n_features=args.datadim, n_non_zero=int(ratio_nonzero * args.datadim), noise_std=sigma_regr)
    X_tensor = torch.tensor(X_np, device=args.device, dtype=torch.float)
    y_tensor = torch.tensor(y_np, device=args.device, dtype=torch.float)

    # alphas_lasso = np.logspace(args.cond_min, args.cond_max, 200)
    # beta_sklearn = np.array(
    #     [Lasso(alpha=alpha, fit_intercept=False).fit(X_np, y_np).coef_ for alpha in
    #      tqdm.tqdm(alphas_lasso * sigma_regr ** 2 / n_samples)])
    # plt.figure(figsize=(14, 14))
    # plt.plot(alphas_lasso, beta_sklearn)
    # plt.xscale('log')
    # plt.show()

    # define target distribution
    target_distr = partial(log_unnorm_posterior, X=X_tensor, y=y_tensor, sigma=sigma_regr, act=monotonic_act, use_l=args.use_likelihood, use_p=args.use_prior)


    # build model
    args.architecture = "ambient"
    if nn_manifold:
        args.norm = chosen_norm
        args.n_layers = 3
        args.n_hidden_features = 64
        args.n_context_features = 64
        flow = build_circular_cond_flow_l1_manifold(args, star_like=True)
    else:
        flow = build_cond_flow_reverse(args, clamp_theta=False)

    out_name = ("mani_" if nn_manifold else "") + monotonic_act + ".csv"
    if use_map_norm_matching:
        # train model
        flow.train()
        iter_norm = args.epochs
        args.epochs = map_its
        flow, loss, loss_T = train_regression_cond(model=flow, log_unnorm_posterior=target_distr, args=args, tn=Tn, manifold=False)
        args.epochs = iter_norm
        flow.eval()
    #    plot_loss(loss)
        samples, cond, kl = generate_samples(flow, args, n_lambdas=2000, cond=True, log_unnorm_posterior=target_distr,
                                             manifold=False, context_size=2000, sample_size=5, n_iter=1)
        # plot_betas_lambda(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
        #                   min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', true_coeff=None)
        if on_manifold:
            our_cond = chosen_norm
        else:
            beta_norms = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            beta_norms_mask = beta_norms > chosen_norm
            bindex = beta_norms_mask.astype(int).sum(-1)
            our_cond = cond[bindex-1]
        print("our_cond", our_cond)
        to_file(samples.reshape(-1, args.datadim), data_folder + "mapall_" + out_name)
        to_file(cond, data_folder + "mapcall_" + out_name)

        if nn_manifold:
            args.norm = chosen_norm
            # flow = build_simple_cond_flow_l1_manifold(args, n_layers=3, n_hidden_features=64, n_context_features=64, clamp_theta=False)
            args.n_layers = 3
            args.n_hidden_features = 64
            args.n_context_features = 64
            flow = build_circular_cond_flow_l1_manifold(args, star_like=True)
        else:
            flow = build_cond_flow_reverse(args, clamp_theta=False)

    flow.train()
    flow, loss, loss_T = train_regression_cond(model=flow, log_unnorm_posterior=target_distr, args=args, tn=args.Tn, manifold=False)
    flow.eval()

    # evaluate model
    samples, cond, kl = generate_samples(flow, args, n_lambdas=2000, cond=True, log_unnorm_posterior=target_distr,
                                         manifold=False, context_size=20, sample_size=1000, n_iter=100)
    # plot_betas_lambda(samples=samples, lambdas=cond, X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
    #                   min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1, gt='linear_regression', true_coeff=None)
    # opt_cond = plot_marginal_likelihood(kl_sorted=kl, cond_sorted=cond, args=args)
    # print("optimal condition: ", opt_cond.item())

    to_file(samples.reshape(-1,args.datadim), data_folder + "all_" + out_name)
    to_file(cond, data_folder + "call_" + out_name)
    to_file(X_np, data_folder + "xnp_" + out_name)
    to_file(y_np, data_folder + "ynp_" + out_name)

    if not use_map_norm_matching:
        if on_manifold:
            our_cond = chosen_norm
        else:
            beta_norms = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            beta_norms_mask = beta_norms > chosen_norm
            bindex = beta_norms_mask.astype(int).sum(-1)
            our_cond = cond[bindex - 1]
        print("our_cond", our_cond)

    posterior_samples, psslogprob = flow.sample_and_log_prob(1000, context=(np.log10(our_cond) if args.log_cond else our_cond)*torch.ones(1,1,device=args.device))
    posterior_samples = posterior_samples.detach().cpu().numpy()
    posterior_samples = posterior_samples.reshape(-1, posterior_samples.shape[-1])
    to_file(posterior_samples, data_folder + "fnorm_" + out_name)

    print("done")


name_map = {"laplace_exact": "laplace", "square_offset": "square lap",
                "mani_laplace_exact": "objective", "root_offset": "root lap"}
def path_print(variant):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    nlambdas = 2000
    subsample_for_path = 20
    files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and (f.startswith("all_") or f.startswith("mapall_"))]
    names = [f[4:-4] if f.startswith("all_") else "m"+f[7:-4] for f in files]

    sigma_regr = 4.0
    d = [pd.read_csv(data_folder + f).loc[:, '0':] for f in files]
    dms = {}
    dma = []
    name_d_dic = dict(zip(names, d))
    for di, on, map_data in zip(d, names, [f.startswith("mapall_") for f in files]):
        if map_data:
            on = on[1:]
        n = name_map[on]
        print(n if not map_data else "map_" + n)
        samples = di.to_numpy().reshape(nlambdas,-1,5) if not map_data else di.to_numpy().reshape(nlambdas,-1,5)
        cond = pd.read_csv(data_folder + ("call_" if not map_data else "mapcall_") + on + ".csv").loc[:, '0':].to_numpy()[...,0]
        X_np = pd.read_csv(data_folder + "xnp_" + on + ".csv").loc[:, '0':].to_numpy()
        y_np = pd.read_csv(data_folder + "ynp_" + on + ".csv").loc[:, '0':].to_numpy()[...,0]
        of = out_folder + n
        di['s'] = n
        dm = di.melt(id_vars='s', var_name='coeff', value_name='value')
        if name_map["mani_laplace_exact"] in n:
            dms[n] = samples
        if not map_data:
            normmeans = np.linalg.norm(samples, axis=-1, ord=1).mean(-1)
            pdt = pd.DataFrame(normmeans)
            dms[n] = pdt.melt(value_name="norm")
            dms[n]["source"] = n
            mapsamples = pd.read_csv(data_folder + "mapall_" + on + ".csv").loc[:, '0':].to_numpy().reshape(nlambdas, -1, 5)
            mapnormmeans = np.linalg.norm(mapsamples, axis=-1, ord=1).mean(-1)
            norms_mask = mapnormmeans > 5.0 if not name_map["mani_laplace_exact"] in n else mapnormmeans < 5.0
            mapindex = norms_mask.astype(int).sum(-1)
            mapcond = pd.read_csv(data_folder + "mapcall_" + on + ".csv").loc[:, '0':].to_numpy()[...,0]
            targetcond = mapcond[mapindex]
            index = (cond < targetcond).astype(int).sum(-1)
            # this approach can lead to small deviations from the desired norm. saves model storing
            # mostly unnoticeable (ie norm==4.99 instead of 5.0)
            index = index if not name_map["mani_laplace_exact"] in n else index - 1
            pda = pd.DataFrame(np.linalg.norm(samples[index], axis=-1, ord=1))
            pda = pda.melt(value_name="norm")
            pda["source"] = n
            dma.append(pda)

            ssamples = samples[index]
            sns.set(style="whitegrid")
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection='3d')
            ssamples = ssamples.reshape(-1, 5)
            dat = ssamples[:, :3]
            # this is a hacky way to show 5 dim samples on the manifold in 3 dim space
            dat = np.linalg.norm(ssamples, axis=-1, ord=1, keepdims=True) * dat / np.linalg.norm(dat, axis=-1, ord=1,
                                                                                           keepdims=True)
            # the norm manifold
            vertices = 5.0 * np.array(
                [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            faces = [[vertices[0], vertices[1], vertices[2]],
                     [vertices[0], vertices[2], vertices[3]],
                     [vertices[0], vertices[3], vertices[4]],
                     [vertices[0], vertices[4], vertices[1]],
                     [vertices[5], vertices[1], vertices[2]],
                     [vertices[5], vertices[2], vertices[3]],
                     [vertices[5], vertices[3], vertices[4]],
                     [vertices[5], vertices[4], vertices[1]]]

            # Define the faces of the diamond
            poly3d = Poly3DCollection(faces, facecolors='#789cb330', linewidths=0)  # 1, edgecolors='#789cb390')
            ax.add_collection3d(poly3d)
            ax.plot([0, 0, 0, 5, 0], [-5, 0, 5, 0, -5], [0, 5, 0, 0, 0], c='#789cb390', zorder=1)
            ax.plot([0, -5, 0, 5, 0, 0], [-5, 0, 0, 0, 0, -5], [0, 0, 5, 0, -5, 0], c='#789cb390', zorder=1)

            # the samples of the norm
            ax.scatter(dat[:, 1], -dat[:, 0], -dat[:, 2], c=np.abs(dat[:, 0]) + np.abs(dat[:, 2]),
                       cmap='Blues', s=8, zorder=5, marker="o", edgecolors='none')
            ax.grid(False)
            ax._axis3don = False

            plt.savefig(of + "_with_mani.pdf", format='pdf')
            #plt.show()
            plt.close()

        match variant:
            case "trace_norm" | 0:
                plot_betas_lambda(samples=samples[::subsample_for_path], lambdas=cond[::subsample_for_path], X_np=X_np, y_np=y_np, sigma=sigma_regr, gt_only=False,
                                  min_bin=None, max_bin=None, n_bins=51, norm=1, conf=0.95, n_plots=1,
                                  gt='linear_regression', true_coeff=None, name=of+("_betas" if not map_data else "_betas_map"))
                sns.boxplot(data=dm, y="coeff", x="value", fliersize=0)
                plt.title(n)
                plt.savefig(of + "_norms.pdf", format='pdf')
                #plt.show()
                plt.close()

    of = out_folder
    match variant:
        case "trace_norm" | 0:
            plt.figure(figsize=(4, 7))
            dma = [dma[1], dma[2], dma[0]]
            dma = pd.concat(dma, ignore_index=True)
            sns.violinplot(data=dma, x="norm", y="source")
            plt.tight_layout()
            plt.savefig(of + "all_std_norms.pdf", format='pdf')
            #plt.show()
            plt.close()


def group_print(variant):
    import os
    with_obj = False
    files = [f for f in os.listdir(data_folder) if f.endswith(".csv") and f.startswith("fnorm") and (with_obj or not "mani" in f)]
    names = [f[6:-4] for f in files]
    names = [name_map[n] if n in name_map else n for n in names]

    d = [pd.read_csv(data_folder + f).loc[:, '0':] for f in files]
    for di, n in zip(d, names):
        di['s'] = n
    d = [d[2],d[0],d[1]]
    dm = [di.melt(id_vars='s', var_name='coeff', value_name='value') for di in d]
    d = pd.concat(dm)

    of = out_folder
    match variant:
        case "box_no_flier" | 1:
            palette_tab10 = sns.color_palette("tab10", 10)
            palette_blue = list(sns.light_palette(palette_tab10[0], n_colors=3))[::-1][:3]
            palette_salmon = list(sns.light_palette(palette_tab10[1], n_colors=6))[::-1][:1]
            palette = palette_blue + palette_salmon if with_obj else palette_blue
            if with_obj:
                pal = {name_map["laplace_exact"]: palette[0], name_map["square_offset"]: palette[1],
                       name_map["root_offset"]: palette[2], name_map["mani_laplace_exact"]: palette[3]}
            else:
                pal = {name_map["laplace_exact"]: palette[0], name_map["square_offset"]: palette[1],
                       name_map["root_offset"]: palette[2]}#, name_map["mani_laplace_exact"]: palette[3]}
            plt.figure(figsize=(8, 4))
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Computer Modern']
            sns.set_style("whitegrid")
            sns.set_context("notebook", font_scale=2.0)
            # , palette=pal
            sns.boxplot(x='coeff', y='value', hue='s', palette="bright", data=d, fliersize=0)
            sns.set(font_scale=1.7)
            plt.legend(ncol=4 if with_obj else 3, loc="upper left")
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.tight_layout()
            plt.savefig(of + "compare.pdf", format='pdf', bbox_inches='tight')
            #plt.show()
            plt.close()


data_folder = "data/runs/"
out_folder = "data/out/"
os.makedirs(data_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)
monotonic_act = ""
on_manifold = True
visualize_all = False
if __name__ == "__main__":
    if len([f for f in os.listdir(data_folder) if f.endswith(".csv")]) > 10:
        print("detected existing runs in " + data_folder)
        print("running only visualization code. if a rerun is required or not all files were generated, delete the folder " + data_folder)
        visualize_all = True
        main()
    else:
        monotonic_act = "laplace_exact"
        to_use_log_cond = False
        main()
        to_use_log_cond = True
        on_manifold = False
        main()
        monotonic_act = "square_offset"
        main()
        monotonic_act = "root_offset"
        main()
        visualize_all = True
        main()
