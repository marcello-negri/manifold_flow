from enflows.transforms.injective.utils import logabsdet_sph_to_car, spherical_to_cartesian_torch, cartesian_to_spherical_torch
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
from matplotlib.ticker import StrMethodFormatter

def find_logabsdet(value, logabsdet, radius):
    logabsdet = np.asarray(logabsdet)
    dist_2 = (logabsdet - value)**2
    opt_idx = np.argmin(dist_2)
    return radius[opt_idx]

r_min, r_max = 1., 2.
n_r = 500
radius_list = np.linspace(r_min, r_max, n_r)

d_min, d_max = 0.5, 4
n_d = 50
d_list = np.unique(np.logspace(d_min, d_max, n_d, dtype="int"))
n_samples = 1000

optimal_r = []
for n_dim in tqdm.tqdm(d_list):
    logabsdet_pd = []
    for sampler in ["uniform_sphere"]:#"gaussian", "uniform"]:
        if sampler == "uniform":
            theta = torch.rand((n_samples, n_dim-1)) * np.pi
            theta[:,-1] *= 2
        elif sampler == "gaussian":
            theta = torch.randn((n_samples, n_dim-1))
            theta = torch.sigmoid(theta)
            theta *= np.pi
            theta[:, -1] *= 2
        elif sampler == "uniform_sphere":
            samples = torch.randn((n_samples, n_dim))
            samples /= torch.norm(samples, dim=-1).reshape(-1, 1)
            theta = cartesian_to_spherical_torch(samples)[:, :-1]

        # plt.hist(theta[:,0].detach().numpy(), bins=20, alpha=0.5)
        # plt.hist(theta[:,-1].detach().numpy(), bins=20, alpha=0.5)
        # plt.show()

        for r in radius_list:#, dtype="int"):
            radius = torch.ones((n_samples,1)) * r
            theta_r = torch.cat((theta, radius), dim=-1)
            logabsdet = logabsdet_sph_to_car(theta_r)
            df_ = pd.DataFrame(logabsdet, columns=["logabsdet"])
            df_["radius"] = r
            df_["sampler"] = sampler
            logabsdet_pd.append(df_)

        logabsdet_pd = pd.concat(logabsdet_pd)

        # ax = sns.boxplot(x="radius", y="logabsdet", hue="sampler", data=logabsdet_pd)
        # ax.axhline(0, color=".3", dashes=(2, 2))

        logabsdet_list = np.array(logabsdet_pd.groupby(by=["radius"])["logabsdet"].mean())
        opt_r = find_logabsdet(0, logabsdet_list, radius_list)
        optimal_r.append(opt_r)
        # plt.xticks(fontsize=14)
        # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        # plt.show()

plt.plot(d_list, optimal_r, "o-", label="optimal radius")
plt.xscale("log")
plt.show()

breakpoint()