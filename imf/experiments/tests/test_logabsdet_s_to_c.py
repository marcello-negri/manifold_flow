from enflows.transforms.injective.utils import logabsdet_sph_to_car, spherical_to_cartesian_torch
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

n_dim = 100
n_samples = 1000
logabsdet_pd = []
for sampler in ["uniform", "gaussian"]:
    if sampler == "uniform":
        theta = torch.rand((n_samples, n_dim-1)) * np.pi
        theta[:,-1] *= 2
    elif sampler == "gaussian":
        theta = torch.randn((n_samples, n_dim-1))*0.1
        theta = torch.sigmoid(theta)
        theta *= np.pi
        theta[:, -1] *= 2

    plt.hist(theta[:,0].detach().numpy(), bins=20, alpha=0.5)
    plt.hist(theta[:,-1].detach().numpy(), bins=20, alpha=0.5)
    plt.show()

    for r in np.logspace(0, 5, 10, dtype="int"):
        radius = torch.ones((n_samples,1)) * r
        theta_r = torch.cat((theta, radius), dim=-1)
        logabsdet = logabsdet_sph_to_car(theta_r)
        df_ = pd.DataFrame(logabsdet, columns=["logabsdet"])
        df_["radius"] = r
        df_["sampler"] = sampler
        logabsdet_pd.append(df_)

logabsdet_pd = pd.concat(logabsdet_pd)
ax = sns.boxplot(x="radius", y="logabsdet", hue="sampler", data=logabsdet_pd)
plt.xticks(fontsize=14)
plt.show()
breakpoint()