from enflows.distributions.uniform import UniformSphere
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_angle_distribution(samples, log_prob):

    prob = np.exp(log_prob)
    n_dim = samples.shape[-1]
    n_rows = int(np.sqrt(n_dim))
    if n_rows**2 != n_dim: n_rows += 1
    n_cols = n_rows

    fig, axs = plt.subplots(figsize=(14,14), nrows=n_rows, ncols=n_cols)

    for i_r in np.arange(n_rows):
        for i_c in np.arange(n_rows):
            try:
                right_range = np.pi
                if n_rows*i_r+i_c+1 == samples.shape[1]: right_range *= 2
                axs[i_r, i_c].hist(samples[:,n_rows*i_r+i_c], bins=50, alpha=0.5, range=(0, right_range), label="gt", density=True)
                # axs[i_r, i_c].plot(samples[:,n_rows*i_r+i_c], prob[n_rows*i_r+i_c], range=(0, right_range), label="flow")
                if n_rows*i_r+i_c+1 == samples.shape[1]: axs[i_r, i_c].legend()
            except:
                pass

    plt.show()

n_dim = 2
n_samples = 1000
grid = torch.ones(n_samples, n_dim)
linspace_x = torch.linspace(0,np.pi,100)
linspace_y = torch.linspace(0,2*np.pi,200)
mesh_x, mesh_y = torch.meshgrid(linspace_x, linspace_y)
delta_x = linspace_x[1]-linspace_x[0]
delta_y = linspace_y[1]-linspace_y[0]
mesh_xx = mesh_x[1:,1:]-delta_x*0.5
mesh_yy = mesh_y[1:,1:]-delta_y*0.5
mesh = torch.cat((mesh_xx.reshape(-1,1), mesh_yy.reshape(-1,1)), dim=1)

base_dist = UniformSphere(shape=[n_dim])
samples = base_dist.sample(n_samples * 10)

log_prob = base_dist.log_prob(mesh)
integral = (log_prob.exp()*delta_x*delta_y).sum()

breakpoint()
plot_angle_distribution(samples.detach().numpy(), log_prob.detach().numpy())




