import numpy as np
import rpy2.robjects as robjects
import matplotlib.pyplot as plt
import numpy as np

def lp_norm(arr, p):
    norm = np.sum(np.power(np.abs(arr), p), 1)
    norm = np.power(norm, 1/p).reshape(-1,1)
    return arr/norm

# Call the function with any required arguments
def sample_generalized_normal(n_samples, beta, alpha=None, mu=0):
    robjects.r.source("../utils.R")
    sample_gen_norm = robjects.r['sample_gen_norm']

    if alpha is None:
        alpha = beta ** (1. / beta)

    samples = sample_gen_norm(x=3 * n_samples, alpha=alpha, beta=beta, mu=mu)
    breakpoint()
    samples = np.array(samples)
    samples = samples.reshape((n_samples, 3))
    samples_norm = lp_norm(samples, p=beta)

    return samples_norm

n_samples=10_000
samples = sample_generalized_normal(n_samples=n_samples, beta=1.)
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(projection='3d')
ax.scatter3D(samples[:,0], samples[:,1], samples[:,2], alpha=0.1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
plt.show()