import torch


def gaussian_elimination(A, b):
    mb, d = A.shape[:2]
    alphas = torch.ones((mb, d-1), device=A.device)
    alphas[:,0] = A[:,-1,0] / A[:,0,0]
    for i in range(1,d-1):
        alphas[:,i] = (A[:,-1,i] - torch.sum(A[:,:i,i] * alphas[:,:i], -1) ) / A[:,i,i]

    A[:, -1, -1] -= torch.sum(A[:, :-1, -1] * alphas, -1)
    b[:, -1] -= torch.sum(b[:, :-1] * alphas, -1)

    solution = torch.linalg.solve_triangular(A.triu(), b.unsqueeze(-1), upper=True).squeeze()

    return solution

def explicit_inversion(A, b):
    solution = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze()

    return solution

def lu_inversion(A, b):
    LU, pivots = torch.linalg.lu_factor(A)
    solution = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1)).squeeze()

    return solution

def lu_inversion_transpose(A, b):
    LU, pivots = torch.linalg.lu_factor(A.mT)
    solution = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(-1), adjoint=True).squeeze()

    return solution

# solve A^-1 * b = x as A * x = b
# when A is almost tringular (see below)

mb = 20
d = 10
A_triu = torch.rand((mb, d, d)).triu()
A_triu[:,-1,:] = torch.rand((mb, d)) # last row is non-zero

vector = torch.rand((mb, d))
# brute force solution
bf_sol = explicit_inversion(A_triu, vector)
lu_sol = lu_inversion(A_triu, vector)
lu_t_sol = lu_inversion_transpose(A_triu, vector)
gauss_sol = gaussian_elimination(A_triu, vector)

import numpy as np
import time
import pandas as pd
import seaborn as sns
import tqdm
import matplotlib.pyplot as plt

device = 'cuda'
mb = 1
n_seeds = 21
df_list = []
methods = ["brufe_force", "LU", "LUT", "gauss"]
for d in tqdm.tqdm(np.logspace(1, 4, num=20, dtype='int')):
    for seed in range(n_seeds):
        A_triu = torch.rand((mb, d, d), device=device).triu()
        A_triu[:,-1,:] = torch.rand((mb, d), device=device) # last row is non-zero
        vector = torch.rand((mb, d), device=device)
        for method in methods:
            start_time = time.monotonic()
            if method == "brufe_force":
                explicit_inversion(A_triu, vector)
            elif method == "LU":
                lu_inversion(A_triu, vector)
            elif method == "LUT":
                lu_inversion_transpose(A_triu, vector)
            elif method == "gauss":
                gaussian_elimination(A_triu, vector)
            end_time = time.monotonic()
            total_time = end_time - start_time
            if seed >0 :
                df_dict = dict(d=d, time=total_time, method=method, seed=seed)
                df_list.append(df_dict)

df_data = pd.DataFrame(df_list)
sns.boxplot(data=df_data, x="d", y="time", hue="method")
plt.show()
