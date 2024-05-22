import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression


def generate_regression_dataset(n_samples, n_features, n_non_zero, noise_std):
    assert n_features >= n_non_zero

    # Generate non-zero coefficients randomly
    non_zero_indices = np.random.choice(n_features, n_non_zero, replace=False)
    coefficients = np.zeros(n_features)
    coefficients[non_zero_indices] = np.random.normal(0, 1, n_non_zero)  # Random non-zero coefficients

    # Generate data matrix X from a Gaussian distribution with covariance matrix sampled from a Wishart distribution
    scale_matrix = np.eye(n_features)  # Identity matrix as the scale matrix
    covariance = sp.stats.wishart(df=n_features, scale=scale_matrix).rvs(1)

    # Sample data matrix X from a multivariate Gaussian distribution with zero mean and covariance matrix
    X = np.random.multivariate_normal(mean=np.zeros(n_features), cov=covariance, size=n_samples)

    # Generate response variable y
    y = np.dot(X, coefficients) + np.random.normal(0, noise_std, n_samples)  # Linear regression model with Gaussian noise

    # compute regression parameters
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    print(f"R^2 score: {r2_score:.4f}")
    sigma_regr = np.sqrt(np.mean(np.square(y - X @ reg.coef_)))
    print(f"Sigma regression: {sigma_regr:.4f}")
    print(f"Norm coefficients: {np.linalg.norm(reg.coef_):.4f}")

    return X, y, coefficients
