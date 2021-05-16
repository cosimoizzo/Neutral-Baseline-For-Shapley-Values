import numpy as np


def generate(n_sim, n_out_of_sample, n_vars=3, degree=1, identity=True, bias=True, random_sign=True):
    """
    :param n_sim: number of simulations
    :param n_out_of_sample: number observations for the test set
    :param n_vars: number of features (defaults is 3)
    :param degree: degree of the interaction among features (default is 1)
    :param identity: whether to generate correlated or uncorrelated features (default is unc.)
    :param bias: whether to add or not the bias term (default is True)
    :param random_sign: whether to draw the sign of the interaction (default is True)
    :return: x train, x test, y train, y test
    """
    # Gen from Normal
    if identity:
        # Identity matrix
        cov = np.identity(n_vars)
    else:
        # Random Covariance Matrix
        cov = np.random.random((n_vars, n_vars))  # np.identity(n_vars)
        while np.linalg.det(cov) <= 0.01:
            cov = np.random.random((n_vars, n_vars))
    x_all = np.random.multivariate_normal(np.zeros(n_vars), cov, n_sim + n_out_of_sample).T
    if random_sign:
        p_sign = np.zeros(n_vars)
        for j in range(n_vars):
            if np.random.binomial(1, 0.5) == 0:
                p_sign[j] = 1
            else:
                p_sign[j] = -1
    else:
        p_sign = np.ones(n_vars)
    # Shuffle
    np.random.shuffle(x_all.T)
    # Compute y*
    y_latent = 0
    for deg in range(0, degree):
        for j in range(n_vars):
            if random_sign:
                coef = np.exp(np.random.normal(j + 1, 1))
            else:
                coef = np.random.normal(j + 1, 1)
            y_latent += p_sign[j] * coef * (x_all[j, :] ** (deg + 1))
    if bias:
        y_latent += np.random.uniform(low=-15, high=+15)
    # compute binary output
    y = np.zeros_like(y_latent)
    perc_ts1 = np.random.uniform(0.3, 0.5) * 100
    perc_ts2 = perc_ts1 + np.random.uniform(0.3, 0.5) * 100
    y[(y_latent > np.percentile(y_latent, perc_ts1)) & (y_latent < np.percentile(y_latent, perc_ts2))] = 1
    x_train = x_all[:, n_out_of_sample:]
    x_new = x_all[:, :n_out_of_sample]
    y_train = y[n_out_of_sample:]
    y_new = y[:n_out_of_sample]
    return x_train.T, x_new.T, y_train, y_new
