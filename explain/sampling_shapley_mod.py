# Function Shapley Sampling modified for Sparse Net
import numpy as np


def shapley_sampling(model, inputs, y_hat, runs=20, baseline=None, input_sparse=None):
    """
    :param model: keras model
    :param inputs: input features
    :param y_hat: prediction
    :param runs: number of runs
    :param baseline: baseline (aka reference point), default is 0
    :param input_sparse: optional input for the sparse network
    :return: Approximate Shapley values via sampling
    """
    results = np.zeros_like(inputs)

    if baseline is None:
        baseline = np.zeros_like(inputs)

    n_features = inputs.shape[0]

    n_baselines = int(np.size(baseline, 0) / n_features)

    if n_baselines > 1:
        if input_sparse is None:
            inputs = np.kron(np.ones(n_baselines), inputs)
        else:
            inputs = input_sparse

    for _ in range(runs):
        p = np.random.permutation(n_features)
        x = inputs.copy()
        y = None
        for i in p:
            if y is None:
                y = model.predict(np.array(x).reshape((1, -1)))
            for j in range(n_baselines):
                x[i + j * n_features] = baseline[i + j * n_features]
            y0 = model.predict(np.array(x).reshape((1, -1)))
            results[i] += np.sum((y - y0) * y_hat, 1)
            y = y0

    results = results / runs
    return results.reshape(n_features)



