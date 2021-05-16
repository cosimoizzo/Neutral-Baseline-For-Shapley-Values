# Reference: https://distill.pub/2020/attribution-baselines/
import numpy as np
import random


# P data baseline - random obs.
def p_data_bs(x, seed=1, n_draws=10):
    """
    :param x: training set
    :param seed: seed
    :param n_draws: number of draws
    :return: baselines
    """
    random.seed(seed)
    inds = []
    n_obs = x.shape[0]
    j = 0
    while j < n_draws:
        draw = random.randint(1, n_obs)
        if draw not in inds:
            inds.append(draw)
            j += 1
    return x[inds]


# Maximum distance baseline
def maximum_distance_bs(x, current_x):
    """
    :param x: training set
    :param current_x: current sample
    :return: baseline
    """
    min_value = np.min(x, 0)
    max_value = np.max(x, 0)
    ave_value = np.mean(x, 0)
    baseline_md = min_value.copy()
    baseline_md[current_x < ave_value] = max_value[current_x < ave_value]
    return baseline_md
