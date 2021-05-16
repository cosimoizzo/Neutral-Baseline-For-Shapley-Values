# Function exact Shapley modified for Sparse Net
from itertools import chain, combinations

import numpy as np
from scipy.special import factorial as fact


def compute_shapley_mod(inputs, model, baseline=None, input_sparse=None):
    """
    :param inputs: input data
    :param model: keras model
    :param baseline: baseline (if None then it is set to 0)
    :param input_sparse: input data for sparse model
    :return: Shapley values
    """
    if baseline is None:
        baseline = np.zeros_like(inputs)
    results = np.zeros_like(inputs)
    n = inputs.shape[0]
    assert inputs.shape == (n,), inputs.shape
    # create mask
    mask = vec_bin_array(np.arange(2 ** (n - 1)), n - 1)
    # check dimensions
    assert mask.shape == (2 ** (n - 1), n - 1)
    coef = (fact(mask.sum(1)) * fact(n - mask.sum(1) - 1)) / fact(n)
    for index in range(n):
        # Copy mask and set the current player active
        mask_wo_index = np.insert(mask, index, np.zeros(2 ** (n - 1)), axis=1)
        mask_wi_index = np.insert(mask, index, np.ones(2 ** (n - 1)), axis=1)
        assert mask_wo_index.shape == (2 ** (n - 1), n), 'Mask shape does not match'
        assert np.max(mask_wo_index) == 1, np.max(mask_wo_index)
        assert np.min(mask_wo_index) == 0, np.min(mask_wo_index)
        # Calculate number of baselines
        n_baseline_x_feature = int(np.size(baseline, 0) / n)
        if n_baseline_x_feature > 1:
            # Kron on mask
            mask_wo_index_new = np.kron(np.ones(n_baseline_x_feature), mask_wo_index)
            mask_wi_index_new = np.kron(np.ones(n_baseline_x_feature), mask_wi_index)
            # Kron on input
            if input_sparse is None:
                inputs_new = np.kron(np.ones(n_baseline_x_feature), inputs)
            else:
                inputs_new = input_sparse.copy()
        else:
            inputs_new = inputs
            mask_wo_index_new = mask_wo_index
            mask_wi_index_new = mask_wi_index
        run_wo_i = model(inputs_new * mask_wo_index_new + baseline * (1 - mask_wo_index_new))  # run all masks at once
        run_wi_i = model(inputs_new * mask_wi_index_new + baseline * (1 - mask_wi_index_new))  # run all masks at once
        r = (run_wi_i - run_wo_i) * coef
        results[index] = r.sum()
    return results


# sub-functions

def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))


def vec_bin_array(arr, m):
    to_str_func = np.vectorize(lambda x: np.binary_repr(x).zfill(m))
    strs = to_str_func(arr)
    ret = np.zeros(list(arr.shape) + [m], dtype=np.int8)
    for bit_ix in range(0, m):
        fetch_bit_func = np.vectorize(lambda x: x[bit_ix] == '1')
        ret[..., bit_ix] = fetch_bit_func(strs).astype("int8")
    return ret
