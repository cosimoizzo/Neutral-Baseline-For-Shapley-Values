import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
import tensorflow as tf
from explain.mlp import fit_mlp


# ROAR
def local_roar(x_train, x_test, y_train, y_test, abs_importance_train, abs_importance_test,
               conf, n, n_train=1, lr=0.05, replace_with_train=None, replace_with_test=None):
    """
    :param x_train: training data
    :param x_test: testing data
    :param y_train: training target
    :param y_test: testing target
    :param abs_importance_train: absolute importance of the training features
    :param abs_importance_test: absolute importance of the testing features
    :param conf: model structure
    :param n: number of features to remove
    :param n_train: number of times to re-train
    :param lr: learning rate
    :param replace_with_train: what to replace with dropped training data
    :param replace_with_test: what to replace with dropped testing data
    :return: average precision score
    """
    x_train_ = x_train.copy()
    x_test_ = x_test.copy()
    n_obs_train, n_obs_test = x_train_.shape[0], x_test_.shape[0]
    if n_obs_train < n_obs_test:
        raise ValueError("Test data should be less than training data")
    arg_abs_importance_train = abs_importance_train.argsort(axis=1)
    arg_abs_importance_test = abs_importance_test.argsort(axis=1)
    if replace_with_train is None:
        replace_with_train = [np.zeros(arg_abs_importance_train.shape[1])]
        replace_with_test = [np.zeros(arg_abs_importance_train.shape[1])]
    res_ = 0
    for this_repl in range(len(replace_with_train)):
        for j in range(n_obs_train):
            for i in range(n + 1):
                if replace_with_train[this_repl].ndim == 2:
                    x_train_[j, arg_abs_importance_train[j, - i - 1]] = replace_with_train[this_repl][
                        j, arg_abs_importance_train[j, - i - 1]]
                    if j < n_obs_test:
                        x_test_[j, arg_abs_importance_test[j, - i - 1]] = replace_with_test[this_repl][
                            j, arg_abs_importance_train[j, - i - 1]]
                else:
                    x_train_[j, arg_abs_importance_train[j, - i - 1]] = replace_with_train[this_repl][
                        arg_abs_importance_train[j, - i - 1]]
                    if j < n_obs_test:
                        x_test_[j, arg_abs_importance_test[j, - i - 1]] = replace_with_test[this_repl][
                            arg_abs_importance_train[j, - i - 1]]
        res = 0
        for i in range(n_train):
            tf.keras.backend.clear_session()
            model, _ = fit_mlp(x_train_, y_train, conf, seed=3 + i, lr=lr)
            y_hat_test = model.predict(x_test_)
            res += average_precision_score(y_test, y_hat_test)
        res /= n_train
        res_ += res
    return res_ / len(replace_with_train)


# local analysis - perturbation tests
def local_analysis(model, x_test, a_attr, reference, asc=False, n_base_x_feat=1, abs_value=True, log_odds=True,
                   feat_rem_x_run=1):
    """
    :param model: model
    :param x_test: testing data
    :param a_attr: feature attributions scores
    :param reference: reference/baseline used
    :param asc: whether to replace in ascending or descending order (default is descending)
    :param n_base_x_feat: number of baseline per feature
    :param abs_value: use absolute value of of the attribution or not (default is True)
    :param log_odds: whether to use log_odds as score function
    :param feat_rem_x_run: number of feature to remove at each iteration
    :return: score and absolute score
    """
    if abs_value:
        a_attr = np.abs(a_attr)
    sorted_a_attr = np.zeros_like(a_attr)
    n_obs = sorted_a_attr.shape[0]
    n_vars = sorted_a_attr.shape[1]
    for obs in range(n_obs):
        if asc:
            sorted_a_attr[obs, :] = np.sort(a_attr[obs, :])
        else:
            sorted_a_attr[obs, :] = np.sort(a_attr[obs, :])[::-1]
    out_ = np.zeros((sorted_a_attr.shape[0], sorted_a_attr.shape[1] // feat_rem_x_run + 1))
    abs_out_ = np.zeros(out_.shape)
    if n_base_x_feat > 1:
        x_for_a_attr = np.zeros((2, len(reference)))
    else:
        x_for_a_attr = np.zeros((2, n_vars))
    for j_obs in range(n_obs):
        if n_base_x_feat > 1:
            x_for_a_attr[0, :] = np.kron(np.ones(n_base_x_feat), x_test[j_obs, :])
        else:
            x_for_a_attr[0, :] = x_test[j_obs, :]
        predictionsmodel_a_attr = model.predict(x_for_a_attr)
        if log_odds:
            out_[j_obs, 0] = np.log(predictionsmodel_a_attr[0][0] / (1 - predictionsmodel_a_attr[0][0]))
        else:
            out_[j_obs, 0] = predictionsmodel_a_attr[0][0]
        abs_out_[j_obs, 0] = np.abs(out_[j_obs, 0])
        for var in range(n_vars // feat_rem_x_run):
            # find index
            index_feat = np.zeros(feat_rem_x_run).astype(int)
            for j_j in range(var * feat_rem_x_run, (var + 1) * feat_rem_x_run):
                index_feat[j_j - var * feat_rem_x_run] = np.where(a_attr[j_obs, :] == sorted_a_attr[j_obs, j_j])[0][0]
            if n_base_x_feat > 1:
                for i_base in range(n_base_x_feat):
                    for j_j in range(feat_rem_x_run):
                        x_for_a_attr[0, index_feat[j_j] + i_base * n_vars] = reference[
                            index_feat[j_j] + i_base * n_vars]
            else:
                if reference.shape == a_attr.shape:
                    for j_j in range(feat_rem_x_run):
                        x_for_a_attr[0, index_feat[j_j]] = reference[j_obs, index_feat[j_j]]
                else:
                    for j_j in range(feat_rem_x_run):
                        x_for_a_attr[0, index_feat[j_j]] = reference[index_feat[j_j]]

            predictionsmodel_a_attr = model.predict(x_for_a_attr)
            if log_odds:
                out_[j_obs, var + 1] = np.log(predictionsmodel_a_attr[0][0]
                                              / (1 - predictionsmodel_a_attr[0][0]))
            else:
                out_[j_obs, var + 1] = predictionsmodel_a_attr[0][0]
            abs_out_[j_obs, var + 1] = np.abs(out_[j_obs, var + 1])
    return out_, abs_out_
