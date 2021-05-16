import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Get the parents folders
import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
parent0_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent0_dir)

from explain.dataset_generator import generate
from explain.sparse_mlp import get_sparse_mlp
from explain.baseline_quantile import NeutralFairBaseline
from explain.exact_shapley_mod import compute_shapley_mod
from explain.other_baselines import maximum_distance_bs
from explain.other_baselines import p_data_bs
from explain.mlp import validate_mlp
from explain.eval import local_roar, local_analysis


def run_sim(n_mc, dict_settings_sim):
    random.seed(n_mc)
    np.random.seed(n_mc)
    n_vars = dict_settings_sim["n_vars"]
    identity = dict_settings_sim["identity"]
    n_sim = dict_settings_sim["n_sim"]
    n_out_of_sample = dict_settings_sim["n_out_of_sample"]
    x_train, x_test, y_train, y_test = generate(n_sim, n_out_of_sample, n_vars=n_vars, identity=identity)
    n_vars = x_test.shape[1]
    n_out_of_sample = x_test.shape[0]
    print("Training set: ", x_train.shape[0])
    print("Testing set: ", n_out_of_sample)
    print('n features: ', n_vars)

    # Set MLP structure
    runs = 300
    hidden_layers = range(0, 6)
    nodes_hidden_layers = range(1, 11, 1)
    # Validate
    model, conf, _ = validate_mlp(x_train, y_train, runs, hidden_layers, nodes_hidden_layers)
    # Get Predictions
    test_predictions = model.predict(x_test)
    score_roc_auc = roc_auc_score(np.array(y_test), test_predictions)
    score_ave_pre = average_precision_score(np.array(y_test), test_predictions)
    print("Best model: ", conf)
    print("ROC AUC: ", score_roc_auc)
    print("average_precision_score: ", score_ave_pre)

    # Get weights and biases and store them in a list
    ws_mlp = []
    bs_mlp = []
    for layer in model.layers:
        ws_mlp.append(layer.get_weights()[0])
        bs_mlp.append(layer.get_weights()[1])
    links = []
    for _, activation in conf:
        links.append(activation)
    # Find baselines
    nf_bas = NeutralFairBaseline()
    reference, errors_list = nf_bas.search_baseline_mlp(ws_mlp, bs_mlp, links, np.array(x_train))
    # Get Sparse Model
    model_sparse = get_sparse_mlp(ws_mlp, bs_mlp, links, reference)

    # 1# Shapley on zeros
    reference_zero = np.zeros_like(x_test)[0, :]
    a_zero = np.array(
        [compute_shapley_mod(xx, lambda x: model.predict(np.array(x)).sum(-1), baseline=reference_zero) for xx in
         x_test])
    a_zero_train = np.array(
        [compute_shapley_mod(xx, lambda x: model.predict(np.array(x)).sum(-1), baseline=reference_zero) for xx in
         x_train])

    # 2# Modified Shapley with neutrality to 0.5
    # Compute Shapley on the sparse network
    a_neutral_05 = np.array(
        [compute_shapley_mod(xx, lambda x: model_sparse.predict(np.array(x)).sum(-1), baseline=reference) for xx
         in x_test])
    a_neutral_05_train = np.array(
        [compute_shapley_mod(xx, lambda x: model_sparse.predict(np.array(x)).sum(-1), baseline=reference) for xx
         in x_train])

    # 3# Shapley on maximum distance
    a_maxdist = np.zeros_like(x_test)
    reference_maxdist = np.zeros_like(x_test)
    for obs in range(x_test.shape[0]):
        this_x = x_test[obs]
        reference_maxdist[obs] = maximum_distance_bs(x_train, this_x)
        a_maxdist[obs] = compute_shapley_mod(this_x, lambda x: model.predict(np.array(x)).sum(-1),
                                             baseline=reference_maxdist[obs])
    a_maxdist_train = np.zeros_like(x_train)
    reference_maxdist_train = np.zeros_like(x_train)
    for obs in range(x_train.shape[0]):
        this_x = x_train[obs]
        reference_maxdist_train[obs] = maximum_distance_bs(x_train, this_x)
        a_maxdist_train[obs] = compute_shapley_mod(this_x, lambda x: model.predict(np.array(x)).sum(-1),
                                                   baseline=reference_maxdist_train[obs])

    # 4# Shapley on P data (10 draws)
    a_pdata = np.zeros_like(x_test)
    a_pdata_train = np.zeros_like(x_train)
    references_pdata = p_data_bs(x_train, seed=1, n_draws=10)
    for draw in range(references_pdata.shape[0]):
        this_reference = references_pdata[draw]
        this_attr = np.array(
            [compute_shapley_mod(xx, lambda x: model.predict(np.array(x)).sum(-1), baseline=this_reference) for xx in
             x_test])
        a_pdata += this_attr
        this_attr_train = np.array(
            [compute_shapley_mod(xx, lambda x: model.predict(np.array(x)).sum(-1), baseline=this_reference) for xx in
             x_train])
        a_pdata_train += this_attr_train
    a_pdata /= references_pdata.shape[0]
    a_pdata_train /= references_pdata.shape[0]

    # Information Content
    cols = list(np.linspace(0, n_vars, n_vars + 1).astype(int).astype(str))
    # 1 # on zeros
    _, abs_log_odds_a_zero = local_analysis(model, x_test, a_zero, reference_zero, asc=False)
    df_a_zero_abs = pd.DataFrame(abs_log_odds_a_zero, columns=cols)
    # 2 # neutral
    n_base_x_feat = int(len(reference) / n_vars)
    _, abs_log_odds_a_neutral_05 = local_analysis(model_sparse, x_test, a_neutral_05, reference, asc=False,
                                                  n_base_x_feat=n_base_x_feat)
    df_a_neutral_05_abs = pd.DataFrame(abs_log_odds_a_neutral_05, columns=cols)
    # 3 # Maximum Distance
    _, abs_log_odds_a_maxdist = local_analysis(model, x_test, a_maxdist, reference_maxdist, asc=False)
    df_a_maxdist_abs = pd.DataFrame(abs_log_odds_a_maxdist, columns=cols)
    # 4 # P_data
    mean_x = np.mean(x_train, 0)
    _, abs_log_odds_a_pdata = local_analysis(model, x_test, a_pdata, mean_x, asc=False)
    df_a_pdata_abs = pd.DataFrame(abs_log_odds_a_pdata, columns=cols)
    # Print to CSV
    df_a_zero_abs.to_csv('./results/result_abs_0_mc_' + str(n_mc) + '.csv')

    df_a_neutral_05_abs.to_csv('./results/result_abs_neutral_mc_' + str(n_mc) + '.csv')

    df_a_maxdist_abs.to_csv('./results/result_abs_maxdist_mc_' + str(n_mc) + '.csv')

    df_a_pdata_abs.to_csv('./results/result_abs_pdata_mc_' + str(n_mc) + '.csv')

    # ROAR
    n_train = 30
    # 1 # on zeros
    abs_a_zero = np.abs(a_zero)
    abs_a_zero_train = np.abs(a_zero_train)
    delta_performance_a_zero = np.zeros(n_vars)
    delta_performance_a_zero[0] = score_ave_pre
    # 2 # Modified with neutrality to 0.5
    abs_a_neutral_05 = np.abs(a_neutral_05)
    abs_a_neutral_05_train = np.abs(a_neutral_05_train)
    delta_performance_a_neutral_05 = np.zeros(n_vars)
    delta_performance_a_neutral_05[0] = score_ave_pre
    # 3 # Maximum Distance
    abs_a_maxdist = np.abs(a_maxdist)
    abs_a_maxdist_train = np.abs(a_maxdist_train)
    delta_performance_a_maxdist = np.zeros(n_vars)
    delta_performance_a_maxdist[0] = score_ave_pre
    # 4 # P_data
    abs_a_pdata = np.abs(a_pdata_train)
    abs_a_pdata_train = np.abs(a_pdata_train)
    delta_performance_a_pdata = np.zeros(n_vars)
    delta_performance_a_pdata[0] = score_ave_pre
    # 5 # random - uniform
    random_imp = np.random.random(abs_a_pdata.shape)
    random_imp_train = np.random.random(abs_a_pdata_train.shape)
    delta_performance_random_imp = np.zeros(n_vars)
    delta_performance_random_imp[0] = score_ave_pre
    for j in range(n_vars - 1):
        print("Removing var: ", j)
        # Compute errors
        # 1 # on zeros
        delta_performance_a_zero[j + 1] = local_roar(x_train, x_test, y_train, y_test, abs_a_zero_train, abs_a_zero,
                                                     conf, j, n_train=n_train,
                                                     replace_with_train=[mean_x],
                                                     replace_with_test=[mean_x])
        # 2 # Modified with neutrality to 0.5
        delta_performance_a_neutral_05[j + 1] = local_roar(x_train, x_test, y_train, y_test, abs_a_neutral_05_train,
                                                           abs_a_neutral_05,
                                                           conf, j, n_train=n_train,
                                                           replace_with_train=[mean_x],
                                                           replace_with_test=[mean_x])
        # 3 # Maximum Distance
        delta_performance_a_maxdist[j + 1] = local_roar(x_train, x_test, y_train, y_test, abs_a_maxdist_train,
                                                        abs_a_maxdist,
                                                        conf, j, n_train=n_train,
                                                        replace_with_train=[mean_x],
                                                        replace_with_test=[mean_x])
        # 4 # P_data
        delta_performance_a_pdata[j + 1] = local_roar(x_train, x_test, y_train, y_test, abs_a_pdata_train, abs_a_pdata,
                                                      conf, j, n_train=n_train,
                                                      replace_with_train=[mean_x],
                                                      replace_with_test=[mean_x])
        # 5 # random - uniform
        delta_performance_random_imp[j + 1] = local_roar(x_train, x_test, y_train, y_test, random_imp_train, random_imp,
                                                         conf, j, n_train=n_train,
                                                         replace_with_train=[mean_x],
                                                         replace_with_test=[mean_x])

    # Convert results to Dataframe
    results_df = pd.DataFrame({'zero _perf': delta_performance_a_zero,  # 1 # sh on zeros
                               'neutral05 _perf': delta_performance_a_neutral_05,
                               # 2 # Modified sh with neutrality to 0.5
                               'max dist _perf': delta_performance_a_maxdist,  # 3 # max dist
                               'pdata _perf': delta_performance_a_pdata,  # 4 # Modified sh with neutrality close to 0
                               'random _perf': delta_performance_random_imp  # 5 # Random
                               })
    # Print to CSV
    results_df.to_csv('./results/results_ROAR_mc_' + str(n_mc) + '.csv')
