import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from tqdm import tqdm
import os
import sys # https://www.dev2qa.com/how-to-run-python-script-py-file-in-jupyter-notebook-ipynb-file-and-ipython/#:~:text=How%20To%20Run%20Python%20Script%20.py%20File%20In,2.%20Invoke%20Python%20Script%20File%20From%20Ipython%20Command-Line.
import networkx as nx
import scipy
from scipy.linalg import svd as robust_svd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, ElasticNetCV, Ridge
from numpy.typing import ArrayLike
from typing import Optional, List, Tuple
from sklearn.metrics import make_scorer
import plotly.express as px
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from numpy.typing import ArrayLike
from skopt import gp_minimize, space
from scipy.sparse.linalg.interface import LinearOperator
import warnings
from sklearn.exceptions import ConvergenceWarning
printdf = lambda *args, **kwargs: print(pd.DataFrame(*args, **kwargs))
rng_seed = 2023 # random seed for reproducibility
randSeed = 123
from packages_needed import *
from error_metrics import *
from DemoDataBuilderXandY import *
from PriorGraphNetwork import *
from Netrem_model_builder import *


class BayesianObjective_Lasso:
    def __init__(self, X, y, cv_folds, model, scorer = "mse", print_network = False):
        from skopt import gp_minimize, space
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        model.view_network = print_network
        self.model = model
        self.scorer_obj = 'neg_mean_squared_error' # the default
        if scorer == "mse":
            self.scorer_obj = mse_custom_scorer
        elif scorer == "nmse":
            self.scorer_obj = nmse_custom_scorer
        elif scorer == "snr":
            self.scorer_obj = snr_custom_scorer
        elif scorer == "psnr":
            self.scorer_obj = psnr_custom_scorer

        
    def __call__(self, params):
        
        alpha_lasso, beta_network = params
        #network = PriorGraphNetwork(edge_list = edge_list)
        netrem_model = self.model
        #print(netrem_model.get_params())
        netrem_model.alpha_lasso = alpha_lasso 
        netrem_model.beta_network = beta_network
        #netrem_model.view_network = self.view_network
        score = -cross_val_score(netrem_model, self.X, self.y, cv=self.cv_folds, scoring=self.scorer_obj).mean()
        return score

def optimal_netrem_model_via_bayesian_param_tuner(netrem_model, X_train, y_train, 
                                      beta_net_min = 0.001, 
                                      beta_net_max = 10, 
                                      alpha_lasso_min = 0.0001,
                                      alpha_lasso_max = 0.1,
                                      num_grid_values = 100,
                                      gridSearchCV_folds = 5,
                                     scorer = "mse",
                                     verbose = False):
    if verbose:
        print(f":) Please note we are running Bayesian optimization (via skopt Python package) for parameter hunting for beta_network and alpha_lasso with model evaluation scorer: {scorer} :)")
        print("we use gp_minimize here for hyperparameter tuning")
        print(f":) Please note this is a start-to-finish optimizer for NetREm (Network regression embeddings reveal cell-type protein-protein interactions for gene regulation)")
    from skopt import gp_minimize, space
    model_type = netrem_model.model_type
#     param_space = [space.Real(alpha_lasso_min, alpha_lasso_max, name='alpha_lasso', prior='log-uniform'),
#            space.Real(beta_net_min, beta_net_max, name='beta_network', prior='log-uniform')]

    if model_type == "LassoCV": 
        print("please note that we can only do this for Lasso model not for LassoCV :(")
        print("Thus, we will alter the model_type to make it Lasso")
        netrem_model.model_type = "Lasso"

    param_space = [space.Real(alpha_lasso_min, alpha_lasso_max, name='alpha_lasso', prior='log-uniform'),
           space.Real(beta_net_min, beta_net_max, name='beta_network', prior='log-uniform')]
    objective = BayesianObjective_Lasso(X_train, y_train, cv_folds = gridSearchCV_folds, model = netrem_model, scorer = scorer)

    # Perform Bayesian optimization
    result = gp_minimize(objective, param_space, n_calls=num_grid_values, random_state=123)
    results_dict = {}
    optimal_model = netrem_model
    if verbose:
        print(":) ######################################################################\n")
        print(f":) Please note the optimal model based on Bayesian optimization found: ")

    bayesian_alpha = result.x[0]
    bayesian_beta = result.x[1]
    optimal_model.alpha_lasso = bayesian_alpha
    optimal_model.beta_network = bayesian_beta
    results_dict["bayesian_alpha"] = bayesian_alpha
    print(f"alpha_lasso = {bayesian_alpha} ; beta_network = {bayesian_beta}")
    if verbose:
        print(":) ######################################################################\n")
        print("Fitting the model using these optimal hyperparameters for beta_net and alpha_lasso...")
    dict_ex = optimal_model.get_params()
    optimal_model = GRegNet(**dict_ex)
    optimal_model.fit(X_train, y_train)
    print(optimal_model.get_params())
    results_dict["optimal_model"] = optimal_model
    results_dict["bayesian_beta"] = bayesian_beta
    results_dict["result"] = result
    return results_dict






def optimal_netrem_model_via_gridsearchCV_param_tuner(netrem_model, X_train, y_train, num_grid_values, num_cv_jobs = -1):
    beta_max = 0.5 * np.max(np.abs(X_train.T.dot(y_train)))
    beta_min = 0.01 * beta_max
    beta_grid = np.logspace(np.log10(beta_max), np.log10(beta_min), num=num_grid_values)
    import copy
    alpha_grid = []
    initial_gregCV = netrem_model
    original_dict = copy.deepcopy(netrem_model.get_params()) 
    original_model = NetREmModel(**netrem_model.get_params())
    initial_gregCV.model_type = "LassoCV"
    #print(initial_gregCV.get_params())
    for beta in beta_grid:
        gregCV_demo = initial_gregCV
        gregCV_demo.beta_network = beta
        gregCV_demo.fit(X_train, y_train)
        optimal_alpha = gregCV_demo.regr.alpha_
        alpha_grid.append(optimal_alpha)

    beta_alpha_grid_dict = {}
    beta_alpha_grid_dict["beta_network_vals"] = beta_grid
    beta_alpha_grid_dict["alpha_lasso_vals"] = alpha_grid #np.array(alpha_grid)
    param_grid = []
    for i in tqdm(range(0, len(beta_alpha_grid_dict["beta_network_vals"]))):
        beta_net = beta_alpha_grid_dict["beta_network_vals"][i]
        alpha_las = beta_alpha_grid_dict["alpha_lasso_vals"][i]
        param_grid.append({"alpha_lasso": [alpha_las], "beta_network": [beta_net]})
    grid_search = GridSearchCV(original_model, param_grid = param_grid, cv=gridSearchCV_folds, 
                           n_jobs = num_cv_jobs, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # Get the best hyperparameters
    best_params = grid_search.best_params_
    optimal_alpha = best_params["alpha_lasso"]
    optimal_beta = best_params["beta_network"]
    if isinstance(optimal_alpha, np.ndarray):
        optimal_alpha = optimal_alpha[0]
    if isinstance(optimal_beta, np.ndarray):
        optimal_beta = optimal_beta[0]
    print(f":) GRegNetCV found that the optimal alpha_lasso = {optimal_alpha} and optimal beta_network = {optimal_beta}")
    update_gregnet = NetREmModel(**original_dict)
    update_gregnet.beta_network = optimal_beta
    update_gregnet.alpha_lasso = optimal_alpha
    update_gregnet = NetREmModel(**update_gregnet.get_params())
    update_gregnet.fit(X_train, y_train)
    return update_gregnet




def model_comparison_metrics_for_target_gene_with_BayesianOpt_andOr_GridSearchCV_ForNetREm(gene_num, target_genes_list,
                                            X_train_all, X_test_all, y_train_all, y_test_all, 
                                             scgrnom_step2_df, tfs, expression_percentile, tf_df, 
                                             js_mini, ppi_edge_list, num_tfs_family, gene_expression_genes, tf_name = "SOX10", 
                                             beta_net_min = 0.001, 
                                          beta_net_max = 10, 
                                          alpha_lasso_min = 0.0001,
                                          alpha_lasso_max = 0.1,
                                          num_grid_values = 100,
                                          gridSearchCV_folds = 5,
                                         scorer = "mse", view_network = False, verbose = False, num_cv_jobs = -1):

    focus_gene = target_genes_list[gene_num] # here, this is tough 9, 10
    print(f"Please note that our focus gene (Target gene (TG) y) is: {focus_gene}")

    y_train = y_train_all[[focus_gene]]
    y_test = y_test_all[[focus_gene]]

    tfs_for_tg = scgrnom_step2_df[scgrnom_step2_df["TG"] == focus_gene]["TF"].tolist()
    tfs_for_tg.sort()

    tfs_for_tg = intersection(tfs_for_tg, tfs)
    len(tfs_for_tg)
    
    low_TFs_bool = False
    if len(tfs_for_tg) < 5:
        print(":( uh-oh!")
        low_TFs_bool = True
    if verbose:
        print(len(tfs_for_use))
    # adding genes from the same family to the set of TFs (based on co-binding from Step 2)
    tf_families_to_add = list(set(tf_df[tf_df["gene"].isin(tfs_for_tg)]["TF_Family"]))
    gene_expression_avg = np.mean(X_train_all, axis=0)

    expression_threshold = np.percentile(gene_expression_avg, expression_percentile)
    if verbose:
        print(f":) Please note that based on the training X data, we find that the {expression_percentile}%ile average gene expression level is: {expression_threshold}") #expression_threshold
    gene_expression_avg_df = pd.DataFrame(gene_expression_avg, columns = ["avg_expression"])
    gene_expression_avg_df["gene"] = gene_expression_avg_df.index
    genes_above_threshold_df = gene_expression_avg_df[gene_expression_avg_df["avg_expression"] >= expression_threshold]
    info_tf_family_expression_df = pd.merge(tf_df, gene_expression_avg_df, how = "inner")
    info_tf_family_expression_df = info_tf_family_expression_df.sort_values(by = ["avg_expression"], ascending = False)
    info_tf_family_expression_df = info_tf_family_expression_df.sort_values(by = ["TF_Family"])
    mini_info_tf_family_express_df = info_tf_family_expression_df[info_tf_family_expression_df["TF_Family"].isin(tf_families_to_add)]
    # sort dataframe by 'TF_Family' and 'avg_expression' in descending order
    df_sorted = mini_info_tf_family_express_df.sort_values(['TF_Family', 'avg_expression'], ascending=False)
    # select the row with the highest 'avg_expression' for each 'TF_Family'
    df_result = df_sorted.groupby('TF_Family').first().reset_index()

    ########################################################################
    df_sorty = info_tf_family_expression_df[info_tf_family_expression_df["gene"].isin(genes_above_threshold_df["gene"].tolist())]
    # sort dataframe by 'TF_Family' and 'avg_expression' in descending order
    df_sorted1 = df_sorty.sort_values(['TF_Family', 'avg_expression'], ascending=False)
    # select the top 2 rows for each 'TF_Family'
    if low_TFs_bool:
        num_to_use_TFs = num_tfs_family + 1
        df_result1 = df_sorted1.groupby('TF_Family').head(n=num_to_use_TFs).reset_index(drop=True)
    else:
        df_result1 = df_sorted1.groupby('TF_Family').head(n=num_tfs_family).reset_index(drop=True)
    if verbose:
        print(df_result1)
    tfs_to_use_list = df_result["gene"].tolist()
    tfs_to_use_list.sort()
    if verbose:
        print(f" :) tfs_to_use_list = {tfs_to_use_list}")

    tfs_for_use = list(set(tfs_to_use_list + df_result1["gene"].tolist()))
    tfs_for_use.sort()

    ##########################################################################
    js_minier = js_mini[js_mini["TF1"].isin(tfs_for_use)]
    js_minier = js_minier[js_minier["TF2"].isin(tfs_for_use)]

    # for each tf from scgrnom step 2, we add the top 3 TFs based on the cobind matrix
    tfs_added_list = []
    for i in tqdm(range(0, len(tfs_to_use_list))):
        tf_num = i#in tfs_for_tg:
        if low_TFs_bool:
            tfs_added_list += js_minier[js_minier["TF1"] == tfs_to_use_list[tf_num]].head(9)["TF2"].tolist()
        else:
            tfs_added_list += js_minier[js_minier["TF1"] == tfs_to_use_list[tf_num]].head(3)["TF2"].tolist()
    
    tfs_added_list.sort()

    
    ####################################
    if verbose:
        print(len(tfs_added_list))
        print(tfs_added_list)
    combo_tfs = list(set(tfs_to_use_list+tfs_added_list))
    if verbose:
        print(len(combo_tfs))
        print(combo_tfs)
    tf_columns = intersection(combo_tfs, gene_expression_genes)
    tf_columns = list(set(tf_columns))
    tf_columns.sort()
    if verbose:
        print(":) # of TFs: ", len(tf_columns))
        print(tf_columns)
   
    if focus_gene in tf_columns:
        tf_columns.remove(focus_gene)
    key_genes = tf_columns
    
    ######################### :) We are filtering the input PPI matrix based on the 
    # final TFs (key_genes) to help us save time: 
    filtered_ppi_edge_list = []
    for edge in ppi_edge_list:
        if edge[0] in key_genes and edge[1] in key_genes:
            filtered_ppi_edge_list.append(edge)

    if verbose:
        print(filtered_ppi_edge_list)

    X_train = X_train_all[tf_columns]
    X_test = X_test_all[tf_columns]
    if verbose:
        print("X_train dimensions: ", X_train.shape)
        print("X_test dimensions: ", X_test.shape)

    netrem_no_intercept = netrem(edge_list = filtered_ppi_edge_list, 
                                      gene_expression_nodes = key_genes,
                                         verbose = verbose,
                                     view_network = view_network)

    netrem_with_intercept = netrem(edge_list = filtered_ppi_edge_list, 
                             y_intercept = True,
                                           verbose = verbose,
                                      gene_expression_nodes = key_genes,
                                     view_network = view_network)

    model_comparison_df1 = pd.DataFrame()
    model_comparison_df2 = pd.DataFrame()
    bayes_optimizer_bool = False
    griddy_optimizer_bool = False

    #####################################################################################
    no_intercept = False
    with_intercept = False
    try:
        optimal_netrem_no_intercept = optimal_netrem_model_via_bayesian_param_tuner(netrem_no_intercept, X_train, y_train, 
                                                                                      beta_net_min, 
                                                                                      beta_net_max, 
                                                                                      alpha_lasso_min,
                                                                                      alpha_lasso_max,
                                                                                      num_grid_values,
                                                                                      gridSearchCV_folds,
                                                                                     scorer,
                                                                                     verbose)
        #optimal_netrem_no_intercept = optimal_netrem_model_via_bayesian_param_tuner(netrem_no_intercept, X_train, y_train, verbose = verbose)
        optimal_netrem_no_intercept = optimal_netrem_no_intercept["optimal_model"]
        no_intercept = True
    except:
        print(":( Bayesian optimizer is not working for no y-intercept")
        optimal_netrem_no_intercept = None

    try:
        optimal_netrem_with_intercept = optimal_netrem_model_via_bayesian_param_tuner(netrem_with_intercept, X_train, y_train, 
                                                                                      beta_net_min, 
                                                                                      beta_net_max, 
                                                                                      alpha_lasso_min,
                                                                                      alpha_lasso_max,
                                                                                      num_grid_values,
                                                                                      gridSearchCV_folds,
                                                                                     scorer,
                                                                                     verbose)
        #optimal_netrem_with_intercept = optimal_netrem_model_via_bayesian_param_tuner(netrem_with_intercept, X_train, y_train, verbose = verbose)
        optimal_netrem_with_intercept = optimal_netrem_with_intercept["optimal_model"]
        with_intercept = True

    except:
        print(":( Bayesian optimizer is not working for y-intercept")
        optimal_netrem_with_intercept = None

    if no_intercept or with_intercept:
        model_comparison_df1 = metrics_for_netrem_models_versus_other_models(netrem_with_intercept = optimal_netrem_with_intercept, netrem_no_intercept = optimal_netrem_no_intercept,
                                                      X_train = X_train, y_train = y_train,
                                                      X_test = X_test, y_test = y_test, filtered_results = False, 
                                                      tf_name = tf_name, target_gene = focus_gene)
        model_comparison_df1["approach"] = "bayes_optimizer"
        bayes_optimizer_bool = True

        #####################################################################################
    no_intercept = False
    with_intercept = False
    try:
        griddy_netrem_no_intercept = optimal_netrem_model_via_gridsearchCV_param_tuner(netrem_no_intercept, X_train, y_train, 
                                                                                  num_grid_values, num_cv_jobs)

        no_intercept = True
    except:
        print(":( gridsearchCV is not working for no y-intercept")
        griddy_netrem_no_intercept = None    

    try:    
        griddy_netrem_with_intercept = optimal_netrem_model_via_gridsearchCV_param_tuner(netrem_with_intercept, X_train, y_train, 
                                                                                  num_grid_values, num_cv_jobs)
        with_intercept = True
    except:
        print(":( gridsearchCV is not working for y-intercept")
        griddy_netrem_with_intercept = None        

    if no_intercept or with_intercept:    
        model_comparison_df2 = metrics_for_netrem_models_versus_other_models(netrem_with_intercept = griddy_netrem_with_intercept, netrem_no_intercept = griddy_netrem_no_intercept,
                                                      X_train = X_train, y_train = y_train,
                                                      X_test = X_test, y_test = y_test, filtered_results = False, 
                                                      tf_name = tf_name, target_gene = focus_gene)

        model_comparison_df2["approach"] = "gridSearchCV"
        griddy_optimizer_bool = True
    # except:
    #     print(":( gridsearchCV optimizer is not working")
    both_approaches_bool = False
    if bayes_optimizer_bool and griddy_optimizer_bool:
        combined_model_compare_df = pd.concat([model_comparison_df1, model_comparison_df2])
        both_approaches_bool = True
    elif bayes_optimizer_bool:
        combined_model_compare_df = pd.concat([model_comparison_df1])
    else:
        combined_model_compare_df = pd.concat([model_comparison_df2])

    if both_approaches_bool:
        res3 = combined_model_compare_df
        res3["combo_key"] = res3["Info"] + "_" + res3["y_intercept"] + "_" + res3["Rank"].astype(str) + "_" + res3["num_TFs"].astype(str)
        # Count the number of occurrences of each combo_key
        combo_key_counts = res3.groupby('combo_key').size()

        # Create a boolean mask for the combo_keys that appear more than once
        combo_key_mask = combo_key_counts > 1

        # Update the approach column for the combo_keys that appear more than once
        res3.loc[res3['combo_key'].isin(combo_key_counts[combo_key_mask].index), 'approach'] = 'both'
        aaa = res3

        aaa['rank_mse_train'] = aaa['train_mse'].rank(ascending=True).astype(int)
        aaa['rank_mse_test'] = aaa['test_mse'].rank(ascending=True).astype(int)
        aaa['rank_nmse_train'] = aaa['train_nmse'].rank(ascending=True).astype(int)
        aaa['rank_nmse_test'] = aaa['test_nmse'].rank(ascending=True).astype(int)

        aaa['rank_snr_train'] = aaa['train_snr'].rank(ascending=False).astype(int)
        aaa['rank_snr_test'] = aaa['test_snr'].rank(ascending=False).astype(int)
        aaa['rank_psnr_train'] = aaa['train_psnr'].rank(ascending=False).astype(int)
        aaa['rank_psnr_test'] = aaa['test_psnr'].rank(ascending=False).astype(int)
        aaa["total_metrics_rank"] = aaa['rank_mse_train'] + aaa['rank_mse_test'] + aaa['rank_nmse_train'] + aaa['rank_nmse_test'] 
        aaa["total_metrics_rank"] +=  aaa['rank_snr_train'] + aaa['rank_snr_test'] + aaa['rank_psnr_train'] + aaa['rank_psnr_test']
        aaa = aaa.drop_duplicates()
        combined_model_compare_df = aaa
        combined_model_compare_df = combined_model_compare_df.drop(columns = ["combo_key"])
    return combined_model_compare_df



def baseline_metrics_function(X_train, y_train, X_test, y_test, tg, model_name, y_intercept, verbose = False):
    from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV, RidgeCV
    if verbose:
        print(f"{model_name} results :) for fitting y_intercept = {y_intercept}")
    try:
        if model_name == "ElasticNetCV":
            regr = ElasticNetCV(cv=5, random_state=0, fit_intercept = y_intercept)
        elif model_name == "LinearRegression":
            regr = LinearRegression(fit_intercept = y_intercept)
        elif model_name == "LassoCV":
            regr = LassoCV(cv=5, fit_intercept = y_intercept)
        elif model_name == "RidgeCV":
            regr = RidgeCV(cv=5, fit_intercept = y_intercept)
        regr.fit(X_train, y_train)
        if model_name in ["RidgeCV", "LinearRegression"]:
            model_df = pd.DataFrame(regr.coef_)
        else:
            model_df = pd.DataFrame(regr.coef_).transpose()
        if verbose:
            print(model_df)
        model_df.columns = X_train.columns.tolist()
        selected_row = model_df.iloc[0]
        selected_cols = selected_row[selected_row != 0].index # Filter out the columns with value 0
        model_df = model_df[selected_cols]
        df = model_df.replace("None", np.nan).apply(pd.to_numeric, errors='coerce')
        sorted_series = df.abs().squeeze().sort_values(ascending=False)
        # convert the sorted series back to a DataFrame
        sorted_df = pd.DataFrame(sorted_series)
        # add a column for the rank
        sorted_df['Rank'] = range(1, len(sorted_df) + 1)
        sorted_df['TF'] = sorted_df.index
        sorted_df = sorted_df.rename(columns = {0:"AbsoluteVal_coefficient"})
        # tfs = sorted_df["TF"].tolist()
        # if tf_name not in tfs:
        #     sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
        #     sorted_df.columns = ["Rank", "TF"]
        sorted_df["Info"] = model_name
        if y_intercept:
            sorted_df["y_intercept"] = "True :)"
        else:
            sorted_df["y_intercept"] = "False :("
        sorted_df["final_model_TFs"] = model_df.shape[1]
        sorted_df["TFs_input_to_model"] = X_train.shape[1]
        sorted_df["original_TFs_in_X"] =  X_train.shape[1]

        predY_train = regr.predict(X_train)
        predY_test = regr.predict(X_test)
        train_mse = em.mse(y_train.values.flatten(), predY_train)
        test_mse = em.mse(y_test.values.flatten(), predY_test)
        sorted_df["train_mse"] = train_mse
        sorted_df["test_mse"] = test_mse
        sorted_df["train_nmse"]  = em.nmse(y_train.values.flatten(), predY_train)
        sorted_df["test_nmse"] = em.nmse(y_test.values.flatten(), predY_test)
        sorted_df["train_snr"] = em.snr(y_train.values.flatten(), predY_train)
        sorted_df["test_snr"] = em.snr(y_test.values.flatten(), predY_test)
        sorted_df["train_psnr"] = em.psnr(y_train.values.flatten(), predY_train)
        sorted_df["test_psnr"] = em.psnr(y_test.values.flatten(), predY_test)
        sorted_df["TG"] = tg
        sorted_df = sorted_df.reset_index().drop(columns = ["index"])
        sorted_df
    except:
        return pd.DataFrame()
    return sorted_df