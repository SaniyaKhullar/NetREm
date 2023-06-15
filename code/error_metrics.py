from packages_needed import *
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.metrics import make_scorer

def calculate_mean_square_error(actual_values, predicted_values):
    # Please note that this function by Saniya calculates the Mean Square Error (MSE)
    
    difference = (actual_values - predicted_values)
    squared_diff = difference ** 2 # square of the difference
    mean_squared_diff = np.mean(squared_diff)
    return mean_squared_diff



def mse(REF: np.ndarray, X: np.ndarray, axis: Optional[int] = None) -> np.float:
    """Compute mean square error between array with a reference array -
    If REF or X is complex, compute mse(REF.real, X.real) + 1j * mse(REF.imag, X.imag)

    Parameters
    ----------
    REF:
        ground truth, or reference array, e.g. shape=(n_sample, n_target) for machine learning
    X:
        result array to compare with reference, e.g. shape=(n_sample, n_target) for machine learning
    axis:
        Axis along which the comparison is computed. Default to None to compute the comparison
        of the flattened array.

    Returns
    -------
    mse_:
         normalized mean square error

    Examples
    -------
        mse(REF, X, axis=0) compute the comparision along n_sample dimension for machine learning
        regression application where shape=(n_sample, n_target)
    """

    if (not np.iscomplexobj(REF)) and (not np.iscomplexobj(X)):
        return ((X - REF)**2).mean(axis=axis)
    else:
        return mse(REF.real, X.real, axis) + 1j * mse(REF.imag, X.imag, axis)


def nmse(REF: np.ndarray, X: np.ndarray, axis: Optional[int] = None) -> np.float:
    """Compute normalized mean square error between array with a reference array -
    If REF or X is complex, compute nmse(REF.real, X.real) + 1j * nmse(REF.imag, X.imag)

    Parameters
    ----------
    REF:
        ground truth, or reference array, e.g. shape=(n_sample, n_target) for machine learning
    X:
        result array to compare with reference, e.g. shape=(n_sample, n_target) for machine learning
    axis:
        Axis along which the comparison is computed. Default to None to compute the comparison
        of the flattened array.

    Returns
    -------
    nmse_:
         normalized mean square error

    Examples
    -------
        nmse(REF, X, axis=0) compute the comparision along n_sample dimension for machine learning
        regression application where shape=(n_sample, n_target)
    """
    if (not np.iscomplexobj(REF)) and (not np.iscomplexobj(X)):
        return ((X - REF)**2).mean(axis=axis) / (REF**2).mean(axis=axis)
    else:
        return nmse(REF.real, X.real, axis) + 1j * nmse(REF.imag, X.imag, axis)


def snr(REF: np.ndarray, X: np.ndarray, axis: Optional[int] = None) -> np.float64:
    """Compare an array with a reference array - compute signal to noise ration in dB.
    If REF or X is complex, compute snr(REF.real, X.real) + 1j * snr(REF.imag, X.imag)

    Parameters
    ----------
    REF:
        ground truth, or reference array, e.g. shape=(n_sample, n_target) for machine learning
    X:
        result array to compare with reference, e.g. shape=(n_sample, n_target) for machine learning
    axis:
        Axis along which the comparison is computed.  The default is to compute the comparison
        of the flattened array.

    Returns
    -------
    snr_:
        signal to noise ration in dB

    Examples
    -------
        snr(REF, X, axis=0) compute the comparision along n_sample dimension for machine learning
        regression application where shape=(n_sample, n_target)
    """
    if (not np.iscomplexobj(REF)) and (not np.iscomplexobj(X)):
        return 10 * np.log10((REF**2).mean(axis=axis) / ((X - REF)**2).mean(axis=axis))
    else:
        return snr(REF.real, X.real, axis) + 1j * snr(REF.imag, X.imag, axis)


def psnr(REF: np.ndarray, X: np.ndarray, axis: Optional[int] = None, max_: Optional[np.float64] = None) -> np.float64:
    """See snr, TODO: copy and modify docstring from snr
    """
    if (not np.iscomplexobj(REF)) and (not np.iscomplexobj(X)):
        if max_ is None:
            max_ = REF.max()
        return 10 * np.log10(max_**2 / ((X - REF)**2).mean(axis=axis))  # change from REF.max() to 255
    else:
        return psnr(REF.real, X.real, axis, max_) + 1j * psnr(REF.imag, X.imag, axis, max_)

    


def nmse_custom_score(y_true, y_pred):
    """
    Calculates the negative normalized mean squared error (MSE) between the true and predicted values.
    """
    import numpy as np
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    if not any(y_pred):  # if all predicted coefficients are 0
        return -np.inf  # return a high negative score
    nmseVal = nmse(y_true, y_pred)
    return -nmseVal



def mse_custom_score(y_true, y_pred):
    """
    Calculates the negative normalized mean squared error (MSE) between the true and predicted values.
    default: greater_is_better, so we set negative mseVal to find the smallest mse
    """
    import numpy as np
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    if not any(y_pred):  # if all predicted coefficients are 0
        return -np.inf  # return a high negative score
    mseVal = mse(y_true, y_pred)
    return -mseVal


def snr_custom_score(y_true, y_pred):
    """
    Higher the SNR the better 
    """
    import numpy as np
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    if not any(y_pred):  # if all predicted coefficients are 0
        return -np.inf  # return a high negative score
    snrVal = snr(y_true, y_pred)
    return snrVal

def psnr_custom_score(y_true, y_pred):
    """
    Higher the psnr, the better
    """
    import numpy as np
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    if not any(y_pred):  # if all predicted coefficients are 0
        return -np.inf  # return a high negative score
    psnrVal = psnr(y_true, y_pred)
    return psnrVal

# Create a custom scorer object using make_scorer
mse_custom_scorer = make_scorer(mse_custom_score)
nmse_custom_scorer = make_scorer(nmse_custom_score)
snr_custom_scorer = make_scorer(snr_custom_score)
psnr_custom_scorer = make_scorer(psnr_custom_score)


def generate_model_metrics_for_baselines_df(X_train, y_train, X_test, y_test, model_name = "ElasticNetCV", y_intercept = False, tf_name = "SOX10"):    
    from sklearn.linear_model import ElasticNetCV, LinearRegression, LassoCV, RidgeCV
    print(f"{model_name} results :) for fitting y_intercept = {y_intercept}")
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
    #print(model_df)
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
    tfs = sorted_df["TF"].tolist()
    if tf_name not in tfs:
        sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
        sorted_df.columns = ["Rank", "TF"]
    sorted_df["Info"] = model_name
    if y_intercept:
        sorted_df["y_intercept"] = "True :)"
    else:
        sorted_df["y_intercept"] = "False :("
    sorted_df["num_TFs"] = model_df.shape[1]
    predY_train = regr.predict(X_train)
    predY_test = regr.predict(X_test)
    train_mse = mse(y_train.values.flatten(), predY_train)
    test_mse = mse(y_test.values.flatten(), predY_test)
    train_nmse = nmse(y_train.values.flatten(), predY_train)
    test_nmse = nmse(y_test.values.flatten(), predY_test)
    sorted_df["train_mse"] = train_mse
    sorted_df["test_mse"] = test_mse
    sorted_df["train_nmse"] = train_nmse
    sorted_df["test_nmse"] = test_nmse
    predY_train = regr.predict(X_train)
    predY_test = regr.predict(X_test)
    sorted_df["train_nmse"]  = nmse(y_train.values.flatten(), predY_train)
    sorted_df["test_nmse"] = nmse(y_test.values.flatten(), predY_test)
    sorted_df["train_snr"] = snr(y_train.values.flatten(), predY_train)
    sorted_df["test_snr"] = snr(y_test.values.flatten(), predY_test)
    sorted_df["train_psnr"] = psnr(y_train.values.flatten(), predY_train)
    sorted_df["test_psnr"] = psnr(y_test.values.flatten(), predY_test)
    return sorted_df


def generate_model_metrics_for_netrem_model_object(netrem_model, y_intercept_fit, X_train, y_train, X_test, y_test, filtered_results = False, tf_name = "SOX10", focus_gene = "y"):
    if netrem_model.model_nonzero_coef_df.shape[1] == 1:
        sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
        sorted_df.columns = ["Rank", "TF"]
        tf_netrem_found = False
        if netrem_model.model_type == "LassoCV":
            sorted_df["Info"] = "NetREm (b = " + str(netrem_model.beta_network) + "; LassoCV)" #+ netrem_info# + str(netrem_model.optimal_alpha) + ")"
        else:
            sorted_df["Info"] = "NetREm (b = " + str(netrem_model.beta_network) + "; a = " + netrem_model.alpha_lasso + ")"# : " + netrem_info# + str(netrem_model.optimal_alpha) + ")" 

        if y_intercept_fit:
            sorted_df["y_intercept"] = "True :)"
        else:
            sorted_df["y_intercept"] = "False :("
        sorted_df["num_TFs"] = 0
    else:
        sorted_df = netrem_model.sorted_coef_df[netrem_model.sorted_coef_df["TF"] == tf_name]
        tfs = sorted_df["TF"].tolist()
        tf_netrem_found = True
        if tf_name not in tfs:
            sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
            sorted_df.columns = ["Rank", "TF"]
            tf_netrem_found = False
        sorted_df["Info"] = "NetREm (b = " + str(netrem_model.beta_network) + "; LassoCV)"# + str(netrem_model.optimal_alpha) + ")" 
        sorted_df["num_TFs"] = netrem_model.model_nonzero_coef_df.drop(columns = ["y_intercept"]).shape[1]
    predY_train = netrem_model.predict(X_train)
    predY_test = netrem_model.predict(X_test)
    sorted_df["train_mse"] = mse(y_train.values.flatten(), predY_train)
    sorted_df["test_mse"] = mse(y_test.values.flatten(), predY_test)
    sorted_df["train_nmse"]  = nmse(y_train.values.flatten(), predY_train)
    sorted_df["test_nmse"] = nmse(y_test.values.flatten(), predY_test)
    sorted_df["train_snr"] = snr(y_train.values.flatten(), predY_train)
    sorted_df["test_snr"] = snr(y_test.values.flatten(), predY_test)
    sorted_df["train_psnr"] = psnr(y_train.values.flatten(), predY_train)
    sorted_df["test_psnr"] = psnr(y_test.values.flatten(), predY_test)
    sorted_df_netrem = sorted_df
    netrem_dict = {"sorted_df_netrem":sorted_df_netrem, "tf_netrem_found":tf_netrem_found}
    return netrem_dict

    
# def metrics_for_netrem_models_versus_other_models(netrem_with_intercept, netrem_no_intercept, X_train, y_train, X_test, y_test, filtered_results = False, tf_name = "SOX10", target_gene = "y"):
#     """ :) This is similar to function metrics_for_netrem_versus_other_models() except it focuses on 2 types of NetREm models:
#     1. with y-intercept fitted
#     2. with no y-intercept fitted
#     :) Please note:
#         MSE (Mean Squared Error) and NMSE (Normalized Mean Squared Error) are both measures of the average difference between the predicted and actual values, where lower values indicate better performance.

#         PSNR (Peak Signal-to-Noise Ratio) and SNR (Signal-to-Noise Ratio) are both measures of the ratio between the maximum possible signal power and the power of the noise, where higher values indicate better performance.

#         However, the specific metrics that are most relevant to a particular machine learning problem can vary depending on the application and the specific goals of the model. So, it's important to consider the context and objectives of each project when selecting evaluation metrics.
#     """
#     focus_gene = target_gene
    
#     netrem_with_intercept_sorted_dict = generate_model_metrics_for_netrem_model_object(netrem_with_intercept, True, X_train, y_train, X_test, y_test, filtered_results, tf_name, focus_gene)
#     netrem_no_intercept_sorted_dict = generate_model_metrics_for_netrem_model_object(netrem_no_intercept, False, X_train, y_train, X_test, y_test, filtered_results, tf_name, focus_gene)        

#     netrem_with_intercept_sorted_df = netrem_with_intercept_sorted_dict["sorted_df_netrem"]
#     netrem_with_intercept_sorted_df["y_intercept"] = "True :)"
#     netrem_no_intercept_sorted_df = netrem_no_intercept_sorted_dict["sorted_df_netrem"]
#     netrem_no_intercept_sorted_df["y_intercept"] = "False :("
#     tf_netrem_found_with_intercept = netrem_with_intercept_sorted_dict["tf_netrem_found"]
#     tf_netrem_found_no_intercept = netrem_no_intercept_sorted_dict["tf_netrem_found"]
    
    
#     sorted_df_elasticcv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = False, tf_name = tf_name)
#     sorted_df_lassocv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = False, tf_name = tf_name)
#     sorted_df_ridgecv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = False, tf_name = tf_name)
#     sorted_df_linear = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = False, tf_name = tf_name)
#     sorted_df_elasticcv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = True, tf_name = tf_name)
#     sorted_df_lassocv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = True, tf_name = tf_name)
#     sorted_df_ridgecv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = True, tf_name = tf_name)
#     sorted_df_linear2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = True, tf_name = tf_name)

#     sorty_combo = pd.concat([netrem_no_intercept_sorted_df, sorted_df_elasticcv, sorted_df_ridgecv, sorted_df_lassocv, sorted_df_linear])
#     sorty_combo = pd.concat([sorty_combo, netrem_with_intercept_sorted_df, sorted_df_elasticcv2, sorted_df_ridgecv2, sorted_df_lassocv2, sorted_df_linear2])
#     sorty_combo = sorty_combo[sorty_combo["TF"] == tf_name]
#     sorty_combo["TG"] = focus_gene
#     sorty_combo = sorty_combo.reset_index().drop(columns = ["index"])
#     sorty_combo = sorty_combo[['AbsoluteVal_coefficient', 'Rank', 'TF', 'Info', 'y_intercept', 'num_TFs', 'TG', 'train_mse',
#        'test_mse', 'train_nmse', 'test_nmse', 'train_snr', 'test_snr',
#        'train_psnr', 'test_psnr']]
#     aaa = sorty_combo
#     aaa['rank_mse_train'] = aaa['train_mse'].rank(ascending=True).astype(int)
#     aaa['rank_mse_test'] = aaa['test_mse'].rank(ascending=True).astype(int)
#     aaa['rank_nmse_train'] = aaa['train_nmse'].rank(ascending=True).astype(int)
#     aaa['rank_nmse_test'] = aaa['test_nmse'].rank(ascending=True).astype(int)

#     aaa['rank_snr_train'] = aaa['train_snr'].rank(ascending=False).astype(int)
#     aaa['rank_snr_test'] = aaa['test_snr'].rank(ascending=False).astype(int)
#     aaa['rank_psnr_train'] = aaa['train_psnr'].rank(ascending=False).astype(int)
#     aaa['rank_psnr_test'] = aaa['test_psnr'].rank(ascending=False).astype(int)
#     aaa["total_metrics_rank"] = aaa['rank_mse_train'] + aaa['rank_mse_test'] + aaa['rank_nmse_train'] + aaa['rank_nmse_test'] 
#     aaa["total_metrics_rank"] +=  aaa['rank_snr_train'] + aaa['rank_snr_test'] + aaa['rank_psnr_train'] + aaa['rank_psnr_test']
#     sorty_combo = aaa
    
#     reduced_results_df = sorty_combo[sorty_combo["Rank"] != "N/A"]
#     reduced_results_df = reduced_results_df.sort_values(by = ["Rank"])
    
    
#     if tf_netrem_found_with_intercept:
#         print(netrem_with_intercept.final_corr_vs_coef_df[["info"] + [tf_name]])
#     elif tf_netrem_found_no_intercept:
#         print(netrem_no_intercept.final_corr_vs_coef_df[["info"] + [tf_name]])
#     if filtered_results:
#         return reduced_results_df
#     else:
#         return sorty_combo
def metrics_for_netrem_models_versus_other_models(netrem_with_intercept, netrem_no_intercept, X_train, y_train, X_test, y_test, filtered_results = False, tf_name = "SOX10", target_gene = "y"):
    """ :) This is similar to function metrics_for_netrem_versus_other_models() except it focuses on 2 types of NetREm models:
    1. with y-intercept fitted
    2. with no y-intercept fitted
    :) Please note:
        MSE (Mean Squared Error) and NMSE (Normalized Mean Squared Error) are both measures of the average difference between the predicted and actual values, where lower values indicate better performance.

        PSNR (Peak Signal-to-Noise Ratio) and SNR (Signal-to-Noise Ratio) are both measures of the ratio between the maximum possible signal power and the power of the noise, where higher values indicate better performance.

        However, the specific metrics that are most relevant to a particular machine learning problem can vary depending on the application and the specific goals of the model. So, it's important to consider the context and objectives of each project when selecting evaluation metrics.
    """
    focus_gene = target_gene
    netrem_intercept_bool = True
    netrem_no_intercept_bool = True
    if netrem_with_intercept is None:
        netrem_intercept_bool = False
        tf_netrem_found_with_intercept = False
    if netrem_no_intercept is None:
        netrem_no_intercept_bool = False
        tf_netrem_found_no_intercept = False
        
    if netrem_with_intercept:
        netrem_with_intercept_sorted_dict = generate_model_metrics_for_netrem_model_object(netrem_with_intercept, True, X_train, y_train, X_test, y_test, filtered_results, tf_name, focus_gene)
        netrem_with_intercept_sorted_df = netrem_with_intercept_sorted_dict["sorted_df_netrem"]
        netrem_with_intercept_sorted_df["y_intercept"] = "True :)"
        tf_netrem_found_with_intercept = netrem_with_intercept_sorted_dict["tf_netrem_found"]

    if netrem_no_intercept_bool:
        netrem_no_intercept_sorted_dict = generate_model_metrics_for_netrem_model_object(netrem_no_intercept, False, X_train, y_train, X_test, y_test, filtered_results, tf_name, focus_gene)        
        netrem_no_intercept_sorted_df = netrem_no_intercept_sorted_dict["sorted_df_netrem"]
        netrem_no_intercept_sorted_df["y_intercept"] = "False :("
        tf_netrem_found_no_intercept = netrem_no_intercept_sorted_dict["tf_netrem_found"]
    
    
    sorted_df_elasticcv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = False, tf_name = tf_name)
    sorted_df_lassocv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = False, tf_name = tf_name)
    sorted_df_ridgecv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = False, tf_name = tf_name)
    sorted_df_linear = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = False, tf_name = tf_name)
    sorted_df_elasticcv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = True, tf_name = tf_name)
    sorted_df_lassocv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = True, tf_name = tf_name)
    sorted_df_ridgecv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = True, tf_name = tf_name)
    sorted_df_linear2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = True, tf_name = tf_name)

    if netrem_no_intercept_bool:
        sorty_combo = pd.concat([netrem_no_intercept_sorted_df, sorted_df_elasticcv, sorted_df_ridgecv, sorted_df_lassocv, sorted_df_linear])
    else:
        sorty_combo = pd.concat([sorted_df_elasticcv, sorted_df_ridgecv, sorted_df_lassocv, sorted_df_linear])
    if netrem_intercept_bool:
        sorty_combo = pd.concat([sorty_combo, netrem_with_intercept_sorted_df, sorted_df_elasticcv2, sorted_df_ridgecv2, sorted_df_lassocv2, sorted_df_linear2])
    else:
        sorty_combo = pd.concat([sorty_combo, sorted_df_elasticcv2, sorted_df_ridgecv2, sorted_df_lassocv2, sorted_df_linear2])        
    sorty_combo = sorty_combo[sorty_combo["TF"] == tf_name]
    sorty_combo["TG"] = focus_gene
    sorty_combo = sorty_combo.reset_index().drop(columns = ["index"])
    if 'AbsoluteVal_coefficient' not in sorty_combo.columns.tolist():
        sorty_combo['AbsoluteVal_coefficient'] = pd.Series([float('nan')]*len(sorty_combo))

    sorty_combo = sorty_combo[['AbsoluteVal_coefficient', 'Rank', 'TF', 'Info', 'y_intercept', 'num_TFs', 'TG', 'train_mse',
       'test_mse', 'train_nmse', 'test_nmse', 'train_snr', 'test_snr',
       'train_psnr', 'test_psnr']]
    aaa = sorty_combo
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
    sorty_combo = aaa
    
    reduced_results_df = sorty_combo[sorty_combo["Rank"] != "N/A"]
    reduced_results_df = reduced_results_df.sort_values(by = ["Rank"])
    
    
    if tf_netrem_found_with_intercept:
        print(netrem_with_intercept.final_corr_vs_coef_df[["info"] + [tf_name]])
    elif tf_netrem_found_no_intercept:
        print(netrem_no_intercept.final_corr_vs_coef_df[["info"] + [tf_name]])
    if filtered_results:
        return reduced_results_df
    else:
        return sorty_combo
    
    
def metrics_for_netrem_versus_other_models(netrem_model, X_train, y_train, X_test, y_test, filtered_results = False, tf_name = "SOX10", target_gene = "y"):
    """ :) Please note:
        MSE (Mean Squared Error) and NMSE (Normalized Mean Squared Error) are both measures of the average difference between the predicted and actual values, where lower values indicate better performance.

        PSNR (Peak Signal-to-Noise Ratio) and SNR (Signal-to-Noise Ratio) are both measures of the ratio between the maximum possible signal power and the power of the noise, where higher values indicate better performance.

        However, the specific metrics that are most relevant to a particular machine learning problem can vary depending on the application and the specific goals of the model. So, it's important to consider the context and objectives of each project when selecting evaluation metrics.
    """
    focus_gene = target_gene
    if netrem_model.model_nonzero_coef_df.shape[1] == 1:
        sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
        sorted_df.columns = ["Rank", "TF"]
        tf_netrem_found = False
        sorted_df["Info"] = "NetREm (b = " + str(netrem_model.beta_network) + "; LassoCV)"# + str(netrem_model.optimal_alpha) + ")" 
        sorted_df["y_intercept"] = "False :("
        sorted_df["num_TFs"] = 0
    else:
        sorted_df = netrem_model.sorted_coef_df[netrem_model.sorted_coef_df["TF"] == tf_name]
        tfs = sorted_df["TF"].tolist()
        tf_netrem_found = True
        if tf_name not in tfs:
            sorted_df = pd.DataFrame(["N/A", tf_name]).transpose()
            sorted_df.columns = ["Rank", "TF"]
            tf_netrem_found = False
        sorted_df["Info"] = "NetREm (b = " + str(netrem_model.beta_network) + "; LassoCV)"# + str(netrem_model.optimal_alpha) + ")" 
        sorted_df["y_intercept"] = "False :("
        sorted_df["num_TFs"] = netrem_model.model_nonzero_coef_df.drop(columns = ["y_intercept"]).shape[1]
    predY_train = netrem_model.predict(X_train)
    predY_test = netrem_model.predict(X_test)
    sorted_df["train_mse"] = mse(y_train.values.flatten(), predY_train)
    sorted_df["test_mse"] = mse(y_test.values.flatten(), predY_test)
    sorted_df["train_nmse"]  = nmse(y_train.values.flatten(), predY_train)
    sorted_df["test_nmse"] = nmse(y_test.values.flatten(), predY_test)
    sorted_df["train_snr"] = snr(y_train.values.flatten(), predY_train)
    sorted_df["test_snr"] = snr(y_test.values.flatten(), predY_test)
    sorted_df["train_psnr"] = psnr(y_train.values.flatten(), predY_train)
    sorted_df["test_psnr"] = psnr(y_test.values.flatten(), predY_test)
    sorted_df_netrem = sorted_df

    sorted_df_elasticcv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = False, tf_name = tf_name)
    sorted_df_lassocv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = False, tf_name = tf_name)
    sorted_df_ridgecv = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = False, tf_name = tf_name)
    sorted_df_linear = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = False, tf_name = tf_name)
    sorted_df_elasticcv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "ElasticNetCV", y_intercept = True, tf_name = tf_name)
    sorted_df_lassocv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LassoCV", y_intercept = True, tf_name = tf_name)
    sorted_df_ridgecv2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "RidgeCV", y_intercept = True, tf_name = tf_name)
    sorted_df_linear2 = generate_model_metrics_for_baselines_df(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, model_name = "LinearRegression", y_intercept = True, tf_name = tf_name)

    sorty_combo = pd.concat([sorted_df_netrem, sorted_df_elasticcv, sorted_df_ridgecv, sorted_df_lassocv, sorted_df_linear])
    sorty_combo = pd.concat([sorty_combo, sorted_df_elasticcv2, sorted_df_ridgecv2, sorted_df_lassocv2, sorted_df_linear2])
    sorty_combo = sorty_combo[sorty_combo["TF"] == tf_name]
    sorty_combo["TG"] = focus_gene
    sorty_combo = sorty_combo.reset_index().drop(columns = ["index"])
    if 'AbsoluteVal_coefficient' not in sorty_combo.columns.tolist():
        sorty_combo['AbsoluteVal_coefficient'] = pd.Series([float('nan')]*len(sorty_combo))

    sorty_combo = sorty_combo[['AbsoluteVal_coefficient', 'Rank', 'TF', 'Info', 'y_intercept', 'num_TFs', 'TG', 'train_mse',
       'test_mse', 'train_nmse', 'test_nmse', 'train_snr', 'test_snr',
       'train_psnr', 'test_psnr']]
    
    aaa = sorty_combo
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
    sorty_combo = aaa
    
    reduced_results_df = sorty_combo[sorty_combo["Rank"] != "N/A"]
    reduced_results_df = reduced_results_df.sort_values(by = ["Rank"])
    if tf_netrem_found:
        print(netrem_model.final_corr_vs_coef_df[["info"] + [tf_name]])
    if filtered_results:
        return reduced_results_df
    else:
        return sorty_combo