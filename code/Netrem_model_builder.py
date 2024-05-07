# February 22, 2024
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from tqdm import tqdm
import os
import sys # https://www.dev2qa.com/how-to-run-python-script-py-file-in-jupyter-notebook-ipynb-file-and-ipython/#:~:text=How%20To%20Run%20Python%20Script%20.py%20File%20In,2.%20Invoke%20Python%20Script%20File%20From%20Ipython%20Command-Line.
import networkx as nx
import scipy
import math
import shap
from pecanpy.graph import SparseGraph, DenseGraph # https://pecanpy.readthedocs.io/en/latest/pecanpy.html#pecanpy.cli.main
from pecanpy import pecanpy as node2vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import svd as robust_svd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model, preprocessing # 9/19
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
printdf = lambda *args, **kwargs: print(pd.DataFrame(*args, **kwargs))
rng_seed = 2023 # random seed for reproducibility
randSeed = 123
# from packages_needed import *
import essential_functions as ef
import error_metrics as em # why to do import
import DemoDataBuilderXandY as demo
import PriorGraphNetwork as graph
import netrem_evaluation_functions as nm_eval
from tqdm.auto import tqdm
from pecanpy import pecanpy as node2vec
from pecanpy.graph import SparseGraph, DenseGraph # https://pecanpy.readthedocs.io/en/latest/pecanpy.html#pecanpy.cli.main
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import subprocess
#import dask.dataframe as dd

os.environ['PYTHONHASHSEED'] = '0'

"""
Optimization for
(1 / (2 * M)) * ||y - Xc||^2_2  +  (beta / (2 * N^2)) * c'Ac  +  alpha * ||c||_1
Which is converted to lasso
(1 / (2 * M)) * ||y_tilde - X_tilde @ c||^2_2   +  alpha * ||c||_1
where M = n_samples and N is the dimension of c.
Check compute_X_tilde_y_tilde() to see how we make sure above normalization is applied using Lasso of sklearn
"""

class NetREmModel(BaseEstimator, RegressorMixin):
    """ :) Please note that this class focuses on building a Gene Regulatory Network (GRN) from gene expression data for Transcription Factors (TFs), gene expression data for the target gene (TG), and a prior biological network (W). This class performs Network-penalized regression :) """
    _parameter_constraints = {
        "alpha_lasso": (0, None),
        "beta_net": (0, None),
        "num_cv_folds": (0, None),
        "y_intercept": [False, True],
        "use_network": [True, False],
        "max_lasso_iterations": (1, None),
        "model_type": ["Lasso", "LassoCV", "Linear"],
        "tolerance": (0, None),
        "num_jobs": (1, 1e10),
        "lasso_selection": ["cyclic", "random"],
        "lassocv_eps": (0, None),
        "lassocv_n_alphas": (1, None),
        "standardize_X": [True, False],
        "standardize_y": [True, False],
        "center_y": [True, False]
    }
    
    os.environ['PYTHONHASHSEED'] = '0'
    def __init__(self,  **kwargs):
        
        self.info = "NetREm Model"
        self.verbose = False
        self.overlapped_nodes_only = False # restrict the nodes to only being those found in the network? overlapped_nodes_only
        self.num_cv_folds = 5 # for cross-validation models
        self.num_jobs = -1 # for LassoCV or LinearRegression (here, -1 is the max possible for CPU)
        self.all_pos_coefs = False # for coefficients
        self.model_type = "Lasso"
        self.standardize_X = True
        self.standardize_y = True
        self.center_y = False
        self.use_network = True
        self.y_intercept = False
        self.max_lasso_iterations = 10000
        self.view_network = False
        self.model_info = "unfitted_model :("
        self.target_gene_y = "Unknown :("
        self.tolerance = 1e-4
        self.lasso_selection = "cyclic" # default in sklearn
        self.lassocv_eps = 1e-3 # default in sklearn        
        self.lassocv_n_alphas = 100 # default in sklearn        
        self.lassocv_alphas = None # default in sklearn 
        self.beta_net = kwargs.get('beta_net', 1)
        self.small_sparse_graph = True,
        self.dimensions: int = 128
        self.walk_length: int = 10
        self.num_walks: int = 10
        self.p: float = 1
        self.q: float = 1 #0.5
        self.workers: int = -1
        self.epochs: int = 1
        ########################################
        
        self.__dict__.update(kwargs)
        required_keys = ["network", "beta_net"]#, "gamma_net"] 
        if self.model_type == "Lasso":
            self.alpha_lasso = kwargs.get('alpha_lasso', 0.01)
            self.optimal_alpha = "User-specified optimal alpha lasso: " + str(self.alpha_lasso)
            required_keys += ["alpha_lasso"]
        elif self.model_type == "LassoCV": 
            self.alpha_lasso = "LassoCV finds optimal alpha"
            self.optimal_alpha = "Since LassoCV is model_type, please fit model using X and y data to find optimal_alpha."
        else:  # linear regression
            self.alpha_lasso = 0#"No alpha needed"
            self.optimal_alpha = 0#"No alpha needed" #
        missing_keys = [key for key in required_keys if key not in self.__dict__] # check that all required keys are present:
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        if self.use_network:
            prior_network = self.network
            self.prior_network = prior_network
            edge_list = self.network.W_df# Feb 12, 2024
            
            edge_list = (
                edge_list
                .reset_index()
                .melt(id_vars=["index"], var_name="TF2", value_name="weight_W")
                .rename(columns={"index": "TF1"})
                .query("TF1 != TF2")
            )
            self.edge_list = edge_list.values.tolist()
            #########################################################################
            self.preprocessed_network = prior_network.preprocessed_network
            self.network_nodes_list = prior_network.final_nodes # tf_names_list
            self.default_edge_weight = prior_network.default_edge_weight
        self.kwargs = kwargs
        self._apply_parameter_constraints() # ensuring that the parameter constraints are met   
   

    def __repr__(self):
        args = [f"{k}={v}" for k, v in self.__dict__.items() if k != 'param_grid' and k in self.kwargs]
        return f"{self.__class__.__name__}({', '.join(args)})"    
    
   
    def check_overlaps_work(self):
        #final_set = self.final_nodes_set
        #network_set = self.ppi_net_nodes # set(self.network_nodes_list)
        if self.tg_is_tf:
            return False
        if self.tg_name in self.final_nodes:
            return False
        return self.ppi_net_nodes != self.final_nodes_set   
    
    
    def standardize_X_data(self, X_df): # if the user opts to 
        """ :) If the user opts to standardize the X data (so that predictors have a mean of 0 
        and a standard deviation of 1), then this method will be run, which uses the preprocessing
        package StandardScalar() functionality. """
        if self.standardize_X:
            # if is_standardized(X_df):
            #     return X_df
            # Transform both the training and test data
            X_scaled = self.scaler_X.transform(X_df)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)
            return X_scaled_df
        else:
            return X_df
        
        
    def standardize_y_data(self, y_df): # if the user opts to 
        """ :) If the user opts to standardize the y data (so that the TG will have a mean of 0 
        and a standard deviation of 1), then this method will be run, which uses the preprocessing
        package StandardScalar() functionality. """
        if self.standardize_y:
            # Transform both the training and test data
            y_scaled = self.scaler_y.transform(y_df)
            y_scaled_df = pd.DataFrame(y_scaled, columns=y_df.columns)
            return y_scaled_df
        else:
            return y_df
        
        
    def center_y_data(self, y_df): # if the user opts to
        """ :) If the user opts to center the response y data:
        subtracting its mean from each observation."""
        if self.center_y:
            # Center the response
            y_train_centered = y_df - self.mean_y_train
            return y_train_centered
        else:
            return y_df
    
    
    def updating_network_and_X_during_fitting(self, X, y):   
        # updated one :)
        """ Update the prior network information and the 
        X input data (training) during the fitting of the model. It determines if the common predictors
        should be used (based on if overlapped_nodes_only is True) or if all of the X input data should be used. """
        X_df = X.sort_index(axis=1)  # sorting the X dataframe by columns. (rows are samples)
        self.target_gene_y = y.columns[0]
        tg_name = self.tg_name
        if self.standardize_X: # we will standardize X then
            if self.verbose:
                print(":) Standardizing the X data")
            self.old_X_df = X_df
            self.scaler_X = preprocessing.StandardScaler().fit(X_df) # Fit the scaler to the training data only
            # this self.scalar will be utilized for the testing data to prevent data leakage and to ensure generalization :)
            self.X_df = self.standardize_X_data(X_df)
            X = self.X_df # overwriting and updating the X df
        else:
            self.X_df = X_df
            X = self.X_df
            
        if self.center_y: # we will center y then
            self.mean_y_train = np.mean(y) # the average y value
            if self.verbose:
                print(":) centering the y data")
            # Assuming y_train and y_test are your training and test labels
            self.old_y = y
            y = self.center_y_data(y)

        if self.standardize_y: # we will standardize y then
            if self.verbose:
                print(":) Standardizing the y data")
            self.old_y_df = y
            self.scaler_y = preprocessing.StandardScaler().fit(y) # Fit the scaler to the training data only
            # this self.scalar will be utilized for the testing data to prevent data leakage and to ensure generalization :)
            self.y_df = self.standardize_y_data(y)
            y = self.y_df # overwriting and updating the y df
        else:
            self.y_df = y 
        if self.tg_name in X_df.columns.tolist():
            X_df.drop(columns = [tg_name], inplace = True)
            self.tg_is_tf = True  # 1/31/24
        gene_expression_nodes = X_df.columns.tolist() #sorted(X_df.columns.tolist())  # these will be sorted
        self.ppi_net_nodes = set(self.network_nodes_list)
        ppi_net_nodes = self.ppi_net_nodes
        common_nodes = list(ppi_net_nodes.intersection(gene_expression_nodes))

        if not common_nodes:  # may be possible that the X dataframe needs to be transposed if provided incorrectly
            print("Please note: we are flipping X dataframe around so that the rows are samples and the columns are gene/TF names :)")
            X_df = X_df.transpose()
            gene_expression_nodes = sorted(X_df.columns.tolist())
            common_nodes = list(ppi_net_nodes.intersection(gene_expression_nodes))

        self.gene_expression_nodes = gene_expression_nodes
        self.common_nodes = sorted(common_nodes)
        self.final_nodes = gene_expression_nodes
        if self.overlapped_nodes_only:
            self.final_nodes = self.common_nodes
        elif self.preprocessed_network:
            self.final_nodes = sorted(self.prior_network.final_nodes)
        else:
            self.final_nodes = gene_expression_nodes
        self.final_nodes_set = set(self.final_nodes)
        final_nodes_set = self.final_nodes_set
        ppi_nodes_to_remove = list(ppi_net_nodes - final_nodes_set)
        self.ppi_nodes_to_remove = ppi_nodes_to_remove
        if self.tg_name in self.final_nodes: 
            self.tg_is_tf = True
            filter_network_bool = True
            self.filter_network_bool = filter_network_bool
            self.final_nodes.remove(self.tg_name)     
            ppi_nodes_to_remove = list(set(ppi_nodes_to_remove).union(set(self.tg_name)))
        self.gexpr_nodes_added = list(set(gene_expression_nodes) - final_nodes_set)
        self.gexpr_nodes_to_add_for_net = list(set(gene_expression_nodes) - set(common_nodes))

        if self.verbose:
            if ppi_nodes_to_remove:
                print(f"Please note that we remove {len(ppi_nodes_to_remove)} nodes found in the input network that are not found in the input gene expression data (X) :)")
                print(ppi_nodes_to_remove)
            else:
                print(f":) Please note that all {len(common_nodes)} nodes found in the network are also found in the input gene expression data (X) :)")           
        self.filter_network_bool = self.check_overlaps_work()
        
        if self.tg_is_tf:
            self.filter_network_bool = True
        if self.filter_network_bool:
            print("Please note that we need to update the network information")
            self.updating_network_A_matrix_given_X()  # updating the A matrix given the gene expression data X
            if self.view_network:
                ef.draw_arrow()
                self.view_W_network = self.view_W_network()
        else:
            self.A_df = self.network.A_df
            self.A = self.network.A
            self.W_df = self.network.W_df
            self.final_input_W_df = self.W_df 
            self.nodes = self.A_df.columns.tolist()
        self.network_info = "fitted_network"
        self.M = y.shape[0]       
        self.N = len(self.final_nodes)  # pre-processing:
        self.X_train = self.preprocess_X_df(X) # dataframe to array
        self.y_train = self.preprocess_y_df(y) # dataframe to array
        return self
    
    def organize_B_interaction_list(self): # TF-TF interactions to output :)
        final_tfs = self.model_nonzero_coef_df
        final_tfs = final_tfs.drop(columns = ["y_intercept"]).columns.tolist()
        if len(final_tfs) == 0:
            self.coord_score_df = pd.DataFrame()
        else:
            X_tilda_train_df = self.X_tilda_train_df
            c_df = self.model_coef_df
            c_df = c_df.drop(columns = ["y_intercept"])
            coeff_vector = c_df.iloc[0].values
            cos_sim = cosine_similarity(X_tilda_train_df.T)  # Transpose DataFrame to calculate column-wise similarity
            cos_sim_df = pd.DataFrame(cos_sim, index = c_df.columns, columns = c_df.columns)
            coeff_matrix = np.outer(coeff_vector, coeff_vector)
            sign_matrix = np.sign(coeff_matrix).astype(int)
            coord_matrix = abs(cos_sim_df) * sign_matrix
            result = coord_matrix.loc[final_tfs, final_tfs]
            np.fill_diagonal(result.values, 0)
            max_other = np.max(np.abs(result)).max()
            coord_matrix = 100.0*result/max_other
            self.coord_score_df = coord_matrix
            self.TF_interaction_df = self.coord_score_df
            self.B_interaction_df = self.coord_score_df
        return self   
    
    
    def fit(self, X, y): # fits a model Function used for model training 
        tg_name = y.columns.tolist()[0]
        self.tg_is_tf = False
        if tg_name in X.columns.tolist():
            if self.verbose:
                print(f":) dropping TG {tg_name} from list of TF predictors!")
            X.drop(columns = [tg_name], inplace = True)
            self.tg_is_tf = True  # 1/31/24
        self.tg_name = tg_name
        self.updating_network_and_X_during_fitting(X, y)
        self.E_train = self.compute_E_matrix(self.X_train)
        self.X_tilda_train, self.y_tilda_train = self.compute_X_tilde_y_tilde(self.E_train, self.X_train, 
                                                                              self.y_train)
        self.standardize_X_tilde_y_tilde()
        # learning latent embedding values for X and y, respectively.
        self.X_training_to_use, self.y_training_to_use = self.X_tilda_train, self.y_tilda_train
        self.regr = self.return_fit_ml_model(self.X_training_to_use, self.y_training_to_use)
        ml_model = self.regr
        self.final_alpha = self.alpha_lasso
        if self.model_type == "LassoCV":
            self.final_alpha = ml_model.alpha_
            self.optimal_alpha = "Cross-Validation optimal alpha lasso: " + str(self.final_alpha)
        if self.verbose:
            print(self.optimal_alpha)
        self.coef = ml_model.coef_ # Please Get the coefficients
        self.coef[self.coef == -0.0] = 0
        if self.y_intercept:
            self.intercept = ml_model.intercept_
        self.predY_train = ml_model.predict(self.X_train) # training data  
        # training metrics:
        self.mse_train = self.calculate_mean_square_error(self.y_train, self.predY_train) # Calculate MSE    
        self.nmse_train = self.calculate_nmse(self.y_train, self.predY_train) # Calculate NMSE      
        self.snr_train = self.calculate_snr(self.y_train, self.predY_train) # Calculate SNR (Signal to Noise Ratio)      
        self.psnr_train = self.calculate_psnr(self.y_train, self.predY_train) # Calculate PSNR (Peak Signal to Noise Ratio)
        if self.y_intercept:
            coeff_terms = [self.intercept] + list(self.coef)
            index_names = ["y_intercept"] + self.nodes 
            self.model_coef_df = pd.DataFrame(coeff_terms, index = index_names).transpose()
        else:
            coeff_terms = ["None"] + list(self.coef)
            index_names = ["y_intercept"] + self.nodes
            self.model_coef_df = pd.DataFrame(coeff_terms, index = index_names).transpose()
        self.model_info = "fitted_model :)"
        selected_row = self.model_coef_df.iloc[0]
        selected_cols = selected_row[selected_row != 0].index # Filter out the columns with value 0
        if len(selected_cols) == 0:
            self.model_nonzero_coef_df = None
            self.num_final_predictors = 0
            self.final_corr_vs_coef_df = pd.DataFrame()
            self.combined_df = pd.DataFrame()
        else:
            self.model_nonzero_coef_df = self.model_coef_df[selected_cols]    
            if len(selected_cols) > 1: # and self.model_type != "Linear":
                self.netrem_model_predictor_results(y)
                self.num_final_predictors = len(selected_cols)
                if "y_intercept" in selected_cols:
                    self.num_final_predictors = self.num_final_predictors - 1
            else:
                self.final_corr_vs_coef_df = pd.DataFrame()
                self.combined_df = pd.DataFrame()
                
            self.organize_B_interaction_list()
        self.TF_coord_scores_pairwise_df = return_TF_coord_scores_df(self)
        return self
    
    
    def netrem_model_predictor_results(self, y): # olders
        """ :) Please note that this function by Saniya works on a netrem model and returns information about the predictors
        such as their Pearson correlations with y, their rankings as well.
        It returns: sorted_df, final_corr_vs_coef_df, combined_df """      
        abs_df = self.model_nonzero_coef_df.replace("None", np.nan).apply(pd.to_numeric, errors='coerce').abs()
        if abs_df.shape[0] == 1:
            abs_df = pd.DataFrame([abs_df.squeeze()])

        sorted_series = abs_df.squeeze().sort_values(ascending=False)
        sorted_df = pd.DataFrame(sorted_series) # convert the sorted series back to a DataFrame
        sorted_df['Rank'] = range(1, len(sorted_df) + 1) # add a column for the rank
        sorted_df['TF'] = sorted_df.index
        sorted_df = sorted_df.rename(columns = {0:"AbsoluteVal_coefficient"})
        self.sorted_coef_df = sorted_df # print the sorted DataFrame
        tg = self.tg_name
        corr = pd.DataFrame(self.X_df.corrwith(y[tg])).transpose()
        corr["info"] = "corr (r) with y: " + tg
        all_df = self.model_coef_df
        all_df = all_df.iloc[:, 1:]
        all_df["info"] = "network regression coeff. with y: " + tg
        all_df = pd.concat([all_df, corr])
        all_df["input_data"] = "X_train"
        sorting = self.sorted_coef_df[["Rank"]].transpose().drop(columns = ["y_intercept"])
        sorting = sorting.reset_index().drop(columns = ["index"])
        sorting["info"] = "Absolute Value NetREm Coefficient Ranking"
        sorting["input_data"] = "X_train"
        all_df = pd.concat([all_df, sorting])
        self.corr_vs_coef_df = all_df
        netrem_model_df = self.model_nonzero_coef_df.transpose()
        netrem_model_df.columns = ["coef"]
        netrem_model_df["TF"] = netrem_model_df.index.tolist()
        netrem_model_df["TG"] = tg
        self.final_corr_vs_coef_df = self.corr_vs_coef_df[["info", "input_data"] + self.model_nonzero_coef_df.columns.tolist()[1:]]
        if self.y_intercept:
            netrem_model_df["info"] = "netrem_with_intercept"
        else:
            netrem_model_df["info"] = "netrem_no_intercept"
        netrem_model_df["train_mse"] = self.mse_train
        netrem_model_df["train_nmse"] = self.nmse_train
        netrem_model_df["train_snr"] = self.snr_train
        netrem_model_df["train_psnr"] = self.psnr_train
        if self.model_type != "Linear":
            netrem_model_df["beta_net"] = self.beta_net
            if self.model_type == "LassoCV":
                netrem_model_df["alpha_lassoCV"] = self.optimal_alpha
            else:
                netrem_model_df["alpha_lasso"] = self.alpha_lasso
        if netrem_model_df.shape[0] > 1:
            self.combined_df = pd.merge(netrem_model_df, self.sorted_coef_df)
            self.combined_df["final_model_TFs"] = max(self.sorted_coef_df["Rank"]) - 1
        else:
            self.combined_df = netrem_model_df
        self.combined_df["TFs_input_to_model"] = len(self.final_nodes)
        self.combined_df["original_TFs_in_X"] = len(self.gene_expression_nodes)
        self.combined_df["standardized_X"] = self.standardize_X
        self.combined_df["standardized_y"] = self.standardize_y
        self.combined_df["centered_y"] = self.center_y
        return self


    def view_W_network(self):
        roundedW = np.round(self.W, decimals=4)
        wMat = ef.view_matrix_as_dataframe(roundedW, column_names_list=self.final_nodes, row_names_list=self.final_nodes)
        w_edgeList = wMat.stack().reset_index()
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns={"level_0": "source", "level_1": "target", 0: "weight"})
        w_edgeList = w_edgeList[w_edgeList["weight"] != 0]
        G = nx.from_pandas_edgelist(w_edgeList, source="source", target="target", edge_attr="weight")
        pos = nx.spring_layout(G)
        weights_list = [G.edges[e]['weight'] * self.prior_network.edge_weight_scaling for e in G.edges]
        fig, ax = plt.subplots()
        if not self.overlapped_nodes_only:
            nodes_to_add = list(set(self.gene_expression_nodes) - set(self.common_nodes))
            if nodes_to_add:
                print(f":) {len(nodes_to_add)} new nodes added to network based on gene expression data {nodes_to_add}")
                node_color_map = {
                    node: self.prior_network.added_node_color_name if node in nodes_to_add else self.prior_network.node_color_name 
                    for node in G.nodes
                }
                nx.draw(G, pos, node_color=node_color_map.values(), edge_color=self.prior_network.edge_color_name, with_labels=True, width=weights_list, ax=ax)
            else:
                nx.draw(G, pos, node_color=self.prior_network.node_color_name, edge_color=self.prior_network.edge_color_name, with_labels=True, width=weights_list, ax=ax)
        else:
            nx.draw(G, pos, node_color=self.prior_network.node_color_name, edge_color=self.prior_network.edge_color_name, with_labels=True, width=weights_list, ax=ax)
        labels = {e: G.edges[e]['weight'] for e in G.edges}
        return nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)       
    
    
    def compute_E_matrix(self, X):
        """ M is N_sample, because ||y - Xc||^2 need to be normalized by 1/n_sample, but not the 2 * beta_L2 * c'Ac term
        see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1 where M = n_sample 
        Calculations"""
        XtX = X.T @ X
        beta_L2 = self.beta_net
        part_1 = XtX/self.M
        part_2 = float(beta_L2) * self.A 
        B = part_1 + part_2
        self.B_df = pd.DataFrame(B, index = self.final_nodes, columns = self.final_nodes) # please fix so it is self.E_df
        self.E_part_XtX = pd.DataFrame(part_1, index = self.final_nodes, columns = self.final_nodes)
        self.E_part_netReg = pd.DataFrame(part_2, index = self.final_nodes, columns = self.final_nodes)
        return B
    
    
    def compute_X_tilde_y_tilde(self, B, X, y):
        """Compute X_tilde, y_tilde such that X_tilde.T @ X_tilde = B,   y_tilde.T @ X_tilde = y.T @ X """
        U, s, _Vh = np.linalg.svd(B, hermitian=True)  # B = U @ np.diag(s) @ _Vh
        self.U = U
        self.s = s
        self.B = B
        if (cond := s[0] / s[-1]) > 1e10:
            print(f'Large conditional number of B matrix: {cond: .2f}')
        S_sqrt = ef.DiagonalLinearOperator(np.sqrt(s))
        S_inv_sqrt = ef.DiagonalLinearOperator(1 / np.sqrt(s))
        X_tilde = S_sqrt @ U.T
        svd_problem_bool = np.isnan(U.T).any() # we may have problems here 
        if svd_problem_bool:
            B_df = self.B_df
            updated_B_df = B_df.dropna(how='all')
            B = updated_B_df.values
            U, s, _Vh = np.linalg.svd(B, hermitian=True)  # B = U @ np.diag(s) @ _Vh
            if (cond := s[0] / s[-1]) > 1e10:
                print(f'Large conditional number of B matrix: {cond: .2f}')
            S_sqrt = ef.DiagonalLinearOperator(np.sqrt(s))
            S_inv_sqrt = ef.DiagonalLinearOperator(1 / np.sqrt(s))
            X_tilde = S_sqrt @ U.T
            self.revised_B_train = B
        y_tilde = (y @ X @ U @ S_inv_sqrt).T
        # assert(np.allclose(y.T @ X, y_tilde.T @ X_tilde))
        # assert(np.allclose(B, X_tilde.T @ X_tilde))
        # scale: we normalize by 1/M, but sklearn.linear_model.Lasso normalize by 1/N because X_tilde is N*N matrix,
        # so Lasso thinks the number of sample is N instead of M, to use lasso solve our desired problem, correct the scale
        scale = np.sqrt(self.N)/ self.M
        X_tilde *= np.sqrt(self.N)
        y_tilde *= scale
        return X_tilde, y_tilde
    
    
    def standardize_X_tilde_y_tilde(self):
        """Compute X_tilde, y_tilde such that X_tilde.T @ X_tilde = B,   y_tilde.T @ X_tilde = y.T @ X """
        self.X_tilda_train_df = pd.DataFrame(self.X_tilda_train, index = self.final_nodes, columns = self.final_nodes)
        scaler = StandardScaler()
        self.X_tilda_train_standardized_df = pd.DataFrame(scaler.fit_transform(self.X_tilda_train), 
                                                          columns=self.final_nodes, index = self.final_nodes)
        scaler = StandardScaler()
        # Assuming y_tilda_train is your 1D array
        y_tilda_train_reshaped = self.y_tilda_train.reshape(-1, 1)
        # Then you use the reshaped array with StandardScaler
        self.y_tilda_train_standardized_df = pd.DataFrame(scaler.fit_transform(y_tilda_train_reshaped))
        self.standardized_X_tilda_train = self.X_tilda_train_standardized_df.values
        self.standardized_y_tilda_train = self.y_tilda_train_standardized_df.T.values[0]
    
    
    def _apply_parameter_constraints(self):
        constraints = {**NetREmModel._parameter_constraints}
        for key, value in self.__dict__.items():
            if key in constraints:
                if isinstance(constraints[key], tuple):
                    if isinstance(constraints[key][0], type) and not isinstance(value, constraints[key][0]):
                        setattr(self, key, constraints[key][0])
                    elif constraints[key][1] is not None and isinstance(constraints[key][1], type) and not isinstance(value, constraints[key][1]):
                        setattr(self, key, constraints[key][1])
                elif value not in constraints[key]:
                    setattr(self, key, constraints[key][0])
        return self
        
        
    def calculate_mean_square_error(self, actual_values, predicted_values):
        difference = (actual_values - predicted_values)# Please note that this function by Saniya calculates the Mean Square Error (MSE)
        squared_diff = difference ** 2 # square of the difference
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff
    
    
    def predict(self, X_test):
        X_test = X_test[self.final_nodes] # Oct 28
        if self.standardize_X:
            self.X_test_standardized = self.standardize_X_data(X_test)
            X_test = self.preprocess_X_df(self.X_test_standardized)
        else:
            X_test = self.preprocess_X_df(X_test) # X_test
        return self.regr.predict(X_test)

    
    def test_mse(self, X_test, y_test):
        X_test = X_test.sort_index(axis=1) # 9/20
        if self.tg_is_tf: # 3/30/24
            X_test = X_test.drop(columns = [self.tg_name])
        
        if self.standardize_X:
            if is_standardized(X_test) == False:
                self.X_test_standardized = self.standardize_X_data(X_test)
            else:
                self.X_test_standardized = X_test
            X_test = self.preprocess_X_df(self.X_test_standardized)
        else:
            X_test = self.preprocess_X_df(X_test) # X_test
        if self.center_y:
            y_test = self.center_y_data(y_test)
        if self.standardize_y:
            if is_standardized(y_test) == False:
                self.y_test_standardized = self.standardize_y_data(y_test)
            else:
                self.y_test_standardized = y_test
            y_test = self.preprocess_y_df(self.y_test_standardized)
        else:
            y_test = self.preprocess_y_df(y_test) # X_test
            
        predY_test = self.regr.predict(X_test) # training data   
        mse_test = self.calculate_mean_square_error(y_test, predY_test) # Calculate MSE
        return mse_test 
    
 ## October 28: :)
    def calculate_nmse(self, actual_values, predicted_values):#(self, X_test, y_test):  
        nmse_test = em.nmse(actual_values, predicted_values) #(y_test, predY_test) # Calculate MSE
        return nmse_test

    
    def calculate_snr(self, actual_values, predicted_values):#(self, X_test, y_test): 
        snr_test = em.snr(actual_values, predicted_values) #(y_test, predY_test) # Calculate MSE
        return snr_test	
    
    
    def calculate_psnr(self, actual_values, predicted_values):#(self, X_test, y_test):
        psnr_test = em.psnr(actual_values, predicted_values) #(y_test, predY_test) # Calculate MSE
        return psnr_test	
## end of Oct 28

    def get_params(self, deep=True):
        params_dict = {"info":self.info, "alpha_lasso": self.alpha_lasso, "beta_net": self.beta_net, 
                "y_intercept": self.y_intercept, "model_type":self.model_type, 
                       "standardize_X":self.standardize_X,
                       "center_y":self.center_y,
                      "max_lasso_iterations":self.max_lasso_iterations, 
                       "network":self.network, "verbose":self.verbose,
                      "all_pos_coefs":self.all_pos_coefs, "model_info":self.model_info,
                      "target_gene_y":self.target_gene_y}
        if self.model_type == "LassoCV":
            params_dict["num_cv_folds"] = self.num_cv_folds
            params_dict["num_jobs"] = self.num_jobs
            params_dict["alpha_lasso"] = "LassoCV finds optimal alpha"
            params_dict["lassocv_eps"] = self.lassocv_eps
            params_dict["lassocv_n_alphas"] = self.lassocv_n_alphas
            params_dict["lassocv_alphas"] = self.lassocv_alphas
            params_dict["optimal_alpha"] = self.optimal_alpha 
        elif self.model_type == "Linear":
            params_dict["alpha_lasso"] = 0 #"No alpha needed"
            params_dict["num_jobs"] = self.num_jobs
        if self.model_type != "Linear":
            params_dict["tolerance"] = self.tolerance
            params_dict["lasso_selection"] = self.lasso_selection
        if not deep:
            return params_dict
        else:
            return copy.deepcopy(params_dict)
             
            
    def set_params(self, **params):
        """ Sets the value of any parameters in this estimator
        Parameters: **params: Dictionary of parameter names mapped to their values
        Returns: self: Returns an instance of self """
        if not params:
            return self
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f'Invalid parameter {key} for estimator {self.__class__.__name__}')
            setattr(self, key, value)
        return self   
    
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        result.optimal_alpha = self.optimal_alpha
        return result

    
    def clone(self):
        return deepcopy(self)


    def score(self, X, y, zero_coef_penalty=10):
        if isinstance(X, pd.DataFrame):
            X = self.preprocess_X_df(X)  # X_test
        if isinstance(y, pd.DataFrame):
            y = self.preprocess_y_df(y)

        # Make predictions using the predict method of your custom estimator
        y_pred = self.predict(X)
        # Handle cases where predictions are exactly zero
        y_pred[y_pred == 0] = 1e-10
        # Calculate the normalized mean squared error between the true and predicted values
        nmse_ = (y - y_pred)**2
        nmse_[y_pred == 1e-10] *= zero_coef_penalty
        nmse_ = nmse_.mean() / (y**2).mean()
        if nmse_ == 0:
            return float("inf")  # Return positive infinity if nmse_ is zero
        else:
            return -nmse_
        
        
    def updating_network_A_matrix_given_X(self) -> np.ndarray:
        """ When we call the fit method, this function is used to help us update the network information.
        Here, we can generate updated W matrix, updated D matrix, and updated V matrix. 
        Then, those updated derived matrices are used to calculate the A matrix. 
        """
        #print("updating_network_A_matrix_given_X")
        network = self.network
        #print("here we go")
        
        if self.tg_name in self.final_nodes: 
            self.tg_is_tf = True
            self.final_nodes.remove(self.tg_name)
        final_nodes = self.final_nodes
        
        W_df = network.W_df.copy()  # updating the W matrix
        
        if self.tg_is_tf: #1/31/24:
            W_df = W_df.drop(columns = [self.tg_name], index = [self.tg_name])

        if len(self.ppi_nodes_to_remove) > 0:
            W_df = W_df.drop(index=self.ppi_nodes_to_remove, columns=self.ppi_nodes_to_remove)
        default_edge_weight = self.prior_network.default_edge_weight

        if len(self.gexpr_nodes_to_add_for_net) > 0: # Simplified addition of new nodes
            for node in self.gexpr_nodes_to_add_for_net: #netrem_chosen_demo.gexpr_nodes_added:
                W_df[node] = default_edge_weight
                W_df.loc[node] = default_edge_weight


        # Consolidated indexing and reindexing operations
        W_df = W_df.reindex(index=final_nodes, columns=final_nodes)

        # Handle missing values
        np.fill_diagonal(W_df.values, 0)
        N = len(final_nodes)
        self.N = N
        W = W_df.values 
        np.fill_diagonal(W, (W.sum(axis=0) - W.diagonal()) / (N - 1))

        self.W = W
        self.W_df = W_df
        self.final_input_W_df = W_df
        # Feb 12, 2024
        edge_list = (
            W_df
            .reset_index()
            .melt(id_vars=["index"], var_name="TF2", value_name="weight_W")
            .rename(columns={"index": "TF1"})
            .query("TF1 != TF2")
        )
        self.edge_list = edge_list.values.tolist()
        ##########################################################################################

        # Update V matrix
        self.V = N * np.eye(N) - np.ones(N)

        # Update D matrix
        if not network.edge_values_for_degree:
            W_bool = (W > network.threshold_for_degree)
            d = np.float64(W_bool.sum(axis=0) - W_bool.diagonal())
        else: 
            if network.w_transform_for_d == "sqrt":
                W_to_use = np.sqrt(W)
            elif network.w_transform_for_d == "square":
                W_to_use = W ** 2
            else:
                W_to_use = W
            if network.w_transform_for_d == "avg": # added on 2/8/24
                d = W_to_use.diagonal()
            else: # summing up the values
                d = W_to_use.diagonal() * (self.N - 1)

        # Handle pseudocount and self loops
        d += network.pseudocount_for_degree

        if network.consider_self_loops:
            d += 1

        d_inv_sqrt = 1 / np.sqrt(d)
        # 2/5/24
        self.node_degree_df = pd.DataFrame(d, index = self.final_nodes, columns = ["d_i"])
        self.D_df = pd.DataFrame(np.diag(d_inv_sqrt))
        annotated_D = self.D_df
        annotated_D.columns = self.final_nodes
        annotated_D.index = self.final_nodes
        self.D_df = annotated_D
        ######
        
        self.D = ef.DiagonalLinearOperator(d_inv_sqrt)

        # Update inv_sqrt_degree_df
        self.inv_sqrt_degree_df = pd.DataFrame({
            "TF": self.final_nodes,
            "degree_D": self.D * np.ones(self.N)
        })

        Amat = self.D @ (self.V * W) @ self.D 
        A_df = pd.DataFrame(Amat, columns=final_nodes, index=final_nodes, dtype=np.float32)
        # Handle nodes based on `overlapped_nodes_only`
        gene_expression_nodes = self.gene_expression_nodes 
        nodes_to_add = list(set(self.gene_expression_nodes ) - set(final_nodes))
        self.nodes_to_add = nodes_to_add
        if not self.overlapped_nodes_only:
            for name in nodes_to_add:
                A_df[name] = 0
                A_df.loc[name] = 0
            A_df = A_df.reindex(columns=sorted(gene_expression_nodes), index=sorted(gene_expression_nodes))
        else:
            if len(nodes_to_add) == 1:
                print(f"Please note that we remove 1 node {nodes_to_add[0]} found in the input gene expression data (X) that is not found in the input network :)")
            elif len(nodes_to_add) > 1:
                print(f":) Since overlapped_nodes_only = True, please note that we remove {len(nodes_to_add)} gene expression nodes that are not found in the input network.")
                print(nodes_to_add)
            A_df = A_df.sort_index(axis=0).sort_index(axis=1)

        self.A_df = A_df
        # if graph.is_positive_semi_definite(A_df) == False:
        #     print(":( Error! A is NOT positive semi-definite! There exist some negative eigenvalues for A! :(")
        self.A = A_df.values
        self.nodes = A_df.columns.tolist()
        self.tf_names_list = self.nodes
        return self
    
    
    def preprocess_X_df(self, X_df):
        if self.tg_name in X_df.columns:
            X_df.drop(columns=[self.tg_name], inplace=True)
        
        # Ensure X_df contains only final_nodes.
        X_df = X_df.loc[:, self.final_nodes] if self.final_nodes else X_df
        return X_df.values    
    
   
    def preprocess_y_df(self, y):
        return y.values.flatten() if isinstance(y, pd.DataFrame) else y
    
        
    def return_Linear_ML_model(self, X, y):
        regr = LinearRegression(fit_intercept = self.y_intercept,
                               positive = self.all_pos_coefs,
                               n_jobs = self.num_jobs)
        regr.fit(X, y)
        return regr    
        
        
    def return_Lasso_ML_model(self, X, y):
        regr = Lasso(alpha = self.alpha_lasso, fit_intercept = self.y_intercept,
                    max_iter = self.max_lasso_iterations, tol = self.tolerance,
                    selection = self.lasso_selection,
                    positive = self.all_pos_coefs)
        regr.fit(X, y)
        return regr

    
    def return_LassoCV_ML_model(self, X, y):
        self.num_cv_folds = min(X.shape[0], self.num_cv_folds) # April 2024
        regr = LassoCV(cv = self.num_cv_folds, random_state = 0, 
                    fit_intercept = self.y_intercept, 
                     max_iter = self.max_lasso_iterations,
                      n_jobs = self.num_jobs,
                      tol = self.tolerance,
                      selection = self.lasso_selection,
                      positive = self.all_pos_coefs,
                      eps = self.lassocv_eps,
                      n_alphas = self.lassocv_n_alphas,
                      alphas = self.lassocv_alphas)
        regr.fit(X, y)
        return regr            

    
    def return_fit_ml_model(self, X, y):
        if self.model_type == "Linear":
            model_to_return = self.return_Linear_ML_model(X, y)
        elif self.model_type == "Lasso":
            model_to_return = self.return_Lasso_ML_model(X, y)
        elif self.model_type == "LassoCV":
            model_to_return = self.return_LassoCV_ML_model(X, y)
        return model_to_return
        
        
def netrem(edge_list, beta_net = 1, alpha_lasso = 0.01, default_edge_weight = 0.01,
                   edge_vals_for_d = True, w_transform_for_d = "none", degree_threshold = 0.5,
                  gene_expression_nodes = [], overlapped_nodes_only = False,
           y_intercept = False, standardize_X = True, standardize_y = True, center_y = False, view_network = False,
           model_type = "Lasso", lasso_selection = "cyclic", all_pos_coefs = False, tolerance = 1e-4, maxit = 10000,
                  num_jobs = -1, num_cv_folds = 5, lassocv_eps = 1e-3, 
                   lassocv_n_alphas = 100, # default in sklearn        
                lassocv_alphas = None, # default in sklearn
                   verbose = False, degree_pseudocount = 0,
                   hide_warnings = True):#, gamma_net = 0,):
    os.environ['PYTHONHASHSEED'] = '0'
    if hide_warnings:
        warnings.filterwarnings("ignore")
    default_beta = False
    default_alpha = False
    if beta_net == 1:
        print(":) netrem (may have prior knowledge): using beta_net default of", 1)
        default_beta = True
    if alpha_lasso == 0.01:
        if model_type != "LassoCV":
            print(":) netrem (may have prior knowledge): using alpha_lasso default of", 0.01)
            default_alpha = True
    self_loops = False  
    prior_graph_dict = {"edge_list": edge_list,
                        "gene_expression_nodes":gene_expression_nodes,
                       "edge_values_for_degree": edge_vals_for_d,
                       "consider_self_loops":self_loops,
                       "pseudocount_for_degree":degree_pseudocount,
                        "default_edge_weight": default_edge_weight,
                        "w_transform_for_d":w_transform_for_d,
                        "threshold_for_degree": degree_threshold, 
                        "verbose":verbose,
                       "view_network":view_network}         
    netty = graph.PriorGraphNetwork(**prior_graph_dict) # uses the network to get features like the A matrix.
    greg_dict = {"network": netty,
                "model_type": model_type,
                 "use_network":True,
                 "standardize_X":standardize_X,
                 "standardize_y":standardize_y,
                 "center_y":center_y,
                 #"gamma_net":gamma_net,
                 "y_intercept":y_intercept, 
                 "overlapped_nodes_only":overlapped_nodes_only,
                 "max_lasso_iterations":maxit,
                 "all_pos_coefs":all_pos_coefs,
                "view_network":view_network,
                "verbose":verbose}
    if default_alpha == False:
        greg_dict["alpha_lasso"] = alpha_lasso
    if default_beta == False:
        greg_dict["beta_net"] = beta_net
    if model_type != "Linear":
        greg_dict["tolerance"] = tolerance
        greg_dict["lasso_selection"] = lasso_selection
    if model_type != "Lasso":
        greg_dict["num_jobs"] = num_jobs
    if model_type == "LassoCV":
        greg_dict["num_cv_folds"] = num_cv_folds
        greg_dict["lassocv_eps"] = lassocv_eps
        greg_dict["lassocv_n_alphas"] = lassocv_n_alphas
        greg_dict["lassocv_alphas"] = lassocv_alphas
    greggy = NetREmModel(**greg_dict)
    return greggy


def netremCV(edge_list, X, y, 
             num_beta: int = 10,
             extra_beta_list = [0.25, 0.5, 0.75, 1], # additional beta to try out
            num_alpha: int = 10,
             max_beta: float = 200,  # max_beta used to help prevent explosion of beta_net values
            reduced_cv_search: bool = False, # should we do a reduced search (Randomized Search) or a GridSearch?
             default_edge_weight: float = 0.1,
            degree_threshold: float = 0.5,
            gene_expression_nodes = [],
            overlapped_nodes_only: bool = False,
             standardize_X: bool = True,
             standardize_y: bool = True,
             center_y: bool = False,
            y_intercept: bool = False,
            model_type = "Lasso",
            lasso_selection = "cyclic",
            all_pos_coefs: bool = False,
            tolerance: float = 1e-4,
            maxit: int = 10000,
            num_jobs: int = -1,
            num_cv_folds: int = 5,
            lassocv_eps: float = 1e-3,
            lassocv_n_alphas: int = 100, # default in sklearn        
            lassocv_alphas = None, # default in sklearn
            verbose = False,
            searchVerbosity: int = 2,
            show_warnings: bool = False):
    
    X_train = X
    y_train = y
    if show_warnings == False:
        warnings.filterwarnings('ignore')
    prior_graph_dict = {"edge_list": edge_list,
                            "gene_expression_nodes":gene_expression_nodes,
                           "edge_values_for_degree": False,
                           "consider_self_loops":False,
                           "pseudocount_for_degree":1e-3,
                            "default_edge_weight": default_edge_weight,
                            "w_transform_for_d":"none",
                            "threshold_for_degree": degree_threshold, 
                            "verbose":verbose,
                           "view_network":False}       

    prior_network = graph.PriorGraphNetwork(**prior_graph_dict)

    # generate the beta grid:
    if isinstance(X_train, pd.DataFrame):
        X_df = X_train
        gene_names_list = list(X_df.columns)
        if overlapped_nodes_only:
            nodes_list = prior_network.nodes#self.nodes
            common_nodes = ef.intersection(gene_names_list, nodes_list)
            common_nodes.sort()

            X_df = X_df.loc[:, X_df.columns.isin(common_nodes)]
            # Reorder columns of dataframe to match order in `column_order`
            X_df = X_df.reindex(columns=common_nodes)
        else:
            X_df = X_df.reindex(columns=gene_names_list)

        X_train_np = X_df.copy()
        y_train_np = y_train.copy()
        if standardize_X:
            if verbose:
                print("standardizing X :)")
            scaler = preprocessing.StandardScaler().fit(X_df)
            X_train_np = scaler.transform(X_df)
        else:
            X_train_np = np.array(X_df.values.tolist())
    if isinstance(y_train, pd.DataFrame):
        y_train_np = y_train_np.values.flatten()    
    beta_max = 0.5 * np.max(np.abs(X_train_np.T.dot(y_train_np)))
    beta_min = 0.01 * beta_max

    var_X = np.var(X_train_np)
    var_y = np.var(y_train_np)
    if beta_max > max_beta: # max_beta used to prevent explosion of beta_net values
        if verbose:
            print(":) using variance to define beta_net values")
        beta_max = 0.5 * np.max(np.abs(var_X * var_y)) * 100
        beta_min = 0.01 * beta_max
    if verbose:
        print(f"beta_min = {beta_min} and beta_max = {beta_max}")    
    beta_grid = np.logspace(np.log10(beta_max), np.log10(beta_min), num=num_beta)
    if extra_beta_list != None:
        if len(extra_beta_list) > 0:
            for add_beta in extra_beta_list: # we add additional beta based on user-defined list
                beta_grid = np.append(add_beta, beta_grid)


    beta_alpha_grid_dict = {"beta_network_vals": [], "alpha_lasso_vals": []}
    # generating the alpha-values that are corresponding
    try:
        with tqdm(beta_grid, desc=":) Generating beta_net and alpha_lasso pairs") as pbar:
            for beta in pbar:
                if verbose:
                    print("beta_network:", beta)
                # please fix it so it reflects what we want more... like the proper defaults
                netremCV_demo = NetREmModel(beta_net=beta, 
                                               model_type="LassoCV", 
                                               network=prior_network, 
                                               overlapped_nodes_only=overlapped_nodes_only,
                                               standardize_X = standardize_X,
                                               standardize_y = standardize_y,
                                               center_y = center_y,
                                              y_intercept = y_intercept, 
                              max_lasso_iterations  = maxit,
                              all_pos_coefs  = all_pos_coefs,
                              tolerance  = tolerance,
                              lasso_selection = lasso_selection,
                               num_cv_folds = num_cv_folds,
                                #num_jobs = num_jobs,
                                lassocv_eps = lassocv_eps,
                                 lassocv_n_alphas = lassocv_n_alphas,
                                lassocv_alphas = lassocv_alphas)
                if lassocv_alphas != None:
                    netremCV_demo.lassocv_alphas = lassocv_alphas

                # Fit the model and compute alpha_max and alpha_min
                netremCV_demo.fit(X_train, y_train)
                X_tilda_train = netremCV_demo.X_tilda_train
                y_tilda_train = netremCV_demo.y_tilda_train
                alpha_max = 0.5 * np.max(np.abs(X_tilda_train.T.dot(y_tilda_train)))
                alpha_min = 0.01 * alpha_max
                if verbose:
                    print(f"alpha_min = {alpha_min} and alpha_max = {alpha_max}")    

                # Generate alpha_grid based on alpha_max and alpha_min
                optimal_alpha = netremCV_demo.regr.alpha_
                # take the cross-validation alpha and apply as the best alpha as well for this beta_net
                beta_alpha_grid_dict["beta_network_vals"].append(beta)
                beta_alpha_grid_dict["alpha_lasso_vals"].append(optimal_alpha)
                # we also utilize the other alphas we have constructed dynamically and will find the best alpha among those
                alpha_grid = np.logspace(np.log10(alpha_min), np.log10(alpha_max), num=num_alpha)

                # Find the best alpha using cross-validation
                best_alpha = None
                best_score = float('-inf')
                for alpha in alpha_grid:
                    netremCV_demo = NetREmModel(beta_net=beta, 
                                alpha_lasso = alpha,
                               model_type="Lasso", 
                               network=prior_network, 
                                standardize_X = standardize_X,
                                standardize_y = standardize_y,
                                 center_y = center_y,
                               overlapped_nodes_only=overlapped_nodes_only,
                                y_intercept = y_intercept, 
                              max_lasso_iterations  = maxit,
                              all_pos_coefs  = all_pos_coefs,
                              tolerance  = tolerance,
                              lasso_selection = lasso_selection)           
                    scores = cross_val_score(netremCV_demo, X_train, y_train, cv=num_cv_folds, scoring = "neg_mean_squared_error")  # You can change cv to your specific cross-validation strategy
                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_alpha = alpha

                # Append the beta and best_alpha to the dictionary
                beta_alpha_grid_dict["beta_network_vals"].append(beta)
                beta_alpha_grid_dict["alpha_lasso_vals"].append(best_alpha)

    except Exception as e:
        print(f"An error occurred: {e}")
    if verbose:
        print("finished generate_alpha_beta_pairs")
        print(beta_alpha_grid_dict)
        print(f"Length of beta_alpha_grid_dict: {len(beta_alpha_grid_dict['beta_network_vals'])}")

    param_grid = [{"alpha_lasso": [alpha_las], "beta_net": [beta_net]} 
                  for alpha_las, beta_net in zip(beta_alpha_grid_dict["alpha_lasso_vals"], 
                                                 beta_alpha_grid_dict["beta_network_vals"])]
    if verbose:
        print(":) Performing NetREmCV with both beta_network and alpha_lasso as UNKNOWN.")
    initial_greg =  NetREmModel(network=prior_network, 
                                   y_intercept = y_intercept, 
                                   standardize_X = standardize_X,
                                   center_y = center_y,
                                   max_lasso_iterations=maxit,
                                   all_pos_coefs=all_pos_coefs,
                                   lasso_selection = lasso_selection,
                                   tolerance = tolerance,
                                    view_network=False,
                                    overlapped_nodes_only=overlapped_nodes_only)
    pbar = tqdm(total=len(param_grid)) # Assuming we're trying 9 combinations of parameters

    if reduced_cv_search:
    # Run RandomizedSearchCV
        if verbose:
            print(f":) since reduced_cv_search = {reduced_cv_search}, we perform RandomizedSearchCV on a reduced search space")
        grid_search= RandomizedSearchCV(initial_greg, 
                                   param_grid, 
                                   n_iter=num_alpha, 
                                   cv=num_cv_folds, 
                                    scoring = "neg_mean_squared_error",
                                   #scoring=make_scorer(custom_mse, greater_is_better=False),
                                   verbose=searchVerbosity)
    else:
    # Run GridSearchCV
        grid_search = GridSearchCV(initial_greg, param_grid=param_grid, cv=num_cv_folds, 
                                   scoring = "neg_mean_squared_error",
                                   #scoring=make_scorer(custom_mse, greater_is_better=False),
                                  verbose = searchVerbosity)
    grid_search.fit(X_train, y_train)

    # Extract and display the best hyperparameters
    best_params = grid_search.best_params_
    optimal_alpha = best_params["alpha_lasso"]
    optimal_beta = best_params["beta_net"]
    print(f":) NetREmCV found that the optimal alpha_lasso = {optimal_alpha} and optimal beta_net = {optimal_beta}")

    newest_netrem = NetREmModel(alpha_lasso = optimal_alpha,
                                   beta_net = optimal_beta, 
                                   network = prior_network,
                                   y_intercept = y_intercept, 
                                   standardize_X = standardize_X,
                                   center_y = center_y,
                                   max_lasso_iterations=maxit,
                                   all_pos_coefs=all_pos_coefs,
                                   lasso_selection = lasso_selection,
                                   tolerance = tolerance,
                                    view_network=False,
                                    overlapped_nodes_only=overlapped_nodes_only)
    newest_netrem.fit(X_train, y_train)
    train_mse = newest_netrem.test_mse(X_train, y_train)
    print(f":) Please note that the training Mean Square Error (MSE) from this fitted NetREm model is {train_mse}")
    return newest_netrem


# Function to lookup coefficient
def lookup_coef(tf, netrem_model):
    model_nonzero_coef_df = netrem_model.model_nonzero_coef_df
    coef_series = model_nonzero_coef_df.iloc[0].drop('y_intercept')
    return coef_series.get(tf, 0)  # Returns 0 if TF is not found


def organize_predictor_interaction_network(netrem_model):
    if "model_nonzero_coef_df" not in vars(netrem_model).keys():
        print(":( No NetREm model was built")
        return None
    TF_interaction_df = netrem_model.TF_interaction_df
    if "model_type" in TF_interaction_df.columns.tolist():
        TF_interaction_df = TF_interaction_df.drop(columns = ["model_type"])
    num_TFs = TF_interaction_df.shape[0]
    TF_coord_scores_pairwise_df = netrem_model.TF_coord_scores_pairwise_df
    if TF_coord_scores_pairwise_df.shape[0] == 0:
        return None
    TF_interaction_df = netrem_model.TF_coord_scores_pairwise_df.drop(columns = ["absVal_coord_score"])
    TF_interaction_df = TF_interaction_df.rename(columns = {"coordination_score":"coord_score_cs"})

    TF_interaction_df["sign"] = np.where((TF_interaction_df.coord_score_cs > 0), ":)", ":(")
    TF_interaction_df["potential_interaction"] = np.where((TF_interaction_df.coord_score_cs > 0), ":) cooperative (+)",
                                                        ":( competitive (-)")
    TF_interaction_df["absVal_coordScore"] = abs(TF_interaction_df["coord_score_cs"])
    TF_interaction_df["model_type"] = netrem_model.model_type
    TF_interaction_df["info"] = "matrix of TF-TF interactions"
    TF_interaction_df["candidate_TFs_N"] = num_TFs
    TF_interaction_df["target_gene_y"] = netrem_model.target_gene_y
    if 'num_final_predictors' in vars(netrem_model).keys():
        TF_interaction_df["num_final_predictors"] = netrem_model.num_final_predictors
    else:
        TF_interaction_df["num_final_predictors"] = "No final model :("
    TF_interaction_df["model_type"] = netrem_model.model_type
    TF_interaction_df["beta_net"] = netrem_model.beta_net
    TF_interaction_df["X_standardized"] = netrem_model.standardize_X
    TF_interaction_df["y_standardized"] = netrem_model.standardize_y

    TF_interaction_df["gene_data"] = "training gene expression data"

    # Step 1: Please Sort the DataFrame
    TF_interaction_df = TF_interaction_df.sort_values('absVal_coordScore', ascending=False)

    # Step 2: Get the rank
    TF_interaction_df['cs_magnitude_rank'] = TF_interaction_df['absVal_coordScore'].rank(method='min', ascending=False)

    # Step 3: Calculate the percentile
    TF_interaction_df['cs_magnitude_percentile'] = (1 - (TF_interaction_df['cs_magnitude_rank'] / TF_interaction_df['absVal_coordScore'].count())) * 100
    TF_interaction_df["TF_TF"] = TF_interaction_df["node_1"] + "_" + TF_interaction_df["node_2"]

    #TF_interaction_df["TF_TF"] = TF_interaction_df["TF1"] + "_" + TF_interaction_df["TF2"]

    standardized_data = True
    if "old_X_df" in vars(netrem_model).keys():
        standardized_X_corr_mat = netrem_model.X_df.corr()
        original_X_corr_mat = netrem_model.old_X_df.corr()
    else:
        original_X_corr_mat = netrem_model.X_df.corr()
        standardized_data = False
    # Melting the DataFrame into a 3-column edge list
    original_X_corr_df = original_X_corr_mat.reset_index().melt(id_vars=["index"], var_name="node_2", value_name="original_corr")
    original_X_corr_df.rename(columns={"index": "node_1"}, inplace=True)
    original_X_corr_df = original_X_corr_df[original_X_corr_df["node_1"] != original_X_corr_df["node_2"]]
    original_X_corr_df["TF_TF"] = original_X_corr_df["node_1"] + "_" + original_X_corr_df["node_2"]
    # Display the first few rows to verify the format

    if standardized_data:
        # Melting the DataFrame into a 3-column edge list
        standardized_X_corr_df = standardized_X_corr_mat.reset_index().melt(id_vars=["index"], var_name="node_2", value_name="standardized_corr")
        standardized_X_corr_df.rename(columns={"index": "node_1"}, inplace=True)
        standardized_X_corr_df = standardized_X_corr_df[standardized_X_corr_df["node_1"] != standardized_X_corr_df["node_2"]]
        standardized_X_corr_df["TF_TF"] = standardized_X_corr_df["node_1"] + "_" + standardized_X_corr_df["node_2"]
        standardized_X_corr_df.drop(columns = ["node_1", "node_2"], inplace = True)
        original_X_corr_df = pd.merge(original_X_corr_df, standardized_X_corr_df).drop(columns = ["node_1", "node_2"])
    TF_interaction_df = pd.merge(TF_interaction_df, original_X_corr_df)
    default_edge_w = netrem_model.network.default_edge_weight
    if "W_df" in vars(netrem_model.network).keys():
        W_df = netrem_model.network.W_df
    else:
        W_df = netrem_model.W_df
    ppi_net_df = W_df.reset_index().melt(id_vars=["index"], var_name="node_2", value_name="PPI_score")
    ppi_net_df.rename(columns={"index": "node_1"}, inplace=True)
    ppi_net_df["novel_link"] = np.where((ppi_net_df.PPI_score == default_edge_w), "yes", "no")
    ppi_net_df["TF_TF"] = ppi_net_df["node_1"] + "_" + ppi_net_df["node_2"]
    ppi_net_df = ppi_net_df[ppi_net_df["node_1"] != ppi_net_df["node_2"]] # 42849 rows × 3 columns
    ppi_net_df = ppi_net_df.drop(columns = ["node_1", "node_2"])
    TF_interaction_df = pd.merge(TF_interaction_df, ppi_net_df)
    TF_interaction_df["absVal_diff_cs_and_originalCorr"] = abs(TF_interaction_df["coord_score_cs"] - TF_interaction_df["standardized_corr"])
    TF_interaction_df['c_1'] = TF_interaction_df['node_1'].apply(lookup_coef, args=(netrem_model,))
    TF_interaction_df['c_2'] = TF_interaction_df['node_2'].apply(lookup_coef, args=(netrem_model,))
    return TF_interaction_df


def min_max_normalize(data, new_min=0.001, new_max=1):
    """
    Scale data to a new range [new_min, new_max].
    
    Parameters:
    - data: array-like, original data.
    - new_min: float, the minimum value of the scaled data.
    - new_max: float, the maximum value of the scaled data.
    
    Returns:
    - Array of normalized data.
    """
    X_min = data.min()
    X_max = data.max()
    
    # Apply the min-max normalization formula adjusted for the new range
    normalized_data = new_min + ((data - X_min) * (new_max - new_min)) / (X_max - X_min)    
    return normalized_data


def netrem_info_breakdown_df(netrem_model):

    part1 = netrem_model.E_part_XtX
    part1_df = part1.reset_index().melt(id_vars=["index"], var_name="TF2", value_name="part1_XtX/M")
    part1_df.rename(columns={"index": "TF1"}, inplace=True)
    part1_df = part1_df[part1_df["TF1"] != part1_df["TF2"]]
    part1_df["TF1_TF2"] = part1_df["TF1"] + "_" + part1_df["TF2"]
    if "W_df" in vars(netrem_model).keys():
        W_part = netrem_model.W_df
    else:
        W_part = netrem_model.network.W_df
    W_part = W_part.reset_index().melt(id_vars=["index"], var_name="TF2", value_name="weight_W")
    W_part.rename(columns={"index": "TF1"}, inplace=True)
    W_part = W_part[W_part["TF1"] != W_part["TF2"]]
    W_part["TF1_TF2"] = W_part["TF1"] + "_" + W_part["TF2"]
    W_part.drop(columns = ["TF1", "TF2"], inplace = True)
    if "A_df" in vars(netrem_model).keys():
        A_part = netrem_model.A_df
    else:
        A_part = netrem_model.network.A_df
    A_part = A_part.reset_index().melt(id_vars=["index"], var_name="TF2", value_name="A")
    A_part.rename(columns={"index": "TF1"}, inplace=True)
    A_part = A_part[A_part["TF1"] != A_part["TF2"]]
    A_part["TF1_TF2"] = A_part["TF1"] + "_" + A_part["TF2"]
    A_part = A_part.drop(columns = ["TF1", "TF2"])
    part2 = netrem_model.E_part_netReg
    part2_df = part2.reset_index().melt(id_vars=["index"], var_name="TF2", value_name="part2_betaA")
    part2_df.rename(columns={"index": "TF1"}, inplace=True)
    part2_df = part2_df[part2_df["TF1"] != part2_df["TF2"]]
    part2_df["beta_net"] = netrem_model.beta_net
    part2_df["TF1_TF2"] = part2_df["TF1"] + "_" + part2_df["TF2"]
    part2_df = part2_df.drop(columns = ["TF1", "TF2"])

    if "node_degree_df" in vars(netrem_model).keys():
        node_degrees_df = netrem_model.node_degree_df
    else:
        node_degrees_df = netrem_model.network.node_degree_df

    node_degrees_df = node_degrees_df.reset_index()

    TF1_degree_df = node_degrees_df.rename(columns = {"index":"TF1", "d_i":"deg_TF1"})
    TF2_degree_df = node_degrees_df.rename(columns = {"index":"TF2", "d_i":"deg_TF2"})


    main_parts_df = pd.merge(part1_df, W_part)
    main_parts_df = pd.merge(main_parts_df, TF1_degree_df)
    main_parts_df = pd.merge(main_parts_df, TF2_degree_df)
    main_parts_df = pd.merge(main_parts_df, A_part)
    main_parts_df = pd.merge(main_parts_df, part2_df)

    main_parts_df["B_score"] = main_parts_df["part1_XtX/M"] + main_parts_df["part2_betaA"]
    main_parts_df["part1/part2"] = abs(main_parts_df["part1_XtX/M"]/main_parts_df["part2_betaA"])
    main_parts_df["abs_total"] = abs(main_parts_df["part1_XtX/M"]) + abs(main_parts_df["part2_betaA"])

    main_parts_df["perc_XtX_part"] = abs(main_parts_df["part1_XtX/M"])/(main_parts_df["abs_total"]) * 100.0
    main_parts_df["perc_betaA_part"] = abs(main_parts_df["part2_betaA"])/(main_parts_df["abs_total"]) * 100.0
    main_parts_df = main_parts_df.drop(columns = ["abs_total", "TF1_TF2"])
    main_parts_df["abs_diff_in_perc"] = abs(main_parts_df["perc_betaA_part"] - main_parts_df["perc_XtX_part"])
    main_parts_df["TF1_TF2"] = main_parts_df["TF1"] + "_" + main_parts_df["TF2"]
    return main_parts_df



def multiply_frobenius_norm(norm, matrix, ignore_main_diag=False):
    """
    Multiply the matrix by its Frobenius norm. If ignore_main_diag is True,
    multiply only the off-diagonal elements by the Frobenius norm of the off-diagonal elements.
    This function now supports receiving a Pandas DataFrame as the matrix.

    Parameters:
    - norm: The norm to multiply the matrix by.
    - matrix: The matrix (as a NumPy array or Pandas DataFrame) to be multiplied.
    - ignore_main_diag (bool): Determines whether the main diagonal should be ignored.

    Returns:
    - The matrix after multiplication, in the same format as the input (NumPy array or DataFrame).
    """
    # Convert DataFrame to NumPy array if necessary
    if isinstance(matrix, pd.DataFrame):
        matrix_np = matrix.values
        was_dataframe = True
    else:
        matrix_np = matrix
        was_dataframe = False
    
    if ignore_main_diag:
        # Create a mask for the off-diagonal elements
        mask = np.ones_like(matrix_np, dtype=bool)
        np.fill_diagonal(mask, False)
        # Multiply only the off-diagonal elements by the norm
        matrix_np[mask] *= norm
    else:
        # Multiply the entire matrix by the norm
        matrix_np *= norm

    # Convert back to DataFrame if the original input was a DataFrame
    if was_dataframe:
        return pd.DataFrame(matrix_np, index=matrix.index, columns=matrix.columns)
    else:
        return matrix_np


    
def is_standardized(df, tol=1e-4):
    """
    Check if the given DataFrame is standardized (mean ~ 0 and std deviation ~ 1) for all columns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        tol (float): Tolerance for the mean and standard deviation checks.
        
    Returns:
        bool: True if the DataFrame is standardized, False otherwise.
    """
    # Calculate means and standard deviations for all columns at once
    means = df.mean()
    stds = df.std()

    # Check if all means are within the tolerance of 0 and all stds are within the tolerance of 1
    return np.all(np.abs(means) < tol) and np.all(np.abs(stds - 1) < tol)



def return_TF_coord_scores_df(netrem_model):
        TF_coord_scores_df = (
            netrem_model.TF_interaction_df
            .reset_index()
            .melt(id_vars='index', var_name='node_2', value_name='coordination_score')
            .rename(columns={"index": "node_1"})
        )

        # Filter out rows where node_1 equals node_2 and calculate absVal_coord_score in one step
        TF_coord_scores_df = TF_coord_scores_df[TF_coord_scores_df["node_1"] != TF_coord_scores_df["node_2"]].assign(absVal_coord_score=lambda x: abs(x.coordination_score))

        # Sort values based on absVal_coord_score without reassigning the dataframe
        TF_coord_scores_pairwise_df = TF_coord_scores_df.sort_values(by="absVal_coord_score", ascending=False)
        return TF_coord_scores_pairwise_df
    
   

def simprem(prior_network, beta_net = 1, alpha_lasso = 0.01, overlapped_nodes_only = False,
           y_intercept = False, standardize_X = True, standardize_y = True, center_y = False, view_network = False,
           model_type = "Lasso", lasso_selection = "cyclic", all_pos_coefs = False, tolerance = 1e-4, maxit = 10000,
                  num_jobs = -1, num_cv_folds = 5, lassocv_eps = 1e-3, 
                   lassocv_n_alphas = 100, # default in sklearn        
                lassocv_alphas = None, # default in sklearn
                   verbose = False, 
                   hide_warnings = True):
    """
    Please note this is :) Simpler NetREm when we do not have prior gene regulatory knowledge and all Target Genes (TGs) 
     in the cell-type have the same set of N* candidate TFs. 
     
     Please note that to obtain prior_network, ye can directly use: graph.build_prior_network(edge_list = ...) function.
     That is, prior_network =  graph.build_prior_network()
    """

    os.environ['PYTHONHASHSEED'] = '0'
    if hide_warnings:
        warnings.filterwarnings("ignore")
    default_beta = False
    default_alpha = False
    if beta_net == 1:
        print(":) simprem (no prior knowledge): using beta_net default of", 1)
        default_beta = True
    if alpha_lasso == 0.01:
        if model_type != "LassoCV":
            print(":) simprem (no prior knowledge): using alpha_lasso default of", 0.01)
            default_alpha = True
  
    greg_dict = {"network": prior_network,
                "model_type": model_type,
                 "use_network":True,
                 "standardize_X":standardize_X,
                 "standardize_y":standardize_y,
                 "center_y":center_y,
                 #"gamma_net":gamma_net,
                 "y_intercept":y_intercept, 
                 "overlapped_nodes_only":overlapped_nodes_only,
                 "max_lasso_iterations":maxit,
                 "all_pos_coefs":all_pos_coefs,
                "view_network":view_network,
                "verbose":verbose}
    if default_alpha == False:
        greg_dict["alpha_lasso"] = alpha_lasso
    if default_beta == False:
        greg_dict["beta_net"] = beta_net
    if model_type != "Linear":
        greg_dict["tolerance"] = tolerance
        greg_dict["lasso_selection"] = lasso_selection
    if model_type != "Lasso":
        greg_dict["num_jobs"] = num_jobs
    if model_type == "LassoCV":
        greg_dict["num_cv_folds"] = num_cv_folds
        greg_dict["lassocv_eps"] = lassocv_eps
        greg_dict["lassocv_n_alphas"] = lassocv_n_alphas
        greg_dict["lassocv_alphas"] = lassocv_alphas
    greggy = NetREmModel(**greg_dict)
    return greggy