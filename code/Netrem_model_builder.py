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
# from packages_needed import *
import essential_functions as ef
import error_metrics as em # why to do import
import Netrem_model_builder as nm
import DemoDataBuilderXandY as demo
import PriorGraphNetwork as graph
import netrem_evaluation_functions as nm_eval
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
import copy
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
        "beta_network": (0, None),
        "num_cv_folds": (0, None),
        "fit_y_intercept": [False, True],
        "use_network": [True, False],
        "max_lasso_iterations": (1, None),
        "model_type": ["Lasso", "LassoCV", "Linear"],
        "tolerance": (0, None),
        "num_jobs": (1, 1e10),
        "lasso_selection": ["cyclic", "random"],
        "lassocv_eps": (0, None),
        "lassocv_n_alphas": (1, None)     
    }
    
    def __init__(self,  **kwargs):
        self.info = "NetREm Model"
        self.verbose = False
        self.overlapped_nodes_only = False # restrict the nodes to only being those found in the network? overlapped_nodes_only
        self.num_cv_folds = 5 # for cross-validation models
        self.num_jobs = -1 # for LassoCV or LinearRegression (here, -1 is the max possible for CPU)
        self.all_pos_coefs = False # for coefficients
        self.model_type = "Lasso"
        self.use_network = True
        self.fit_y_intercept = False
        self.max_lasso_iterations = 10000
        self.view_network = True
        self.model_info = "unfitted_model :("
        self.target_gene_y = "Unknown :("
        self.tolerance = 1e-4
        self.lasso_selection = "cyclic" # default in sklearn
        self.lassocv_eps = 1e-3 # default in sklearn        
        self.lassocv_n_alphas = 100 # default in sklearn        
        self.lassocv_alphas = None # default in sklearn 
        self.beta_network = kwargs.get('beta_network', 1)
        self.__dict__.update(kwargs)
        required_keys = ["network", "beta_network"] 
        if self.model_type == "Lasso":
            self.alpha_lasso = kwargs.get('alpha_lasso', 0.01)
            self.optimal_alpha = "User-specified optimal alpha lasso: " + str(self.alpha_lasso)
            required_keys += ["alpha_lasso"]
        elif self.model_type == "LassoCV": 
            self.alpha_lasso = "LassoCV finda optimal alpha"
            self.optimal_alpha = "Since LassoCV is model_type, please fit model using X and y data to find optimal_alpha."
        else: # model_type == "Linear":
            self.alpha_lasso = "No alpha needed"
            self.optimal_alpha = "No alpha needed" #
        missing_keys = [key for key in required_keys if key not in self.__dict__] # check that all required keys are present:
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        if self.use_network:
            prior_network = self.network
            self.prior_network = prior_network
            self.preprocessed_network = prior_network.preprocessed_network
            self.network_params = prior_network.param_lists
            self.network_nodes_list = prior_network.final_nodes # tf_names_list
        self.kwargs = kwargs
        self._apply_parameter_constraints() # ensuring that the parameter constraints are met   
   

    def __repr__(self):
        args = [f"{k}={v}" for k, v in self.__dict__.items() if k != 'param_grid' and k in self.kwargs]
        return f"{self.__class__.__name__}({', '.join(args)})"    
    
    
    def check_overlaps_work(self, X):
        X_df = X.sort_index(axis=0).sort_index(axis=1) # sorting the X dataframe by rows and columns. 
        final_set = set(self.final_nodes)
        network_set = set(self.network_nodes_list) # ppi_net_nodes
        if network_set == final_set:
            return False
        return True
    
    
    def updating_network_and_X_during_fitting(self, X, y):   
        """ :) Please note that this function will update the prior network information and the 
        X input data (training) during the fitting of the model. It will determine if the common predictors
        should be used (based on if overlapped_nodes_only is True) or if all of the X input data should be used. """
        X_df = X.sort_index(axis=0).sort_index(axis=1) # sorting the X dataframe by rows and columns. 
        self.X_df = X_df
        self.target_gene_y = y.columns.tolist()[0]
        gene_expression_nodes = list(X_df.columns) # these will be sorted
        gene_expression_nodes.sort()
        ppi_net_nodes = self.network_nodes_list
        common_nodes = ef.intersection(ppi_net_nodes, gene_expression_nodes)
        # defensive-programming :)
        if len(common_nodes) == 0: # may be possible that the X dataframe needs to be transposed if provided incorrectly
            print("Please note: we are flipping X dataframe around so that the rows are samples and the columns are gene/TF names :)")
            X_df = X_df.transpose()
            gene_expression_nodes = list(X_df.columns) # these will be sorted
            common_nodes = ef.intersection(ppi_net_nodes, gene_expression_nodes)
        common_nodes.sort()
        self.gene_expression_nodes = gene_expression_nodes
        self.common_nodes = common_nodes 
        if self.overlapped_nodes_only == True:
            self.final_nodes = common_nodes
        elif self.preprocessed_network:
            self.final_nodes = self.prior_network.final_nodes
        else:
            self.final_nodes = gene_expression_nodes
        final_nodes = self.final_nodes
        ppi_nodes_to_remove = list(set(ppi_net_nodes) - set(final_nodes))
        self.gexpr_nodes_added = list(set(gene_expression_nodes) - set(final_nodes))
        self.gexpr_nodes_to_add_for_net = list(set(gene_expression_nodes) - set(common_nodes))
        if self.verbose:
            if len(ppi_nodes_to_remove) > 0:
                if len(ppi_nodes_to_remove) == 1:
                    print(f"Please note that we remove 1 node {ppi_nodes_to_remove[0]} found in the input network that is not found in the input gene expression data (X) :)")
                else:
                    print(f"Please note that we remove {len(ppi_nodes_to_remove)} nodes found in the input network that are not found in the input gene expression data (X) :)")
                    print(ppi_nodes_to_remove)
            else:
                print(f":) Please note that all {len(common_nodes)} nodes found in the network are also found in the input gene expression data (X) :)")           
        filter_network_bool = self.check_overlaps_work(X_df)
        if filter_network_bool:
            print("Please note that we need to update the network information")
            self.updating_network_A_matrix_given_X() # updating the A matrix given the gene expression data X
            if self.view_network:
                ef.draw_arrow()
                self.view_W_network = self.view_W_network()
        else:
            self.A_df = self.network.A_df
            self.A = self.network.A
            self.nodes = self.A_df.columns.tolist()
        self.network_params = self.prior_network.param_lists
        self.network_info = "fitted_network"
        self.M = y.shape[0]       
        self.N = len(self.final_nodes) # pre-processing:
        self.X_train = self.preprocess_X_df(X)
        self.y_train = self.preprocess_y_df(y)
        return self
    
    
    def organize_B_interaction_list(self): # TF-TF interactions to output :)
        self.B_train = self.compute_B_matrix(self.X_train)
        self.B_interaction_df = pd.DataFrame(self.B_train, index = self.final_nodes, columns = self.final_nodes)
#         self.B_interaction_df = self.B_interaction_df.reset_index().melt(id_vars='index', var_name='TF2', value_name='B_train_weight')
#         self.B_interaction_df = self.B_interaction_df.rename(columns = {"index":"TF1"})  
#         self.B_interaction_df = self.B_interaction_df[self.B_interaction_df["TF1"] != self.B_interaction_df["TF2"]]
#         self.B_interaction_df = self.B_interaction_df.sort_values(by = ['B_train_weight'], ascending = False)
#         self.B_interaction_df["beta_network"] = self.beta_network
        return self
    
    
    def fit(self, X, y): # fits a model Function used for model training 
        self.updating_network_and_X_during_fitting(X, y)
        self.organize_B_interaction_list()
        self.B_train_times_M = self.compute_B_matrix_times_M(self.X_train)
        self.X_tilda_train, self.y_tilda_train = self.compute_X_tilde_y_tilde(self.B_train_times_M, self.X_train, 
                                                                              self.y_train)
        self.X_training_to_use, self.y_training_to_use = self.X_tilda_train, self.y_tilda_train
        self.regr = self.return_fit_ml_model(self.X_training_to_use, self.y_training_to_use)
        ml_model = self.regr
        if self.model_type == "LassoCV":
            self.optimal_alpha = "Cross-Validation optimal alpha lasso: " + str(ml_model.alpha_)
        if self.verbose:
            print(self.optimal_alpha)
        self.coef = ml_model.coef_ # Please Get the coefficients
        self.coef[self.coef == -0.0] = 0
        if self.fit_y_intercept:
            self.intercept = ml_model.intercept_
        self.predY_tilda_train = ml_model.predict(self.X_training_to_use) # training data   
        self.mse_tilda_train = self.calculate_mean_square_error(self.y_training_to_use, self.predY_tilda_train) # Calculate MSE
        self.predY_train = ml_model.predict(self.X_train) # training data   
        self.mse_train = self.calculate_mean_square_error(self.y_train, self.predY_train) # Calculate MSE      
        if self.fit_y_intercept:
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
        else:
            self.model_nonzero_coef_df = self.model_coef_df[selected_cols]    
            if len(selected_cols) > 1: # and self.model_type != "Linear":
                self.netrem_model_predictor_results(y)
        return self
    
    
    def netrem_model_predictor_results(self, y):
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
        tg = y.columns.tolist()[0]
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
        self.final_corr_vs_coef_df = self.corr_vs_coef_df[["info", "input_data"] + self.model_nonzero_coef_df.columns.tolist()[1:]]
  
        netrem_model_df = self.model_nonzero_coef_df.transpose()
        netrem_model_df.columns = ["coef"]
        netrem_model_df["TF"] = netrem_model_df.index.tolist()
        netrem_model_df["TG"] = tg
        if self.fit_y_intercept:
            netrem_model_df["info"] = "netrem_with_intercept"
        else:
            netrem_model_df["info"] = "netrem_no_intercept"
        netrem_model_df["train_mse"] = self.mse_train
        if self.model_type != "Linear":
            netrem_model_df["beta_net"] = self.beta_network
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
        return self

    
    def view_W_network(self):
        roundedW = np.round(self.W, decimals=4)
        wMat = ef.view_matrix_as_dataframe(roundedW, column_names_list=self.final_nodes, row_names_list=self.final_nodes)
        w_edgeList = wMat.stack().reset_index()
        # https://stackoverflow.com/questions/48218455/how-to-create-an-edge-list-dataframe-from-a-adjacency-matrix-in-python
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns={"level_0": "source", "level_1": "target", 0: "weight"})
        w_edgeList = w_edgeList[w_edgeList["weight"] != 0]
        G = nx.from_pandas_edgelist(w_edgeList, source="source", target="target", edge_attr="weight")
        pos = nx.spring_layout(G)
        weights_list = [G.edges[e]['weight'] * self.prior_network.edge_weight_scaling for e in G.edges]
        fig, ax = plt.subplots()
        if self.overlapped_nodes_only == False:
            nodes_to_add = list(set(self.gene_expression_nodes) - set(self.common_nodes))
        if self.overlapped_nodes_only == False and len(nodes_to_add) > 0:
            new_nodes = nodes_to_add 
            print(f":) {len(new_nodes)} new nodes added to network based on gene expression data {new_nodes}")
            node_color_map = {node: self.prior_network.added_node_color_name if node in new_nodes else self.prior_network.node_color_name for node in G.nodes}
            nx.draw(G, pos, node_color=node_color_map.values(), edge_color=self.prior_network.edge_color_name, with_labels=True, width=weights_list, ax=ax)
        else:
            nx.draw(G, pos, node_color=self.prior_network.node_color_name, edge_color=self.prior_network.edge_color_name, with_labels=True, width=weights_list, ax=ax)
        labels = {e: G.edges[e]['weight'] for e in G.edges}
        return nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
    
    
    def compute_B_matrix_times_M(self, X):
        """ M is N_sample, because ||y - Xc||^2 need to be normalized by 1/n_sample, but not the 2 * beta_L2 * c'Ac term
        see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1 where M = n_sample 
        Calculations"""
        XtX = X.T @ X
        beta_L2 = self.beta_network
        N_squared = self.N * self.N
        part_2 = 2.0 * float(beta_L2) * self.M / (N_squared) * self.A
        B = XtX + part_2 
        return B
    
    
    def compute_B_matrix(self, X):
        """ M is N_sample, because ||y - Xc||^2 need to be normalized by 1/n_sample, but not the 2 * beta_L2 * c'Ac term
        see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        where M = n_sample 
        Outputting for user """
        return self.compute_B_matrix_times_M(X) / self.M
    
    
    def compute_X_tilde_y_tilde(self, B, X, y):
        """Compute X_tilde, y_tilde such that X_tilde.T @ X_tilde = B,   y_tilde.T @ X_tilde = y.T @ X """
        U, s, _Vh = np.linalg.svd(B, hermitian=True)  # B = U @ np.diag(s) @ _Vh
        if (cond := s[0] / s[-1]) > 1e10:
            print(f'Large conditional number of B matrix: {cond: .2f}')
        S_sqrt = ef.DiagonalLinearOperator(np.sqrt(s))
        S_inv_sqrt = ef.DiagonalLinearOperator(1 / np.sqrt(s))
        X_tilde = S_sqrt @ U.T
        y_tilde = (y @ X @ U @ S_inv_sqrt).T
        # assert(np.allclose(y.T @ X, y_tilde.T @ X_tilde))
        # assert(np.allclose(B, X_tilde.T @ X_tilde))
        # scale: we normalize by 1/M, but sklearn.linear_model.Lasso normalize by 1/N because X_tilde is N*N matrix,
        # so Lasso thinks the number of sample is N instead of M, to use lasso solve our desired problem, correct the scale
        scale = np.sqrt(self.N / self.M)
        X_tilde *= scale
        y_tilde *= scale
        return X_tilde, y_tilde
    
    
    def predict_y_from_y_tilda(self, X, X_tilda, pred_y_tilda):
        X = self.preprocess_X_df(X) # calculate transpose of inverse of X
        X_inv = np.linalg.inv(X)
        X_inv_transpose = X_inv.T
        X_tilda_transpose = X_tilda.T # calculate transpose of X_tilda
        pred_y = X_inv_transpose.dot(X_tilda_transpose).dot(pred_y_tilda) # calculate matrix multiplication
        return pred_y
    
    
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
        X_test = self.preprocess_X_df(X_test) # X_test
        return self.regr.predict(X_test)

    
    def test_mse(self, X_test, y_test):
        X_test = self.preprocess_X_df(X_test) # X_test
        y_test = self.preprocess_y_df(y_test) 
        predY_test = self.regr.predict(X_test) # training data   
        mse_test = self.calculate_mean_square_error(y_test, predY_test) # Calculate MSE
        return mse_test #mse_test
    
    
    def get_params(self, deep=True):
        params_dict = {"info":self.info, "alpha_lasso": self.alpha_lasso, "beta_network": self.beta_network, 
                "fit_y_intercept": self.fit_y_intercept, "model_type":self.model_type, 
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
            params_dict["alpha_lasso"] = "No alpha needed"
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
            X = self.preprocess_X_df(X) # X_test
        if isinstance(y, pd.DataFrame):
            y = self.preprocess_y_df(y)
        # Make predictions using the predict method of your custom estimator
        y_pred = self.predict(X)
        # Calculate the normalized mean squared error between the true and predicted values
        nmse_ = (y - y_pred)**2
        nmse_[y_pred==0] *= zero_coef_penalty
        nmse_ = nmse_.mean() / (y**2).mean()
        return -nmse_ # Return the negative normalized mean squared error
    
    
    def updating_network_A_matrix_given_X(self) -> np.ndarray:
        """ When we call the fit method, this function is used to help us update the network information.
        Here, we can generate updated W matrix, updated D matrix, and updated V matrix. 
        Then, those updated derived matrices are used to calculate the A matrix. 
        """
        network = self.network
        final_nodes = self.final_nodes
        W_df = network.W_df # updating the W matrix
        if len(self.gexpr_nodes_added) > 0:
            new_nodes = self.gexpr_nodes_added # New nodes to add
            n = W_df.shape[0] + len(new_nodes) # Create larger dataframe with new nodes and NaN values
            df2 = pd.DataFrame(np.nan, index=range(n), columns=range(n))
            df2.iloc[:W_df.shape[0], :W_df.shape[1]] = W_df.values # Copy over values from original dataframe
            df2.index = list(W_df.columns) + new_nodes # Update index and column names to include new nodes
            df2.columns = list(W_df.columns) + new_nodes
            W_df = df2
        W_df = W_df.loc[:, W_df.columns.isin(final_nodes)] # Reorder columns of dataframe to match order in `column_order`
        W_df = W_df.loc[W_df.index.isin(final_nodes),:]     
        W_df = W_df.reindex(columns = final_nodes)
        W_df = W_df.reindex(index = final_nodes)
        W_df = W_df.fillna(value = self.prior_network.default_edge_weight)
        np.fill_diagonal(W_df.values, 0)
        N = len(final_nodes)
        self.N = N
        W = W_df.values 
        np.fill_diagonal(W, (W.sum(axis=0) - W.diagonal()) / (N - 1))
        self.W = W
        self.W_df = pd.DataFrame(self.W, index = final_nodes, columns = final_nodes)
        if ef.check_symmetric(W) == False:
            print(":( W matrix is NOT symmetric")
        self.V = N * np.eye(N) - np.ones(N) ### updating the V matrix:
        ### updating the D matrix:
        if network.edge_values_for_degree == False:
            W_bool = (W > network.threshold_for_degree)
            d = np.float64(W_bool.sum(axis=0) - W_bool.diagonal())
        else: 
            if network.w_transform_for_d == "sqrt":   # taking the square root of the weights for the edges
                W_to_use = np.sqrt(W)
            elif network.w_transform_for_d == "square": # self.squaring_weights_for_degree:
                W_to_use = W ** 2
            else:
                W_to_use = W
            d = W_to_use.diagonal() * (self.N - 1) # summing the edge weights
        d += network.pseudocount_for_degree
        if network.consider_self_loops:
            d += 1 # we also add in a self-loop :)    
        d_inv_sqrt = 1 / np.sqrt(d)
        self.D = ef.DiagonalLinearOperator(d_inv_sqrt)
        self.inv_sqrt_degree_df = pd.DataFrame(self.final_nodes, columns = ["TF"])
        self.inv_sqrt_degree_df["degree_D"] = self.D  * np.ones(self.N)
        Amat = self.D @ (self.V * W) @ self.D 
        A_df = pd.DataFrame(Amat, columns = final_nodes, index = final_nodes, dtype=np.float64)
        # we need to then see if we have self.overlapped_nodes_only or not
        gene_expression_nodes = self.gene_expression_nodes 
        nodes_to_add = list(set(gene_expression_nodes) - set(final_nodes))
        self.nodes_to_add = nodes_to_add
        if self.overlapped_nodes_only == False: # adding all gene expression nodes with value of 0 here.
            for name in nodes_to_add:
                A_df.loc[name] = 0
                A_df[name] = 0
            gene_expression_nodes = gene_expression_nodes.sort()
            A_df = A_df.reindex(columns = gene_expression_nodes)
            A_df = A_df.reindex(index = gene_expression_nodes)
        else: 
            if self.verbose:
                if len(nodes_to_add) == 1:
                    print(f"Please note that we remove 1 node {nodes_to_add[0]} found in the input gene expression data (X) that is not found in the input network :)")
                else:
                    print(f":) Since overlapped_nodes_only = True, please note that we remove {len(nodes_to_add)} gene expression nodes that are not found in the input network.")
                    print(nodes_to_add)
            A_df = A_df.sort_index(axis=0).sort_index(axis=1)
        self.A_df = A_df
        self.A = A_df.values
        self.nodes = A_df.columns.tolist()
        self.tf_names_list = self.nodes
        return self
    
    
    def preprocess_X_df(self, X):
        if isinstance(X, pd.DataFrame):
            X_df = X 
            column_names_list = list(X_df.columns)
            overlap_num = len(ef.intersection(column_names_list, self.final_nodes))
            if overlap_num == 0:
                print("Please note: we are flipping X dataframe around so that the rows are samples and the columns are gene/TF names :)")
                X_df = X_df.transpose()
                column_names_list = list(X_df.columns)
                overlap_num = len(ef.intersection(column_names_list, self.common_nodes))
            gene_names_list = self.final_nodes # so that this matches the order of columns in A matrix as well
            X_df = X_df.loc[:, X_df.columns.isin(gene_names_list)] # filtering the X_df as needed based on the columns
            X_df = X_df.reindex(columns=gene_names_list)# Reorder columns of dataframe to match order in `column_order`
            X = np.array(X_df.values.tolist())      
        return X
    
    
    def preprocess_y_df(self, y):
        if isinstance(y, pd.DataFrame):
            y = y.values.flatten()
        return y
        
        
    def return_Linear_ML_model(self, X, y):
        regr = LinearRegression(fit_intercept = self.fit_y_intercept,
                               positive = self.all_pos_coefs,
                               n_jobs = self.num_jobs)
        regr.fit(X, y)
        return regr    
        
        
    def return_Lasso_ML_model(self, X, y):
        regr = Lasso(alpha = self.alpha_lasso, fit_intercept = self.fit_y_intercept,
                    max_iter = self.max_lasso_iterations, tol = self.tolerance,
                    selection = self.lasso_selection,
                    positive = self.all_pos_coefs)
        regr.fit(X, y)
        return regr

    
    def return_LassoCV_ML_model(self, X, y):
        regr = LassoCV(cv = self.num_cv_folds, random_state = 0, 
                    fit_intercept = self.fit_y_intercept, 
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
        
        
def netrem(edge_list, beta_net = 1, alpha_lasso = 0.01, default_edge_weight = 0.1,
                  degree_threshold = 0.5, gene_expression_nodes = [], overlapped_nodes_only = False,
           y_intercept = False, view_network = False,
           model_type = "Lasso", lasso_selection = "cyclic", all_pos_coefs = False, tolerance = 1e-4, maxit = 10000,
                  num_jobs = -1, num_cv_folds = 5, lassocv_eps = 1e-3,
                   lassocv_n_alphas = 100, # default in sklearn        
                lassocv_alphas = None, # default in sklearn
                   verbose = False,
                   hide_warnings = True):
    degree_pseudocount = 1e-3
    if hide_warnings:
        warnings.filterwarnings("ignore")
    default_beta = False
    default_alpha = False
    if beta_net == 1:
        print("using beta_net default of", 1)
        default_beta = True
    if alpha_lasso == 0.01:
        if model_type != "LassoCV":
            print("using alpha_lasso default of", 0.01)
            default_alpha = True
    edge_vals_for_d = False
    self_loops = False
    w_transform_for_d = "none"
    
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
                 "fit_y_intercept":y_intercept, 
                 "overlapped_nodes_only":overlapped_nodes_only,
                 "max_lasso_iterations":maxit,
                 "all_pos_coefs":all_pos_coefs,
                "view_network":view_network,
                "verbose":verbose}
    if default_alpha == False:
        greg_dict["alpha_lasso"] = alpha_lasso
    if default_beta == False:
        greg_dict["beta_network"] = beta_net
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