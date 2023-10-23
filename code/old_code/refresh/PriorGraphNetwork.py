# PriorGraphNetwork Class: :)
# Standard libraries
import os
import sys
import random
import copy
import warnings

# Third-party libraries
import pandas as pd
import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm
import jinja2

# Scikit-learn imports
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, ElasticNetCV, Ridge

# Scipy imports
from scipy.linalg import svd as robust_svd
from scipy.sparse.linalg.interface import LinearOperator

# Type hinting
from typing import Optional, List, Tuple
from numpy.typing import ArrayLike

# Custom module imports
import essential_functions as ef
import error_metrics as em
import DemoDataBuilderXandY as demo


import math
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec

        
# Constants
rng_seed = 2023  # random seed for reproducibility
randSeed = 123

# Utility functions
printdf = lambda *args, **kwargs: print(pd.DataFrame(*args, **kwargs))


class PriorGraphNetwork:
    """:) Please note that this class focuses on incorporating information from a prior network (in our case, 
    a biological network of some sort). The input would be an edge list with: source, target, weight. If no
    weights are given then a weight of 1 will be automatically assumed. 
    If the prior network is NOT symmetric (most likely directed):
    please note we can use graph embedding techniques like weighted node2vec (on the directed graph) to generate
    an embedding, find the cosine similarity, and then use the node-node similarity values for our network.
    Ultimately, this class builds the W matrix (for the prior network weights to be used for our network 
    regularization penalty), the D matrix (of degrees), and the V matrix (custom for our approach)."""  
    
    _parameter_constraints = {
        "w_transform_for_d": ["none", "sqrt", "square"],
        "degree_pseudocount": (0, None),
        "default_edge_weight": (0, None),
        "threshold_for_degree": (0, None),   
        "view_network":[True, False],
        "verbose":[True, False]}
    
    def __init__(self, **kwargs): # define default values for constants

        self.edge_values_for_degree = False # we instead consider a threshold by default (for counting edges into our degrees)
        self.consider_self_loops = False # no self loops considered
        self.verbose = True # printing out statements
        self.pseudocount_for_degree = 1e-3 # to ensure that we do not have any 0 degrees for any node in our matrix.
        self.undirected_graph_bool = True # by default we assume the input network is undirected and symmetric :)
        self.default_edge_weight = 0.1 # if an edge is missing weight information   
        # these are the nodes we may wish to include. If these are provided, then we may utilize these in our model. 
        self.gene_expression_nodes = [] # default if we use edge weights for degree:
        # if edge_values_for_degree is True: we can use the edge weight values to get the degrees. 
        self.w_transform_for_d = "none"
        #self.square_root_weights_for_degree = False # take the square root of the edge weights for the degree calculations
        #self.squaring_weights_for_degree = False # square the edge weights for the degree calculations
        # default if we use a threshold for the degree:
        self.threshold_for_degree = 0.5
        self.view_network = False
        ####################
#         self.dimensions = 64
#         self.walk_length = 30
#         self.num_walks = 200
#         self.p = 1
#         self.q = 0.5
#         self.workers = 4
#         self.window = 10
#         self.min_count = 1
#         self.batch_words = 4
        self.node_color_name = "yellow"
        self.added_node_color_name = "lightblue"
        self.edge_color_name = "red"
        self.edge_weight_scaling = 5
        self.debug = False
        ####################
        self.preprocessed_network = False # is the network preprocessed with the final gene expression nodes
        self.__dict__.update(kwargs) # overwrite with any user arguments :)
        required_keys = ["edge_list"] # if consider_self_loops is true, we add 1 to degree value for each node, 
        # check that all required keys are present:
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note since edge_values_for_degree = {self.edge_values_for_degree} ye are missing information for these keys: {missing_keys}")
        self.network_nodes = self.network_nodes_from_edge_list()
        # other defined results:
        # added Aug 30th:
        if isinstance(self.edge_list, pd.DataFrame):
            print(":( Please input edgelist as a list of lists instead of a dataframe. For your edge_df, try: edge_df.values.tolist()")
            #self.edge_list = self.edge_list.values.tolist()
        self.original_edge_list = self.edge_list
        if len(self.gene_expression_nodes) > 0: # is not None:
            self.preprocessed_network = True
            self.gene_expression_nodes.sort()
            gene_expression_nodes = self.gene_expression_nodes
            self.final_nodes = gene_expression_nodes
            common_nodes = ef.intersection(self.network_nodes, self.gene_expression_nodes)
            common_nodes.sort()
            self.common_nodes = common_nodes
            self.gex_nodes_to_add = list(set(self.gene_expression_nodes) - set(self.common_nodes))
            self.network_nodes_to_remove = list(set(self.network_nodes) - set(self.common_nodes))
            # filtering the edge_list:
            self.edge_list = [edge for edge in self.original_edge_list if edge[0] in gene_expression_nodes and edge[1] in gene_expression_nodes]
        else:
            self.final_nodes = self.network_nodes
        if self.verbose:
            print(self.final_nodes)
        self.tf_names_list = self.final_nodes
        self.nodes = self.final_nodes
        self.N = len(self.tf_names_list)      
        self.V = self.create_V_matrix()
        if self.undirected_graph_bool:
            self.directed=False
            self.undirected_edge_list_to_matrix()
            self.W_original = self.W 
            #self.edge_df = self.undirected_edge_list_updated().drop_duplicates()
        else:
            self.directed=True
            self.W_original = self.directed_node2vec_similarity(self.edge_list, self.dimensions,
                                                     self.walk_length, self.num_walks,
                                                     self.p, self.q, self.workers,
                                                     self.window, self.min_count, self.batch_words)  
            self.W = self.generate_symmetric_weight_matrix()
            self.W_df = pd.DataFrame(self.W, columns = self.nodes, index = self.nodes)
        if self.view_network:
            self.view_W_network = self.view_W_network()
        else:
            self.view_W_network = None
        self.degree_vector = self.generate_degree_vector_from_weight_matrix()
        self.D = self.generate_degree_matrix_from_weight_matrix()
        # added on April 26, 2023
        degree_df = pd.DataFrame(self.final_nodes, columns = ["TF"])
        degree_df["degree_D"] = self.D  * np.ones(self.N)
        self.inv_sqrt_degree_df = degree_df ########
        self.edge_list_from_W = self.return_W_edge_list()
        self.A = self.create_A_matrix()
        self.A_df = pd.DataFrame(self.A, columns = self.nodes, index = self.nodes, dtype=np.float64)
        self.param_lists = self.full_lists()
        self.param_df = pd.DataFrame(self.full_lists(), columns = ["parameter", "data type", "description", "value", "class"])
        self.node_status_df = self.find_node_status_df()
        self._apply_parameter_constraints()
        
        
    def find_node_status_df(self):
        """ Returns the node status """
        preprocessed_result = "No :("
        if self.preprocessed_network:
            preprocessed_result = "Yes :)"
        if self.preprocessed_network:
            common_df = pd.DataFrame(self.common_nodes, columns = ["node"])
            common_df["preprocessed"] = preprocessed_result
            common_df["status"] = "keep :)"
            common_df["info"] = "Common Node (Network and Gene Expression)"
            full_df = common_df
            if len(self.gex_nodes_to_add) > 0:
                gex_add_df = pd.DataFrame(self.gex_nodes_to_add, columns = ["node"])
                gex_add_df["preprocessed"] = preprocessed_result
                gex_add_df["status"] = "keep :)"
                gex_add_df["info"] = "Gene Expression Node Only"
                full_df = pd.concat([common_df, gex_add_df])
            if len(self.network_nodes_to_remove) > 0:
                net_remove_df = pd.DataFrame(self.network_nodes_to_remove, columns = ["node"])
                net_remove_df["preprocessed"] = preprocessed_result
                net_remove_df["status"] = "remove :("
                net_remove_df["info"] = "Network Node Only"
                full_df = pd.concat([full_df, net_remove_df])
        else:
            full_df = pd.DataFrame(self.network_nodes, columns = ["node"])
            full_df["preprocessed"] = preprocessed_result
            full_df["status"] = 'unknown :|'
            full_df["info"] = "Original Network Node"
        return full_df
        
        
    def network_nodes_from_edge_list(self):
        edge_list = self.edge_list
        network_nodes = list({node for edge in edge_list for node in edge[:2]})
        network_nodes.sort()
        return network_nodes
        
        
    def _apply_parameter_constraints(self):
        constraints = {**PriorGraphNetwork._parameter_constraints}
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
        
        
    def create_V_matrix(self):
        V = self.N * np.eye(self.N) - np.ones(self.N)
        return V
    
    
    
        # Optimized functions
    def preprocess_edge_list(self):
        processed_edge_list = []
        default_edge_weight = self.default_edge_weight

        for sublst in self.edge_list:
            if len(sublst) == 2:
                processed_edge_list.append(sublst + [default_edge_weight])
            else:
                processed_edge_list.append(sublst)

        return processed_edge_list

    def undirected_edge_list_to_matrix(self):
        all_nodes = self.final_nodes
        edge_list = self.preprocess_edge_list()
        default_edge_weight = self.default_edge_weight
        N = len(all_nodes)
        self.N = N
        weight_df = np.full((N, N), default_edge_weight)

        # Create a mapping from node to index
        node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}

        for edge in tqdm(edge_list) if self.verbose else edge_list:
            try:
                source, target, *weight = edge
                weight = weight[0] if weight else default_edge_weight
                weight = np.nan_to_num(weight, nan=default_edge_weight)
                source_idx, target_idx = node_to_idx[source], node_to_idx[target]
                weight_df[source_idx, target_idx] = weight
                weight_df[target_idx, source_idx] = weight
            except ValueError as e:
                print(f"An error occurred: {e}")
                continue

        np.fill_diagonal(weight_df, 0)
        W = weight_df
        np.fill_diagonal(W, (W.sum(axis=0) - W.diagonal()) / (N - 1))

        if not ef.check_symmetric(W):
            print(":( W matrix is NOT symmetric")

        self.W = W
        self.W_df = pd.DataFrame(W, columns=all_nodes, index=self.final_nodes, dtype=np.float64)
        return self
    
    
    def generate_symmetric_weight_matrix(self) -> np.ndarray:
        """generate symmetric W matrix. W matrix (Symmetric --> W = W_Transpose).
        Note: each diagonal element is the summation of other non-diagonal elements in the same row divided by (N-1)
        2023.02.14_Xiang. TODO: add parameter descriptions"""
        W = self.W_original
        np.fill_diagonal(W, (W.sum(axis=0) - W.diagonal()) / (self.N - 1))
        symmetric_W = ef.check_symmetric(W)
        if symmetric_W == False:
            print(":( W matrix is NOT symmetric")
            return None
        return W
    
    
    def return_W_edge_list(self):
        wMat = ef.view_matrix_as_dataframe(self.W, column_names_list = self.tf_names_list, row_names_list = self.tf_names_list)
        w_edgeList = wMat.stack().reset_index()
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns = {"level_0":"source", "level_1":"target", 0:"weight"})
        w_edgeList = w_edgeList.sort_values(by = ["weight"], ascending = False)
        return w_edgeList

    
    def view_W_network(self):
        roundedW = np.round(self.W, decimals=4)
        wMat = ef.view_matrix_as_dataframe(roundedW, column_names_list=self.nodes, row_names_list=self.nodes)
        w_edgeList = wMat.stack().reset_index()
        # https://stackoverflow.com/questions/48218455/how-to-create-an-edge-list-dataframe-from-a-adjacency-matrix-in-python
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns={"level_0": "source", "level_1": "target", 0: "weight"})
        w_edgeList = w_edgeList[w_edgeList["weight"] != 0]
        
        G = nx.from_pandas_edgelist(w_edgeList, source="source", target="target", edge_attr="weight")
        pos = nx.spring_layout(G)
        weights_list = [G.edges[e]['weight'] * self.edge_weight_scaling for e in G.edges]
        fig, ax = plt.subplots()
        if self.preprocessed_network and len(self.gex_nodes_to_add) > 0:
            new_nodes = self.gex_nodes_to_add
            print("new nodes:", new_nodes)
            node_color_map = {node: self.added_node_color_name if node in new_nodes else self.node_color_name for node in G.nodes}
            nx.draw(G, pos, node_color=node_color_map.values(), edge_color=self.edge_color_name, with_labels=True, width=weights_list, ax=ax)
        else:
            nx.draw(G, pos, node_color=self.node_color_name, edge_color=self.edge_color_name, with_labels=True, width=weights_list, ax=ax)

        labels = {e: G.edges[e]['weight'] for e in G.edges}
        return nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)

    
    def generate_degree_vector_from_weight_matrix(self) -> np.ndarray:
        """generate d degree vector.  2023.02.14_Xiang TODO: add parameter descriptions
        """
        if self.edge_values_for_degree == False:
            W_bool = (self.W > self.threshold_for_degree)
            d = np.float64(W_bool.sum(axis=0) - W_bool.diagonal())
        else: 
            if self.w_transform_for_d == "sqrt": #self.square_root_weights_for_degree: # taking the square root of the weights for the edges
                W_to_use = np.sqrt(self.W)
            elif self.w_transform_for_d == "square": # self.squaring_weights_for_degree:
                W_to_use = self.W ** 2
            else:
                W_to_use = self.W
            d = W_to_use.diagonal() * (self.N - 1) # summing the edge weights
        d += self.pseudocount_for_degree
        if self.consider_self_loops:
            d += 1 # we also add in a self-loop :)    
        # otherwise, we can just use this threshold for the degree
        if self.verbose:
            print(":) Please note: we are generating the prior network:")
            if self.edge_values_for_degree:
                print(":) Please note that we use the sum of the edge weight values to get the degree for a given node.")
            else:
                print(f":) Please note that we count the number of edges with weight > {self.threshold_for_degree} to get the degree for a given node.")
            if self.consider_self_loops:
                print(f":) Please note that since consider_self_loops = {self.consider_self_loops} we also add 1 to the degree for each node (as a self-loop).")
            print(f":) We also add {self.pseudocount_for_degree} as a pseudocount to our degree value for each node.")
            print() #
        return d

    
    def generate_degree_matrix_from_weight_matrix(self): # D matrix
        """:) Please note that this function returns the D matrix as a diagonal matrix 
        where the entries are 1/sqrt(d). Here, d is a vector corresponding to the degree of each matrix"""
        # we see that the D matrix is higher for nodes that are singletons, a much higher value because it is not connected
        d = self.degree_vector
        d_inv_sqrt = 1 / np.sqrt(d)
        # D = np.diag(d_inv_sqrt)  # full matrix D, only suitable for small scale. Use DiagonalLinearOperator instead.
        D = ef.DiagonalLinearOperator(d_inv_sqrt)
        return D
    
    
    def create_A_matrix(self): # A matrix
        """ Please note that this function by Saniya creates the A matrix, which is:   
        :) here: %*% refers to matrix multiplication
        and * refers to element-wise multiplication (for 2 dataframes with same exact dimensions,
        component-wise multiplication)
        # Please note that this function by Saniya creates the A matrix, which is:
        # (D_transpose) %*% (V*W) %*% (D)        
        """
        A = self.D @ (self.V * self.W) @ self.D
        approxSame = ef.check_symmetric(A) # please see if A is symmetric
        if approxSame:
            return A
        else:
            print(f":( False. A is NOT a symmetric matrix.")
            print(A)
        return False
    
    
    def full_lists(self):
            # network arguments used:
            # argument, description, our value
        full_lists = []
        term_to_add_last = "PriorGraphNetwork"
        row1 = ["default_edge_w", ">= 0", "edge weight for any edge with missing weight info", self.default_edge_weight, term_to_add_last]
        row2 = ["self_loops", "boolean", "add 1 to the degree for each node (based on self-loops)?", self.consider_self_loops, term_to_add_last]

        full_lists.append(row1)
        full_lists.append(row2)
        if self.pseudocount_for_degree != 0:
            row3 = ["d_pseudocount", ">= 0",
                    "to ensure that no nodes have 0 degree value in D matrix", 
                    self.pseudocount_for_degree, term_to_add_last]
            full_lists.append(row3)
        if self.edge_values_for_degree:
            row_to_add = ["edge_vals_for_d", "boolean",
                          "if True, we use the edge weight values to derive our degrees for matrix D", True, term_to_add_last]
            full_lists.append(row_to_add)# arguments to add in:
            if self.w_transform_for_d == "sqrt": # take the square root of the edge weights for the degree calculations
                row_to_add = ["w_transform_for_d: sqrt", "string",
                              "for each edge, we use the square root of the edge weight values to derive our degrees for matrix D", self.w_transform_for_d, term_to_add_last]
                full_lists.append(row_to_add)        
            if self.w_transform_for_d == "square":  # square the edge weights for the degree calculations
                row_to_add = ["w_transform_for_d: square", "string",
                              "for each edge, we square the edge weight values to derive our degrees for matrix D", self.w_transform_for_d, term_to_add_last]
                full_lists.append(row_to_add)    
        else: # default if we use a threshold for the degree:
            row_to_add = ["edge_vals_for_d", "boolean",
                          "if False, we use a threshold instead to derive our degrees for matrix D", False, term_to_add_last]
            full_lists.append(row_to_add)
            self.threshold_for_degree = 0.5 # edge weights > this threshold are counted as 1 for the degree
            to_add_text = "edge weights > " + str(self.threshold_for_degree) + " are counted as 1 for the degree"
            row_to_add = ["thresh_for_d", ">= 0",
                          to_add_text, self.threshold_for_degree, term_to_add_last]
            full_lists.append(row_to_add)
        return full_lists
        
        
def build_prior_network(edge_list, gene_expression_nodes = [], default_edge_weight = 0.1,
                  degree_threshold = 0.5,
                  degree_pseudocount = 1e-3, 
                  view_network = True,
                  verbose = True):
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
                       "view_network": view_network,
                       "verbose":verbose}
    if verbose:
        print("building prior network:")
        print("prior graph network used")
    netty = PriorGraphNetwork(**prior_graph_dict) # uses the network to get features like the A matrix. ####################
    return netty


def directed_node2vec_similarity(edge_list: List[Tuple[int, int, float]],
                                                 dimensions: int = 64,
                                                 walk_length: int = 30,
                                                 num_walks: int = 200,
                                                 p: float = 1, q: float = 0.5,
                                                 workers: int = 4, window: int = 10,
                                                 min_count: int = 1,
                                                 batch_words: int = 4) -> np.ndarray:
    print("directed_node2vec_similarity")
    """ Given an edge list and node2vec parameters, returns a scaled similarity matrix for the node embeddings generated
    by training a node2vec model on the directed graph defined by the edge list.

    Parameters:
    -----------
    edge_list: List[List[int, int, float]]
        A list of lists representing the edges of a directed graph. Each edge should be a list of three values:
        [source_node, target_node, edge_weight]. If no edge weight is specified, it is assumed to be 1.0.

    dimensions: int, optional (default=64)
        The dimensionality of the node embeddings.

    walk_length: int, optional (default=30)
        The length of each random walk during the node2vec training process.

    num_walks: int, optional (default=200)
        The number of random walks to generate for each node during the node2vec training process.

    p: float, optional (default=1)
        The return parameter for the node2vec algorithm.

    q: float, optional (default=0.5)
        The in-out parameter for the node2vec algorithm.

    workers: int, optional (default=4)
        The number of worker threads to use during the node2vec training process.

    window: int, optional (default=10)
        The size of the window for the skip-gram model during training.

    min_count: int, optional (default=1)
        The minimum count for a word in the training data to be included in the model.

    batch_words: int, optional (default=4)
        The number of words in each batch during training.

    Returns:
    --------
    scaled_similarity_matrix: np.ndarray
        A scaled (0-1 range) cosine similarity matrix for the node embeddings generated by training a node2vec model
        on the directed graph defined by the edge list.
    """
    print("Creating directed graph from edge list")
    directed_graph = nx.DiGraph()
    for edge in edge_list:
        source, target = edge[:2]
        weight = edge[2] if len(edge) == 3 else 1.0
        directed_graph.add_edge(source, target, weight=weight)
    
    # Extract unique node names from the graph
    node_names = list(directed_graph.nodes)
    
    print("Initializing the Node2Vec model")
    model = Node2Vec(directed_graph, dimensions=dimensions, walk_length=walk_length, 
                     num_walks=num_walks, p=p, q=q, workers=workers)
    
    print("Training the model")
    model = model.fit(window=window, min_count=min_count, batch_words=batch_words)
    
    print("Getting node embeddings")
    node_embeddings = np.array([model.wv[node] for node in node_names])
    
    print("Calculating cosine similarity matrix")
    similarity_matrix = cosine_similarity(node_embeddings)
    
    print("Scaling similarity matrix to 0-1 range")
    scaled_similarity_matrix = (similarity_matrix + 1) / 2
    
    # Create a DataFrame with rows and columns labeled as node names
    similarity_matrix = pd.DataFrame(scaled_similarity_matrix, index=node_names, columns=node_names)
    print(f":) First 5 entries of the symmetric similarity matrix for {similarity_matrix.shape[0]} nodes.")
    print(similarity_matrix.iloc[0:5, 0:5])
    
    similarity_df = similarity_matrix.reset_index().melt(id_vars='index', var_name='TF2', value_name='cosine_similarity')
    #similarity_df = similarity_df[similarity_df['index'] < similarity_df['TF2']]
    similarity_df = similarity_df.rename(columns = {"index":"node_1", "TF2":"node_2"})
    similarity_df = similarity_df[similarity_df["node_1"] != similarity_df["node_2"]]
    results_dict = {}
    print("\n :) ######################################################## \n")
    print(":) Please note that we return a dictionary with 3 keys based on Node2Vec and cosine similarity computations:")
    print("1. similarity_matrix: the cosine similarity matrix for the nodes in the original directed graph")
    results_dict["similarity_matrix"] = similarity_matrix
    print("2. similarity_df: simplified dataframe of the cosine similarity values from the similarity_matrix.")

    results_dict["similarity_df"] = similarity_df
    print("3. NetREm_edgelist: an edge_list that is based on similarity_df that is ready to be input for NetREm.")

    results_dict["NetREm_edgelist"] = similarity_df.values.tolist()
    print(results_dict.keys())
    return results_dict