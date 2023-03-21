# By: Saniya Khullar, Xiang Huang, Daifeng Wang
# University of Wisconsin - Madison
# GRegulNetwork

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import os
import sys # https://www.dev2qa.com/how-to-run-python-script-py-file-in-jupyter-notebook-ipynb-file-and-ipython/#:~:text=How%20To%20Run%20Python%20Script%20.py%20File%20In,2.%20Invoke%20Python%20Script%20File%20From%20Ipython%20Command-Line.
import networkx as nx
import scipy
from scipy.linalg import svd as robust_svd
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV, LinearRegression, Ridge
from numpy.typing import ArrayLike
import plotly.express as px
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from scipy.sparse.linalg.interface import LinearOperator
import warnings
from sklearn.exceptions import ConvergenceWarning
printdf = lambda *args, **kwargs: print(pd.DataFrame(*args, **kwargs))
rng_seed = 2023 # random seed for reproducibility
randSeed = 123

"""
Optimization for
(1 / (2 * M)) * ||y - Xc||^2_2  +  (beta / (2 * N^2)) * c'Ac  +  alpha * ||c||_1
Which is converted to lasso
(1 / (2 * M)) * ||y_tilde - X_tilde @ c||^2_2   +  alpha * ||c||_1
where M = n_samples and N is the dimension of c.
Check compute_X_tilde_y_tilde() to see how we make sure above normalization is applied using Lasso of sklearn
"""

class DiagonalLinearOperator(LinearOperator):
    """Construct a diagonal matrix as a linear operator instead a full numerical matirx np.diag(d).
    This saves memory and computation time which is especially useful when d is huge.
    D.T = D
    For 2d matrix A:
    D @ A = d[:, np.newwaxis]* A  # scales rows of A
    A @ D =  A * d[np.newaxis, :]  # scales cols of A

    For 1d vector v:
    D @ v = d * v
    v @ D = v * d

    NOTE: Coding just for fun: using a numerical matrix or a sparse matrix maybe just fine for network regularization.
    By Xiang Huang
    """
    def __init__(self, d):
        """d is a 1d vector of dimension N"""
        N = len(d)
        self.d = d
        super().__init__(dtype=None, shape=(N, N))

    def _transpose(self):
        return self

    def _matvec(self, v):
        return self.d * v

    def _matmat(self, A):
        return self.d[:, np.newaxis] * A

    def __rmatmul__(self, x):
        """Implmentation of A @ D, and x @ D
        We could implment __matmul__ in a similar way without inheriting LinearOperator
        Because we inherit from LinearOperator, we can implment _matvec, and _matmat instead.
        """
        if x.ndim == 2:
            return x * self.d[np.newaxis, :]
        elif x.ndim == 1:
            return x * self.d
        else:
            raise ValueError(f'Array should be 1d or 2d, but it is {x.ndim}d')
    # Generally A @ D will call A.__matmul__(D) which raises a ValueError and not a NotImplemented
    # We need to set __array_priority__ to high value higher than 0 (np.array) and 10.1 (scipy.sparse.csr_matrix)
    # https://github.com/numpy/numpy/issues/8155
    # https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 1000

    
# create X and Y matrix
class DemoDataBuilderXandY: 
    #Build_X_and_Y_demo_data:
    """:) Please note that this class focuses on building Y data based on a normal distribution (specified mean
    and standard deviation). M is the # of samples we want to generate. Thus, Y is a vector with M elements. 
    Then, this class returns X for a set of N predictors (each with M # of samples) based on a list of N correlation
    values. For instance, if N = 5 predictors (the Transcription Factors (TFs)), we have [X1, X2, X3, X4, X5],
    and a respective list of correlation values: [cor(X1, Y), cor(X2, Y), cor(X3, Y), cor(X4, Y), cor(X5, Y)].
    Then, this class will generate X, a matrix of those 5 predictors (based on similar distribution as Y) 
    with these respective correlations."""    
    
    def __init__(self, **kwargs):
        # define default values for constants
        self.same_train_test_data = False
        self.same_train_and_test_data_bool = self.same_train_test_data
        self.test_data_percent = 30
        self._mu = 0
        self._sd = 1
        self._num_iters_for_generating_X = 100
        self._rng_seed = 2023 # for Y
        self._randSeed = 123 # for X
        self.orthogonal_X_bool = False
        self.ortho_scalar = 10
        self.view_input_correlations_plot = False
        # reading in user inputs
        self.__dict__.update(kwargs)
        # check that all required keys are present:
        required_keys = ["corrVals", "num_samples_M"]
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        self.M = self.num_samples_M
        self.N = self.get_N()
        self.y = self.generate_Y()
        self.X = self.generate_X()
        self.same_train_and_test_data_bool = self.same_train_test_data
        if self.same_train_and_test_data_bool:
            self.testing_size = 1
        else:
            self.testing_size = (self.test_data_percent/100.0)
        data_sets = self.generate_training_and_testing_data() # [X_train, X_test, y_train, y_test]
        self.X_train = data_sets[0]
        self.X_test = data_sets[1]
        self.y_train = data_sets[2]
        self.y_test = data_sets[3]
        
        self.tf_names_list = self.get_tf_names_list()
        self.corr_df = self.return_correlations_dataframe()
        self.combined_correlations_df = self.get_combined_correlations_df()
        if self.view_input_correlations_plot:
            self.view_input_correlations = self.view_input_correlations()
        
    def get_tf_names_list(self):
        tf_names_list = []
        for i in range(0, self.N):
            term = "TF" + str(i)
            tf_names_list.append(term)
        return tf_names_list
    
    # getter method
    def get_N(self):
        N = len(self.corrVals)
        return N 
    
    def return_correlations_dataframe(self):
        #N = len(tf_names_list)
        corr_info = ["expected_correlations"] * self.N
        import pandas as pd
        corr_df = pd.DataFrame(corr_info, columns = ["info"])
        corr_df["TF"] = self.tf_names_list
        corr_df["value"] = self.corrVals
        corr_df["data"] = "correlations"
        return corr_df
    
    def generate_Y(self):
        seed_val = self._rng_seed
        import numpy as np
        rng = np.random.default_rng(seed=seed_val)
        y = rng.normal(self._mu, self._sd, self.M)
        return y
    
        # Check if Q is orthogonal using the is_orthogonal function
    def is_orthogonal(matrix):
        import numpy as np
        """
        Checks if a given matrix is orthogonal.

        Parameters:
        matrix (numpy.ndarray): The matrix to check

        Returns:
        bool: True if the matrix is orthogonal, False otherwise.
        """
        # Compute the transpose of the matrix
        matrix_T = matrix.T

        # Compute the product of the matrix and its transpose
        matrix_matrix_T = np.dot(matrix, matrix_T)

        # Check if the product is equal to the identity matrix
        return np.allclose(matrix_matrix_T, np.eye(matrix.shape[0]))


    def generate_X(self):
        """Generates a design matrix X with the given correlations.
        
        Parameters:
        orthogonal (bool): Whether to generate an orthogonal matrix (default=False).
        
        Returns:
        numpy.ndarray: The design matrix X.
        """
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
       
        
        np.random.seed(self._randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N # len(corrVals)
        numIterations = self._num_iters_for_generating_X
        correlations = self.corrVals
        corrVals = [correlations[0]] + correlations
        e = np.random.normal(0, 1, (n, numTFs + 1))
        X = np.copy(e)
        X[:, 0] = y * np.sqrt(1.0 - corrVals[0]**2) / np.sqrt(1.0 - np.corrcoef(y, X[:,0])[0,1]**2)
        for j in range(numIterations):
            for i in range(1, numTFs + 1):
                corr = np.corrcoef(y, X[:, i])[0, 1]
                X[:, i] = X[:, i] + (corrVals[i] - corr) * y
        
        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X)[0]
            Q = scalar * Q
            return Q[:, 1:]
        else:
            # Return the X matrix without orthogonalization
            return X[:, 1:]
       
    
    def generate_training_and_testing_data(self):
        same_train_and_test_data_bool = self.same_train_and_test_data_bool 
        from sklearn.model_selection import train_test_split
        X = self.X
        y = self.y
        if same_train_and_test_data_bool == False: # different training and testing datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.testing_size)
            print(f":) Please note that since we hold out {self.testing_size * 100.0}% of our {self.M} samples for testing, we have:")
            print(f":) X_train = {X_train.shape[0]} rows (samples) and {X_train.shape[1]} columns (N = {self.N} predictors) for training.")
            print(f":) X_test = {X_test.shape[0]} rows (samples) and {X_test.shape[1]} columns (N = {self.N} predictors) for testing.")
            print(f":) y_train = {y_train.shape[0]} corresponding rows (samples) for training.")
            print(f":) y_test = {y_test.shape[0]} corresponding rows (samples) for testing.")
        else: # training and testing datasets are the same :)
            X_train, X_test, y_train, y_test = X, X, y, y
            y_train = y
            y_test = y_train
            print(f":) Please note that since we use the same data for training and for testing :) of our {self.M} samples. Thus, we have:")
            print(f":) X_train = X_test = {X_train.shape[0]} rows (samples) and {X_train.shape[1]} columns (N = {self.N} predictors) for training and for testing")
            print(f":) y_train = y_test = {y_train.shape[0]} corresponding rows (samples) for training and for testing.")    
        return [X_train, X_test, y_train, y_test]
    
    
    def get_combined_correlations_df(self):
        combined_correlations_df = self.actual_vs_expected_corrs_DefensiveProgramming_all_groups(self.X, self.y, 
                                                                                            self.X_train, 
                                                                                            self.y_train,
                                                                                        self.X_test, 
                                                                                            self.y_test,
                                                                                self.corrVals, 
                                                                                self.tf_names_list, 
                                                                             self.same_train_and_test_data_bool)
        return combined_correlations_df
    
    def actual_vs_expected_corrs_DefensiveProgramming_all_groups(self, X, y, X_train, y_train, X_test, y_test,
                                                                corrVals, tf_names_list,
                                                                 same_train_and_test_data_bool):
        import pandas as pd
        from tqdm import tqdm
        import numpy as np
        overall_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X, y, corrVals, 
                                                                                         tf_names_list, same_train_and_test_data_bool, "Overall")
        training_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X_train, y_train, corrVals, 
                                                                                          tf_names_list, same_train_and_test_data_bool, "Training")
        testing_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X_test, y_test, corrVals, 
                                                                                         tf_names_list, same_train_and_test_data_bool, "Testing")
        combined_correlations_df = pd.concat([overall_corrs_df, training_corrs_df, testing_corrs_df]).drop_duplicates()
        return combined_correlations_df
    
    def compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(self, X_matrix, y, corrVals, 
                                                              predictor_names_list,
                                                              same_train_and_test_data_boolean,
                                                              data_type):
        # please note that this function by Saniya ensures that the actual and expected correlations are close
        # so that the simulation has the x-y correlations we were hoping for in corrVals
        import pandas as pd
        from tqdm import tqdm
        import numpy as np
        updatedDF = pd.DataFrame(X_matrix)#.shape
        actualCorrsList = []
        for i in tqdm(range(0, len(corrVals))):
            expectedCor = corrVals[i]
            actualCor = np.corrcoef(updatedDF[i], y)[0][1]
            difference = abs(expectedCor - actualCor)
            predictor_name = predictor_names_list[i]
            actualCorrsList.append([i, predictor_name, expectedCor, actualCor, difference])
        comparisonDF = pd.DataFrame(actualCorrsList, columns = ["i", "predictor", "expected_corr_with_Y", "actual_corr", "difference"])
        comparisonDF["X_group"] = data_type
        num_samples = X_matrix.shape[0]
        if same_train_and_test_data_boolean:
            comparisonDF["num_samples"] = "same " + str(num_samples)
        else:
            comparisonDF["num_samples"] = "unique " + str(num_samples)
        return comparisonDF

        # Visualizing Functions :)
    def view_input_correlations(self):
        corr_val_df = pd.DataFrame(self.corrVals, columns = ["correlation"])#.transpose()
        corr_val_df.index = self.tf_names_list
        corr_val_df["TF"] = self.tf_names_list
        import plotly.express as px
        fig = px.bar(corr_val_df, x='TF', y='correlation',  barmode='group')
        fig.show()
        return fig
    
    
def view_matrix_as_dataframe(matrix, column_names_list = [], row_names_list = []):
    # :) Please note this function by Saniya returns a dataframe representation of the numpy matrix
    # optional are the names of the columns and names of the rows (indices)
    import pandas as pd
    matDF = pd.DataFrame(matrix)
    if len(column_names_list) == matDF.shape[1]: 
        matDF.columns = column_names_list
    if len(row_names_list) == matDF.shape[0]: 
        matDF.index = row_names_list    
    return matDF

def check_symmetric(a, rtol=1e-05, atol=1e-08):
            # https://stackoverflow.com/questions/42908334/checking-if-a-matrix-is-symmetric-in-numpy
            # Please note that this function checks if a matrix is symmetric in Python
            # for square matrices (same # of rows and columns), there is a possiblity they may be symmetric
            # returns True if the matrix is symmetric (matrix = matrix_tranpose)
            # returns False if the matrix is NOT symmetric
    import numpy as np
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

# create W and D and V matrices
class PriorGraphNetwork:
    """:) Please note that this class focuses on incorporating information from a prior network (in our case, 
    a biological network of some sort). The input would be an edge list with: source, target, weight. If no
    weights are given then a weight of 1 will be automatically assumed. 
    
    If the prior network is NOT symmetric (most likely directed):
    please note we can use graph embedding techniques like weighted node2vec (on the directed graph) to generate
    an embedding, find the cosine similarity, and then use the node-node similarity values for our network.
    
    Ultimately, this class builds the W matrix (for the prior network weights to be used for our network 
    regularization penalty), the D matrix (of degrees), and the V matrix (custom for our approach). """  
    
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from node2vec import Node2Vec
    import networkx as nx
    from typing import List, Tuple
    def __init__(self, **kwargs): # define default values for constants
        self.edge_values_for_degree = False # we instead consider a threshold by default (for counting edges into our degrees)
        self.consider_self_loops = False # no self loops considered
        
        self.pseudocount_for_degree = 1e-3 # to ensure that we do not have any 0 degrees for any node in our matrix.
        self.undirected_graph_bool = True # by default we assume the input network is undirected and symmetric :)
        self.default_edge_weight = 0.1 # if an edge is missing weight information   
        
        # default if we use edge weights for degree:
        # if edge_values_for_degree is True: we can use the edge weight values to get the degrees. 
        self.square_root_weights_for_degree = False # take the square root of the edge weights for the degree calculations
        self.squaring_weights_for_degree = False # square the edge weights for the degree calculations
        # default if we use a threshold for the degree:
        self.threshold_for_degree = 0.5
        
        ####################
        self.dimensions = 64
        self.walk_length = 30
        self.num_walks = 200
        self.p = 1
        self.q = 0.5
        self.workers = 4
        self.window = 10
        self.min_count = 1
        self.batch_words = 4
        self.node_color_name = "yellow"
        self.edge_color_name = "red"
        self.edge_weight_scaling = 5
        self.debug = False
        
        self.__dict__.update(kwargs) # overwrite with any user arguments :)
        required_keys = ["edge_list"]  
        # if consider_self_loops is true, we add 1 to degree value for each node, 
        # so each node has some degree and connection with itself. 
#         if self.edge_values_for_degree: # instead, we can use the edge weight values to get the degrees. 
#             # check that all required keys are present:
#             required_keys = required_keys + ["square_root_weights_bool"]
#         else: # we are using a threshold:
#             required_keys = required_keys + ["threshold_for_degree"] # otherwise, we can just use this threshold for the degree. We would then not use threshold_for_degree
        self.tf_names_list = self.retrieve_tf_names_list()
        self.N = len(self.tf_names_list)
        # check that all required keys are present:
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note since edge_values_for_degree = {self.edge_values_for_degree} ye are missing information for these keys: {missing_keys}")
        # other defined results:
        
        self.V = self.create_V_matrix()
        if self.undirected_graph_bool:
            self.directed=False
            self.W_original = self.undirected_edge_list_to_matrix()#self.edge_list, self.default_weight)
            self.edge_df = self.undirected_edge_list_updated()
        else:
            self.directed=True
            self.W_original = self.directed_node2vec_similarity(self.edge_list,
                                                     self.dimensions,
                                                     self.walk_length,
                                                     self.num_walks,
                                                     self.p, self.q, self.workers,
                                                     self.window,
                                                     self.min_count,
                                                     self.batch_words)  
        self.W = self.generate_symmetric_weight_matrix()
        self.view_W_network = self.view_W_network()
        self.degree_vector = self.generate_degree_vector_from_weight_matrix()
        self.D = self.generate_degree_matrix_from_weight_matrix()
        self.edge_list_from_W = self.return_W_edge_list()
        self.A = self.create_A_matrix()
        self.param_lists = self.full_lists()
        import pandas as pd
        self.param_df = pd.DataFrame(self.full_lists(), columns = ["parameter", "data type", "description", "value", "class"])
        
        
    def retrieve_tf_names_list(self):
        import pandas as pd
        edges_df = pd.DataFrame(self.edge_list)
        all_nodes = list(set(list(edges_df[0]) + list(edges_df[1])))
        all_nodes.sort()
        min_edge = min(all_nodes)
        max_edge = max(all_nodes)
        #N = len(all_nodes)
        tf_names_list = []
        for i in range(0, len(all_nodes)):
            term = "TF" + str(all_nodes[i])
            tf_names_list.append(term)
        return tf_names_list

    def create_V_matrix(self):
        N = self.N
        V = N * np.eye(N) - np.ones(N)
        return V
    
    #  Building out the Weight matrix from the edge lists:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from node2vec import Node2Vec
    import networkx as nx
    from typing import List, Tuple
    def undirected_edge_list_to_matrix(self):#, edge_list, default_weight=1):
        #print("undirected_edge_list_to_matrix")
        import numpy as np
        nodes = list(set([edge[0] for edge in self.edge_list] + [edge[1] for edge in self.edge_list]))
        num_nodes = len(nodes)
        weight_matrix = np.zeros((num_nodes, num_nodes))
        for edge in self.edge_list:
            source = nodes.index(edge[0])
            target = nodes.index(edge[1])
            weight = edge[2] if len(edge) == 3 else self.default_edge_weight
            weight_matrix[source][target] = weight
            weight_matrix[target][source] = weight
        return weight_matrix
    
    
    def undirected_edge_list_updated(self):
        edge_list = self.edge_list
        p1 = pd.DataFrame(edge_list, columns = ["source", "target", "weight"])
        p2 = pd.DataFrame(edge_list, columns = ["target", "source", "weight"])
        edge_df = pd.concat([p1, p2]).drop_duplicates()
        edge_df = edge_df['weight'].fillna(self.default_edge_weight)
        edge_df["info"] = "undirected_prior_networks"
        return edge_df
        
    
    def generate_symmetric_weight_matrix(self) -> np.ndarray:
        """generate symmetric W matrix. W matrix (Symmetric --> W = W_Transpose).
        Note: each diagonal element is the summation of other non-diagonal elements in the same row divided by (N-1)
        2023.02.14_Xiang. TODO: add parameter descriptions
        """
        # fix diagonal elements
        import numpy as np
        W = self.W_original
        np.fill_diagonal(W, (W.sum(axis=0) - W.diagonal()) / (self.N - 1))
        symmetric_W = check_symmetric(W)
        if symmetric_W == False:
            print(":( W matrix is NOT symmetric")
            return None
        return W
    

    def directed_node2vec_similarity(self, edge_list: List[Tuple[int, int, float]],
                                                     dimensions: int = 64,
                                                     walk_length: int = 30,
                                                     num_walks: int = 200,
                                                     p: float = 1,
                                                     q: float = 0.5,
                                                     workers: int = 4,
                                                     window: int = 10,
                                                     min_count: int = 1,
                                                     batch_words: int = 4) -> np.ndarray:
        print("directed_node2vec_similarity")
        """
        Given an edge list and node2vec parameters, returns a scaled similarity matrix for the node embeddings generated
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
        import networkx as nx
        from node2vec import Node2Vec
        from typing import List, Tuple
        import numpy as np
        # Create directed graph from edge list
        directed_graph = nx.DiGraph()
        for edge in edge_list:
            source, target = edge[:2]
            weight = edge[2] if len(edge) == 3 else 1.0
            directed_graph.add_edge(source, target, weight=weight)

        # Initialize the Node2Vec model
        model = Node2Vec(directed_graph, dimensions=dimensions, walk_length=walk_length, 
                         num_walks=num_walks, p=p, q=q, workers=workers)

        # Train the model
        model = model.fit(window=window, min_count=min_count, batch_words=batch_words)

        # Get node embeddings
        node_embeddings = model.wv.vectors

        # Calculate cosine similarity matrix
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(node_embeddings)

        # Scale similarity matrix to 0-1 range
        scaled_similarity_matrix = (similarity_matrix + 1) / 2

        return scaled_similarity_matrix

    def return_W_edge_list(self):
        wMat = view_matrix_as_dataframe(self.W, column_names_list = self.tf_names_list, row_names_list = self.tf_names_list)
        w_edgeList = wMat.stack().reset_index()
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns = {"level_0":"source", "level_1":"target", 0:"weight"})
        w_edgeList = w_edgeList.sort_values(by = ["weight"], ascending = False)
        return w_edgeList
    
    
    def view_W_network(self):
        roundedW = np.round(self.W, decimals=4)
        wMat = view_matrix_as_dataframe(roundedW, column_names_list = self.tf_names_list, row_names_list = self.tf_names_list)
        w_edgeList = wMat.stack().reset_index()
        # https://stackoverflow.com/questions/48218455/how-to-create-an-edge-list-dataframe-from-a-adjacency-matrix-in-python
        w_edgeList = w_edgeList[w_edgeList["level_0"] != w_edgeList["level_1"]]
        w_edgeList = w_edgeList.rename(columns = {"level_0":"source", "level_1":"target", 0:"weight"})
        w_edgeList = w_edgeList[w_edgeList["weight"] != 0]
        import networkx as nx
        G = nx.from_pandas_edgelist(w_edgeList, source = "source", target = "target", edge_attr = "weight")
        pos = nx.spring_layout(G)
        weights_list = [G.edges[e]['weight'] * self.edge_weight_scaling for e in G.edges]
        nx.draw(G, pos, node_color = self.node_color_name, edge_color = self.edge_color_name, with_labels = True, width = weights_list) #weights)#w_edgeList['weight'])
        labels = {e: G.edges[e]['weight'] for e in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels = labels)
        
    
    def generate_degree_vector_from_weight_matrix(self) -> np.ndarray:
        """generate d degree vector.  2023.02.14_Xiang TODO: add parameter descriptions
        """
        if self.edge_values_for_degree == False:
            W_bool = (self.W > self.threshold_for_degree)
            d = np.float64(W_bool.sum(axis=0) - W_bool.diagonal())
        else: 
            if self.square_root_weights_for_degree: # taking the square root of the weights for the edges
                W_to_use = np.sqrt(self.W)
            elif self.squaring_weights_for_degree:
                W_to_use = self.W ** 2
            else:
                W_to_use = self.W
            d = W_to_use.diagonal() * (self.N - 1) # summing the edge weights
        d += self.pseudocount_for_degree
        if self.consider_self_loops:
            d += 1 # we also add in a self-loop :)    
        # otherwise, we can just use this threshold for the degree
        if self.edge_values_for_degree:
            print(":) Please note that we use the sum of the edge weight values to get the degree for a given node.")
        else:
            print(f":) Please note that we count the number of edges with weight > {self.threshold_for_degree} to get the degree for a given node.")
        if self.consider_self_loops:
            print(f":) Please note that since consider_self_loops = {self.consider_self_loops} we also add 1 to the degree for each node (as a self-loop).")
        print(f":) We also add {self.pseudocount_for_degree} as a pseudocount to our degree value for each node.")
        print() #
        return d

    # D matrix
    def generate_degree_matrix_from_weight_matrix(self):
        """:) Please note that this function returns the D matrix as a diagonal matrix 
        where the entries are 1/sqrt(d). Here, d is a vector corresponding to the degree of each matrix"""
        # we see that the D matrix is higher for nodes that are singletons, a much higher value because it is not connected
        d = self.degree_vector
        d_inv_sqrt = 1 / np.sqrt(d)
        # D = np.diag(d_inv_sqrt)  # full matrix D, only suitable for small scale. Use DiagonalLinearOperator instead.
        # D = DiagonalLinearOperator(d_inv_sqrt)
        D = np.diag(d_inv_sqrt)
        return D
    
    # A matrix
    def create_A_matrix(self):
        # :) here: %*% refers to matrix multiplication
        # and * refers to element-wise multiplication (for 2 dataframes with same exact dimensions,
        # component-wise multiplication)
        # Please note that this function by Saniya creates the A matrix, which is:   
        # :) here: %*% refers to matrix multiplication
        # and * refers to element-wise multiplication (for 2 dataframes with same exact dimensions,
        # component-wise multiplication)
        # Please note that this function by Saniya creates the A matrix, which is:
        # (D_transpose) %*% (V*W) %*% (D)
        VW = self.V * self.W
        D = self.D
        D_transpose = D.T
        Amat = np.matmul(D_transpose, VW)
        Amat = np.matmul(Amat, D) 
        # please see if A is symmetric
        approxSame = check_symmetric(Amat)
        if approxSame:
            #print(":) A is symmetric!")
            return Amat
        else:
            print(f":( False. A is NOT a symmetric matrix.")
            print(Amat)
        return False
    
    
    
    def full_lists(self):
            # network arguments used:
            # argument, description, our value
        full_lists = []
        term_to_add_last = "PriorGraphNetwork"
        row1 = ["default_edge_weight", ">= 0", "edge weight for any edge with missing weight info", self.default_edge_weight, term_to_add_last]
        row2 = ["consider_self_loops", "boolean", "add 1 to the degree for each node (based on self-loops)?", self.consider_self_loops, term_to_add_last]

        full_lists.append(row1)
        full_lists.append(row2)
        if self.pseudocount_for_degree != 0:
            #self.pseudocount_for_degree # to ensure that we do not have any 0 degrees for any node in our matrix.
            row3 = ["pseudocount_for_degree", ">= 0",
                    "to ensure that no nodes have 0 degree value in D matrix", 
                    self.pseudocount_for_degree, term_to_add_last]
            full_lists.append(row3)
        if self.edge_values_for_degree:
            row_to_add = ["edge_values_for_degree", "boolean",
                          "if True, we use the edge weight values to derive our degrees for matrix D", True, term_to_add_last]
            full_lists.append(row_to_add)
            # arguments to add in:
            if self.square_root_weights_for_degree: # take the square root of the edge weights for the degree calculations
                row_to_add = ["square_root_weights_for_degree", "boolean",
                              "for each edge, we use the square root of the edge weight values to derive our degrees for matrix D", True, term_to_add_last]
                full_lists.append(row_to_add)        
            if self.squaring_weights_for_degree:  # square the edge weights for the degree calculations
                row_to_add = ["squaring_weights_for_degree", "boolean",
                              "for each edge, we square the edge weight values to derive our degrees for matrix D", True, term_to_add_last]
                full_lists.append(row_to_add)    
        else: # default if we use a threshold for the degree:
            row_to_add = ["edge_values_for_degree", "boolean",
                          "if False, we use a threshold instead to derive our degrees for matrix D", False, term_to_add_last]
            full_lists.append(row_to_add)
            self.threshold_for_degree = 0.5 # edge weights > this threshold are counted as 1 for the degree
            to_add_text = "edge weights > " + str(self.threshold_for_degree) + " are counted as 1 for the degree"
            row_to_add = ["threshold_for_degree", ">= 0",
                          to_add_text, self.threshold_for_degree, term_to_add_last]
            full_lists.append(row_to_add)
        return full_lists
    
    
# https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/
class GRegulNet:
    """ :) Please note that this class focuses on building a Gene Regulatory Network (GRN)
    from gene expression data for Transcription Factors (TFs), gene expression data 
    for the target gene (TG), and a prior biological network (W). 
    This class performs Network-penalized regression :) """
    
    _parameter_constraints = {
        "alpha_lasso": (0, None),
        "beta_network": (0, None),
        "num_cv_folds": (0, None),
        "fit_y_intercept": [False, True],
        "same_train_and_test_data_bool": [False, True],
        "use_network": [True, False],
        "use_cross_validation_for_model_bool": [False, True],
        "max_lasso_iterations": (1, None),
        "model_type": ["Lasso", "LassoCV", "Linear"]#,
        #"network":(PriorGraphNetwork(), None)
    }
    
    def __init__(self,  **kwargs):# beta_network, alpha_lasso, X_train, y_train, X_test, y_test):
        self.same_train_and_test_data_bool = False # different or same training and testing data?
        self.use_cross_validation_for_model_bool = False
        self.num_cv_folds = 5 # for cross validation models
        self.model_type = "Lasso"
        self.use_network = True
        self.fit_y_intercept = False
        self.max_lasso_iterations = 10000
        self.__dict__.update(kwargs)
        required_keys = ["X_train", "y_train"]
        if self.use_network:
            required_keys += ["network", "beta_network"] #"same_train_and_test_data_bool"]
        else: # baseline situation :)
             required_keys += []
        if self.use_cross_validation_for_model_bool == False: # we then need to provide our own alpha lasso values
            #required_keys = ["alpha_lasso", "same_train_and_test_data_bool"]
            self.optimal_alpha = "User-specified optimal alpha lasso: " + str(self.alpha_lasso)
            required_keys += ["alpha_lasso"]
            if self.alpha_lasso == 0:
                self.model_type = "Linear"
            
        else: 
            self.model_type = "LassoCV"
        # check that all required keys are present:
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        if self.use_network:
            prior_network = self.network
            self.prior_network = prior_network
            self.A = prior_network.A
            self.network_params = prior_network.param_lists
            self.tf_names_list = prior_network.tf_names_list
        self.all_parameters_list = self.full_lists_gregulnet()

        import pandas as pd
        self.parameters_df = pd.DataFrame(self.all_parameters_list, 
                                          columns = ["parameter", "data type", "description", "value", "class"]).drop_duplicates()

        self._apply_parameter_constraints() # ensuring that the parameter constraints are met
        self.fit()
        
        
    def retrieve_tf_names_list_ml_model(self):
        import pandas as pd
#         edges_df = pd.DataFrame(self.edge_list)
#         all_nodes = list(set(list(edges_df[0]) + list(edges_df[1])))
#         all_nodes.sort()
#         min_edge = min(all_nodes)
        #max_edge = max(all_nodes)
        #N = len(all_nodes)
        tf_names_list = []
        for i in range(0, self.N):
            term = "TF" + str(i + 1)
            tf_names_list.append(term)
        return tf_names_list
           
    def fit(self): # fits a model Function used for model training 
        self.M = self.y_train.shape[0]
        #self.X_train = X
        self.N = self.X_train.shape[1]
        self.tf_names_list = self.retrieve_tf_names_list_ml_model()
        #self.y_train = y
        if self.use_network:
            print("network used")
            self.B_train = self.compute_B_matrix("train")
            self.X_tilda_train, self.y_tilda_train = self.compute_X_tilde_y_tilde(self.B_train, self.X_train, 
                                                                                  self.y_train)
            self.X_training_to_use, self.y_training_to_use = self.X_tilda_train, self.y_tilda_train
            self.data_used = "X_tilda_train, y_tilda_train"
        else:
            print("baseline used")
            self.X_training_to_use, self.y_training_to_use = self.X_train, self.y_train
            self.data_used = "X_train, y_train"
        
        self.regr = self.return_fit_ml_model(self.X_training_to_use, self.y_training_to_use)
        ml_model = self.regr
        if self.use_cross_validation_for_model_bool:
            self.optimal_alpha = "Cross-Validation optimal alpha lasso: " + str(ml_model.alpha_)
        coef = ml_model.coef_
        coef[coef == -0.0] = 0
        self.coef = coef # Get the coefficients
        if self.fit_y_intercept:
            self.intercept = ml_model.intercept_
        self.predY_tilda_train = ml_model.predict(self.X_training_to_use) # training data   
        self.mse_train = self.calculate_mean_square_error(self.y_training_to_use, self.predY_tilda_train) # Calculate MSE
        
        import pandas as pd
        
        #self.model_coefficients_df = pd.DataFrame(self.coef, index = self.tf_names_list).transpose()
        if self.fit_y_intercept:
            coeff_terms = [self.intercept] + list(self.coef)
            index_names = ["y_intercept"] + self.tf_names_list
            self.model_coefficients_df = pd.DataFrame(coeff_terms, index = index_names).transpose()
        else:
            coeff_terms = ["None"] + list(self.coef)
            index_names = ["y_intercept"] + self.tf_names_list
            self.model_coefficients_df = pd.DataFrame(coeff_terms, index = index_names).transpose()  
#             self.coefficients_df["y_intercept"] = self.intercept
        return self
    
    # # https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/
    def _apply_parameter_constraints(self):
        constraints = {**GRegulNet._parameter_constraints}
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
    
     # Function for model training
    # getter method
    def get_beta_network(self):
        return self.beta_network
    
     # getter method
    def get_alpha_lasso(self):
        return self.alpha_lasso
    
    def compute_B_matrix(self, data_type):##betaRidgeVal, X, A, M):
        """M is N_sample, because ||y - Xc||^2 need to be normalized by 1/n_sample, but not the 2 * beta_L2 * c'Ac term
        see https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
        The optimization objective for Lasso is:
        (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
        where M = n_sample
        """
        if data_type == "train":
            print("Training GRegulNet :)")
            X = self.X_train
        else:
            print("Testing GRegulnet :)")
            X = self.X_test
        
        XtX = X.T @ X
        beta_L2 = self.beta_network
        #N = A.shape[0]
        # N = self.A.shape[0]
        N_squared = self.N ** 2
        part_2 = 2 * beta_L2 * self.M / (N_squared) * self.A
        B = XtX + part_2 
        return B
    
    def compute_X_tilde_y_tilde(self, B, X, y):
        """Compute X_tilde, y_tilde such that X_tilde.T @ X_tilde = B,   y_tilde.T @ X_tilde = y.T @ X """
        U, s, _Vh = np.linalg.svd(B, hermitian=True)  # B = U @ np.diag(s) @ _Vh
        if (cond := s[0] / s[-1]) > 1e10:
            print(f'Large conditional number of B matrix: {cond: .2f}')
        S_sqrt = DiagonalLinearOperator(np.sqrt(s))
        S_inv_sqrt = DiagonalLinearOperator(1 / np.sqrt(s))
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
       
        
    def return_Linear_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import LinearRegression
        model_name = "Linear"
        #print(model_name)
        regr = LinearRegression(fit_intercept = self.fit_y_intercept)
        regr.fit(X, y)
        return regr    
        
    def return_Lasso_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import Lasso
        model_name = "Lasso"
        #print(model_name)
        regr = Lasso(alpha = self.alpha_lasso, fit_intercept = self.fit_y_intercept,
                    max_iter = self.max_lasso_iterations)
        regr.fit(X, y)
        return regr

    def return_LassoCV_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import LassoCV
        #model_name = "LassoCV"
        #print(model_name)
        regr = LassoCV(cv = self.num_cv_folds, random_state = 0, fit_intercept = self.fit_y_intercept)
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

    def calculate_mean_square_error(self, actual_values, predicted_values):
        # Please note that this function by Saniya calculates the Mean Square Error (MSE)
        import numpy as np
        difference = (actual_values - predicted_values)
        squared_diff = difference ** 2 # square of the difference
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff
    
    def predict(self, X_test, y_test):
        import pandas as pd
        import numpy as np
        self.X_test = X_test
        self.y_test = y_test
        ml_model = self.regr
        B_test = self.compute_B_matrix("test")
        X_tilda_test, y_tilda_test = self.compute_X_tilde_y_tilde(B_test, X_test, y_test)
        X_testing_to_use, y_testing_to_use = X_tilda_test, y_tilda_test
        predY_test = ml_model.predict(X_testing_to_use) # training data   
        mse_test = self.calculate_mean_square_error(y_testing_to_use, predY_test) # Calculate MSE
        return mse_test
    
    
    
    def full_lists_gregulnet(self):
        # network arguments used:
        # argument, description, our value
        term_to_add_last = "GRegulNet"
        if self.use_network:
            current_network_lists = self.network_params
            full_lists = current_network_lists
        else:
            full_lists = ["baseline", "no network", "original Lasso problem", "original", term_to_add_last]
            current_network_lists = full_lists
        row1 = ["model_type", "set of options", 
                "which model type should be used for geneRegulatNet",
                self.model_type, term_to_add_last]   
        full_lists.append(row1)
        
        if self.model_type == "Lasso":
            row1 = ["max_lasso_iterations", ">= 1", 
                "the maximum # of iterations for Lasso",
                self.model_type, term_to_add_last]   
            full_lists.append(row1)
        
        row1 = ["use_cross_validation_for_model_bool", "boolean", 
                "should we use cross validation for training the model",
                self.use_cross_validation_for_model_bool, term_to_add_last]   
        full_lists.append(row1)
        if self.use_cross_validation_for_model_bool == False:
            if self.alpha_lasso == 0:
                row1 = ["alpha_lasso", "0", "linear problem since alpha = 0", self.alpha_lasso, term_to_add_last]
            else:
                row1 = ["alpha_lasso", ">= 0", "value for alpha for the lasso problem", self.alpha_lasso, term_to_add_last]
            full_lists.append(row1)
        else:
            row1 = ["num_cv_folds", ">= 0", "the # of cross-validation folds to use", self.num_cv_folds, term_to_add_last]
            full_lists.append(row1)
        
        if self.use_network == False:
            row1 = ["use_network", "boolean", "baseline since no network regularization is done", self.use_network, term_to_add_last]
            full_lists.append(row1)
            
            row1 = ["beta_network", ">= 0", "baseline since no network regularization is done", self.beta_network, term_to_add_last]
            full_lists.append(row1)
            
        elif self.beta_network == 0:
            row1 = ["use_network", "boolean", "perform network regularization using a network prior", self.use_network, term_to_add_last]
            full_lists.append(row1)
            
            row1 = ["beta_network", ">= 0", "baseline since no network regularization is done", self.beta_network, term_to_add_last]
            full_lists.append(row1)
        else:
            
            row1 = ["use_network", "boolean", "perform network regularization using a network prior", self.use_network, term_to_add_last]
            full_lists.append(row1)
            
            row1 = ["beta_network", ">= 0", "value of beta for the network regularization problem", self.beta_network, term_to_add_last]
            full_lists.append(row1)           

            
        row1 = ["fit_y_intercept", "boolean", "fit a y-intercept for our regression problem", 
                self.fit_y_intercept, term_to_add_last]
        full_lists.append(row1)   
        
        return full_lists

    

# https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/
class baselineModel:
    """ :) Please note that this class focuses on building a baseline model
    from gene expression data for Transcription Factors (TFs) and gene expression data 
    for the target gene (TG) :) """
    
    _parameter_constraints = {
        "alpha_lasso": (0, None),
        "num_cv_folds": (0, None),
        "fit_y_intercept": [False, True],
        "same_train_and_test_data_bool": [False, True],
        "use_cross_validation_for_model_bool": [False, True],
        "max_lasso_iterations": (1, None),
        "model_type": ["Lasso", "LassoCV", "Linear"]#,
    }
    
    def __init__(self,  **kwargs):# beta_network, alpha_lasso, X_train, y_train, X_test, y_test):
        self.same_train_and_test_data_bool = False # different or same training and testing data?
        self.use_cross_validation_for_model_bool = False
        self.num_cv_folds = 5 # for cross validation models
        self.model_type = "Lasso"
        self.fit_y_intercept = False
        self.max_lasso_iterations = 10000
        self.__dict__.update(kwargs)
        required_keys = ["X_train", "y_train"]
        if self.use_cross_validation_for_model_bool == False: # we then need to provide our own alpha lasso values
            #required_keys = ["alpha_lasso", "same_train_and_test_data_bool"]
            self.optimal_alpha = "User-specified optimal alpha lasso: " + str(self.alpha_lasso)
            required_keys += ["alpha_lasso"]
            if self.alpha_lasso == 0:
                self.model_type = "Linear"
        else: 
            self.model_type = "LassoCV"
        # check that all required keys are present:
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        self.all_parameters_list = self.full_lists_baseline()

        import pandas as pd
        self.parameters_df = pd.DataFrame(self.all_parameters_list, 
                                          columns = ["parameter", "data type", "description", "value", "class"]).drop_duplicates()

        self._apply_parameter_constraints() # ensuring that the parameter constraints are met
        self.fit()
        
        
    def retrieve_tf_names_list_ml_model(self):
        import pandas as pd
        tf_names_list = []
        for i in range(0, self.N):
            term = "TF" + str(i + 1)
            tf_names_list.append(term)
        return tf_names_list
           
    def fit(self): # fits a model Function used for model training 
        self.M = self.y_train.shape[0]
        #self.X_train = X
        self.N = self.X_train.shape[1]
        self.tf_names_list = self.retrieve_tf_names_list_ml_model()
        #self.y_train = y
        print("baseline used")
        self.X_training_to_use, self.y_training_to_use = self.X_train, self.y_train
        self.data_used = "X_train, y_train"
        
        self.regr = self.return_fit_ml_model(self.X_training_to_use, self.y_training_to_use)
        ml_model = self.regr
        coef = ml_model.coef_
        if self.use_cross_validation_for_model_bool:
            self.optimal_alpha = "Cross-Validation optimal alpha lasso: " + str(ml_model.alpha_)

        coef[coef == -0.0] = 0
        self.coef = coef # Get the coefficients
        if self.fit_y_intercept:
            self.intercept = ml_model.intercept_
        self.predY_train = ml_model.predict(self.X_training_to_use) # training data   
        self.mse_train = self.calculate_mean_square_error(self.y_training_to_use, self.predY_train) # Calculate MSE
        
        import pandas as pd
        
        #self.model_coefficients_df = pd.DataFrame(self.coef, index = self.tf_names_list).transpose()
        if self.fit_y_intercept:
            coeff_terms = [self.intercept] + list(self.coef)
            index_names = ["y_intercept"] + self.tf_names_list
            self.model_coefficients_df = pd.DataFrame(coeff_terms, index = index_names).transpose()
        else:
            coeff_terms = ["None"] + list(self.coef)
            index_names = ["y_intercept"] + self.tf_names_list
            self.model_coefficients_df = pd.DataFrame(coeff_terms, index = index_names).transpose()  
#             self.coefficients_df["y_intercept"] = self.intercept
        return self
    
    # # https://www.geeksforgeeks.org/implementation-of-elastic-net-regression-from-scratch/
    def _apply_parameter_constraints(self):
        constraints = {**GRegulNet._parameter_constraints}
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
    
     # Function for model training
    # getter method
    
     # getter method
    def get_alpha_lasso(self):
        return self.alpha_lasso       
        
    def return_Linear_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import LinearRegression
        model_name = "Linear"
        #print(model_name)
        regr = LinearRegression(fit_intercept = self.fit_y_intercept)
        regr.fit(X, y)
        return regr    
        
    def return_Lasso_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import Lasso
        model_name = "Lasso"
        #print(model_name)
        regr = Lasso(alpha = self.alpha_lasso, fit_intercept = self.fit_y_intercept,
                    max_iter = self.max_lasso_iterations)
        regr.fit(X, y)
        return regr

    def return_LassoCV_ML_model(self, X, y):
        # model_type can be baseline or network
        from sklearn.linear_model import LassoCV
        #model_name = "LassoCV"
        #print(model_name)
        regr = LassoCV(cv = self.num_cv_folds, random_state = 0, fit_intercept = self.fit_y_intercept)
        regr.fit(X, y)
        #self.optimal_alpha = "Cross-Validation: " + str(regr.alpha_)
        return regr            

    def return_fit_ml_model(self, X, y):
        if self.model_type == "Linear":
            model_to_return = self.return_Linear_ML_model(X, y)
        elif self.model_type == "Lasso":
            model_to_return = self.return_Lasso_ML_model(X, y)
        elif self.model_type == "LassoCV":
            model_to_return = self.return_LassoCV_ML_model(X, y)
        return model_to_return

    def calculate_mean_square_error(self, actual_values, predicted_values):
        # Please note that this function by Saniya calculates the Mean Square Error (MSE)
        import numpy as np
        difference = (actual_values - predicted_values)
        squared_diff = difference ** 2 # square of the difference
        mean_squared_diff = np.mean(squared_diff)
        return mean_squared_diff
    
    def predict(self, X_test, y_test):
        import pandas as pd
        import numpy as np
        self.X_test = X_test
        self.y_test = y_test
        ml_model = self.regr
        X_testing_to_use, y_testing_to_use = X_test, y_test
        predY_test = ml_model.predict(X_testing_to_use) # training data   
        mse_test = self.calculate_mean_square_error(y_testing_to_use, predY_test) # Calculate MSE
        return mse_test
    
    
    def full_lists_baseline(self):
        # network arguments used:
        # argument, description, our value
        term_to_add_last = "baselineModel"
        full_lists = [["baseline", "no network", "original Lasso problem", "original", term_to_add_last]]
        current_network_lists = full_lists
        row1 = ["model_type", "set of options", 
                "which model type should be used for geneRegulatNet",
                self.model_type, term_to_add_last]   
        full_lists.append(row1)
        
        if self.model_type == "Lasso":
            row1 = ["max_lasso_iterations", ">= 1", 
                "the maximum # of iterations for Lasso",
                self.model_type, term_to_add_last]   
            full_lists.append(row1)
        
        row1 = ["use_cross_validation_for_model_bool", "boolean", 
                "should we use cross validation for training the model",
                self.use_cross_validation_for_model_bool, term_to_add_last]   
        full_lists.append(row1)
        if self.use_cross_validation_for_model_bool == False:
            if self.alpha_lasso == 0:
                row1 = ["alpha_lasso", "0", "linear problem since alpha = 0", self.alpha_lasso, term_to_add_last]
            else:
                row1 = ["alpha_lasso", ">= 0", "value for alpha for the lasso problem", self.alpha_lasso, term_to_add_last]
            full_lists.append(row1)
        else:
            row1 = ["num_cv_folds", ">= 0", "the # of cross-validation folds to use", self.num_cv_folds, term_to_add_last]
            full_lists.append(row1)
        
        row1 = ["use_network", "boolean", "baseline since no network regularization is done", False, term_to_add_last]
        full_lists.append(row1)   
            
        row1 = ["fit_y_intercept", "boolean", "fit a y-intercept for our regression problem", 
                self.fit_y_intercept, term_to_add_last]
        full_lists.append(row1)   
        
        return full_lists

def geneRegulatNet(X, y, edge_list, beta_net, cv_for_alpha = False, alpha_lasso = 0.1, 
                   edge_vals_for_d = False,
                  self_loops = False, d_pseudocount = 1e-3, 
                  default_edge_w = 0.1, sqrt_w_for_d = False, 
                  square_w_for_d = False, thresh_for_d = 0.5,
                 num_cv_folds = 5, 
                model_type = "Lasso", use_network = True, y_intercept = False,
                   max_lasso_iters = 10000):
    
    prior_graph_dict = {"edge_list": edge_list,
                       "edge_values_for_degree": edge_vals_for_d,
                       "consider_self_loops":self_loops,
                       "pseudocount_for_degree":d_pseudocount,
                        "default_edge_weight": default_edge_w,
                        "square_root_weights_for_degree":sqrt_w_for_d, 
                        "squaring_weights_for_degree": square_w_for_d, 
                        "threshold_for_degree": thresh_for_d}
                        
           ####################
    if use_network:
        print("prior graph network used")
        netty = PriorGraphNetwork(**prior_graph_dict) # uses the network to get features like the A matrix.
        greg_dict = {"X_train": X, 
                     "y_train": y,
                     "alpha_lasso": alpha_lasso,
                    "beta_network":beta_net,
                    "network": netty,
                    "use_cross_validation_for_model_bool": cv_for_alpha,
                     "num_cv_folds":num_cv_folds, 
                     "model_type":model_type, 
                     "use_network":use_network,
                     "fit_y_intercept":y_intercept, 
                     "max_lasso_iterations":max_lasso_iters
                    }
        greggy = GRegulNet(**greg_dict)
        return greggy
    else:
        print("baseline model (no prior network)")
        #baselineModel
        baseline_dict = {"X_train": X, 
                        "y_train": y,
                         "alpha_lasso": alpha_lasso,
                    "use_cross_validation_for_model_bool": cv_for_alpha,
                     "num_cv_folds":num_cv_folds, 
                     "model_type":model_type, 
                     "fit_y_intercept":y_intercept, 
                     "max_lasso_iterations":max_lasso_iters
                    }
        baseliney = baselineModel(**baseline_dict)
        return baseliney




    