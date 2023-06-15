from scipy.sparse.linalg.interface import LinearOperator
import pandas as pd
import numpy as np
from packages_needed import *

# Python program to illustrate the intersection
# of two lists in most simple way
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def view_matrix_as_dataframe(matrix, column_names_list = [], row_names_list = []):
    # :) Please note this function by Saniya returns a dataframe representation of the numpy matrix
    # optional are the names of the columns and names of the rows (indices)
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
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


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
    
    
def normalize_data_zero_to_one(data):
    # https://stackoverflow.com/questions/18380419/normalization-to-bring-in-the-range-of-0-1
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def draw_arrow(direction = "down", color = "blue"):
    x = [0.5, 0.5]
    if direction == "down":
        # Define the coordinates for the arrow
        y = [0.9, 0.1]
    else: # up-arrow
        y = [0.1, 0.9]
    fig, ax = plt.subplots(figsize=(2,2))
    # Plot the arrow using Matplotlib
    plt.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], head_width=0.05, head_length=0.1, fc=color, ec=color)
    # Set the x and y limits to adjust the plot size
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off') # Hide the axis labels
    plt.show() # Show the plot