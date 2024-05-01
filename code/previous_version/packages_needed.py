import pandas as pd
import numpy as np
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
from skopt import gp_minimize, space
from typing import Optional, List, Tuple
from sklearn.metrics import make_scorer
import plotly.express as px
from sklearn.base import RegressorMixin, ClassifierMixin, BaseEstimator
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