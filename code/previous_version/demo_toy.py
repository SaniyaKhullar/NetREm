import sys
sys.path.append("../code")  # assuming "code" is one directory up and then down into "code"

from DemoDataBuilderXandY import generate_dummy_data
from Netrem_model_builder import netrem, netremCV
import PriorGraphNetwork as graph
import error_metrics as em 
import essential_functions as ef
import netrem_evaluation_functions as nm_eval
import Netrem_model_builder as nm

dummy_data = generate_dummy_data(corrVals = [0.9, 0.5, 0.4, -0.3, -0.8], # the # of elements in corrVals is the # of predictors (X)
                                 num_samples_M = 100000, # the number of samples M
                                 train_data_percent = 70) # the remainder out of 100,000 will be kept for testing. If 100, then ALL data is used for training and testing.

X_df = dummy_data.X_df
X_df.head()

y_df = dummy_data.y_df
y_df.head()

# 70,000 samples for training data (used to train and fit GRegulNet model)
X_train = dummy_data.view_X_train_df()
y_train = dummy_data.view_y_train_df()

# 30,000 samples for testing data
X_test = dummy_data.view_X_test_df()
y_test = dummy_data.view_y_test_df()

X_train.corr() # pairwise correlations among the training samples
X_test.corr() # pairwise correlations among the training samples


# prior network edge_list (missing edges or edges with no edge weight will be added with the default_edge_list so the network is fully-connected):
edge_list = [["TF1", "TF2", 0.8], ["TF4", "TF5", 0.95], ["TF1", "TF3"], ["TF1", "TF4"], ["TF1", "TF5"], 
             ["TF2", "TF3"], ["TF2", "TF4"], ["TF2", "TF5"], ["TF3", "TF4"], ["TF3", "TF5"]]

beta_network_val = 1 
# by default, model_type is Lasso, so alpha_lasso_val will be specified for the alpha_lasso parameter. 
# However, we will specify model_type = LassoCV, so our alpha_lasso is determined by cross-validation on training data).

# Building the network regularized regression model: 
# By default, edges are constructed between all of the nodes; nodes with a missing edge are assigned the default_edge_weight. 
netrem_demo = netrem(edge_list = edge_list, 
                     beta_net = beta_network_val,
                     model_type = "LassoCV",
                     view_network = True)

# Fitting the NetREm model on training data: X_train and y_train:
netrem_demo.fit(X_train, y_train)


pred_y_test = netrem_demo.predict(X_test) # predicted values for y_test
mse_test = netrem_demo.test_mse(X_test, y_test)
print(f"Please note that the testing Mean Square Error (MSE) is {mse_test}")

# To view and extract the predicted model coefficients for the predictors: 
netrem_demo.model_coef_df

netrem_demo.B_interaction_df


netrem_demo.final_corr_vs_coef_df
netrem_demo.combined_df
organize_B_interaction_network(netrem_demo)
