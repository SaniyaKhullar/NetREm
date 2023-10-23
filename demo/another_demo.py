import sys
sys.path.append("../code")  # assuming "code" is one directory up and then down into "code"

from DemoDataBuilderXandY import generate_dummy_data
from Netrem_model_builder import netrem, netremCV
import PriorGraphNetwork as graph
import error_metrics as em 
import essential_functions as ef
import netrem_evaluation_functions as nm_eval
import Netrem_model_builder as nm

dummy_data = generate_dummy_data(corrVals = [0.9, 0.5, 0.1, -0.2, -0.8, -0.3],
                    standardize_X = False,
                    center_y = False, 
                   num_samples_M = 100,
                   train_data_percent = 70)

X_df = dummy_data.X_df
X_df.head()

y_df = dummy_data.y_df
y_df.head()

# 70 samples for training data (used to train and fit GRegulNet model)
X_train = dummy_data.view_X_train_df()
y_train = dummy_data.view_y_train_df()

# 30 samples for testing data
X_test = dummy_data.view_X_test_df()
y_test = dummy_data.view_y_test_df()

# prior network edge_list:
edge_list = [["TF1", "TF2", 0.9], ["TF4", "TF5", 0.75], ["TF1", "TF3"], ["TF1", "TF4"], ["TF1", "TF5"], 
              ["TF2", "TF3"], ["TF2", "TF4"], ["TF2", "TF5"], ["TF3", "TF4"], ["TF3", "TF5"]]

beta_network_val = 3 
# by default, cv_for_alpha is False, so alpha_lasso_val will be specified for the alpha_lasso parameter.
alpha_lasso_val = 0.01

# Building the network regularized regression model: 
# Please note: To include nodes found in the gene expression data that are not found in the PPI Network (e.g. TF6 in our case), we use False for the overlapped_nodes_only argument (otherwise, we would only use TFs 1 to 5):
netrem_demo = nm.netrem(edge_list = edge_list, 
                        beta_net = beta_network_val,
                        alpha_lasso = alpha_lasso_val,
                        overlapped_nodes_only = False, # so we include TF6
                        view_network = True)

# Fitting the gregulnet model on training data: X_train and y_train:
netrem_demo.fit(X_train, y_train)

pred_y_test = netrem_demo.predict(X_test) # predicted values for y_test
mse_test = netrem_demo.test_mse(X_test, y_test)
print(f"Please note that the testing Mean Square Error (MSE) is {mse_test}")

# To view and extract the predicted model coefficients for the predictors: 
netrem_demo.model_coef_df

netrem_demo.B_interaction_df
