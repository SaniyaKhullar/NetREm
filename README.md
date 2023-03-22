<!-- ---
layout: default
---

{% include mathjax.html %} -->

# GRegulNet

## Gene Regular(ized/atory) Network

#### By: Saniya Khullar, Xiang Huang, Daifeng Wang
##### Daifeng Wang Lab

## Summary

GRegulNet is a software package that utilizes network-constrained regularization for biological applications and other network-based learning tasks. In biology, traditional regression methods can struggle with correlated predictors, particularly transcription factors (TFs) that regulate target genes in gene regulatory networks (GRNs). GRegulNet incorporates information from prior biological networks to improve predictions and identify complex relationships among predictors. This approach can highlight important nodes and edges in the network, provide insights into underlying biological processes, and improve model accuracy and biological/clinical significance of the models. GRegulNet can incorporate multiple types of network data, including PPI networks, gene co-expression networks, and metabolic networks.

<!-- In summary, network-constrained regularization may bolster the construction of more accurate and interpretable models that incorporate prior knowledge of the network structure among predictors. -->

## Pipeline

*Pipeline image of Gregulnet*

## Hardware Requirements

The analysis is based on Python version 3.10. Please note that larger prior graph networks may require more memory, space, and time. We anticipate that you would only need a standard computer (e.g. 32 GB RAM and 32 GB storage) with enough RAM to support the operations.

## Software Requirements

Please ensure you have cloned or downloaded our GRegulNet Github code and package. Please run the following command in the terminal or command prompt window to install the packages (and respective package versions and other dependencies) specified in our *requirements.txt* file: **pip install -r requirements.txt**
In short, we our code uses the following Python packages: *matplotlib.pyplot, networkx, numpy, numpy.typing, os, pandas, plotly.express, random, scipy, sklearn, sys, tqdm, warnings*. To install these packages manually, please run *pip install [package]* or *pip3 install [package]* in the terminal or run *conda install [package]* in the Anaconda navigator prompt.

## Description of GRegulNet pipeline function: geneRegulatNet

Please note that our package, GRegulNet, is run by the following function **geneRegulatNet** in Python. Fits a Network-constrained Lasso regression machine learning model given an undirected prior network and regularization parameters. 

## Usage

geneRegulatNet(<br>
                <!-- $X$, <br>
                $y$, <br> -->
                edge_list,<br> 
                beta_net, <br>
                cv_for_alpha = False,<br> 
                alpha_lasso = 0.1, <br>
                <!-- edge_vals_for_d = False, <br> -->
                <!-- self_loops = False, <br> -->
                default_edge_weight = 0.1, <br>
                degree_threshold = 0.5, <br>
                degree_pseudocount = 1e-3, <br> 
                num_cv_folds = 5, <br>
                <!-- model_type = "Lasso", <br> -->
                use_net = True,<br>
                y_intercept = False, <br>
                max_iters = 10000<br>
                )<br>

<!-- has 2 options with respect to the alpha_lasso_val ($\alpha_{lasso} \geq 0$) for the lasso regularization on the overall model: 
* default: the user may specify $\alpha_{lasso}$ manually (if *cv_for_alpha_lasso_model_bool = False*). If no alpha_lasso_val is specified, 0.1 will be used. 
* alternative: the user may opt for GRegulNet to select $\alpha_{lasso}$ based on cross-validation (CV) on training data (if *cv_for_alpha_lasso_model_bool = True*) -->

<!-- Ultimately, this function uses a prior network edge list and $\beta_{network}$ to build an estimator object from the class GRegulNet. This estimator can then take in input $X$ and $y$ data:  transforms them to $\tilde{X}$ and $\tilde{y}$, respectively, and use them to fit a Lasso regression model with a regularization value of $\alpha_{lasso}$. Overall, the trained model is more reflective of an underlying network structure among predictors and may be more biologically meaningful and interpretable.  -->




<!-- $$
\begin{cases}
  \text{geneRegulatNet(edge_list, } \beta_{network}, \text{cv_for_alpha_lasso_model_bool = } False, \alpha_{lasso}\text{)} & \text{if cv_for_alpha_lasso_model_bool = } False \\
  \text{geneRegulatNet(edge_list, } \beta_{network}, \text{cv_for_alpha_lasso_model_bool = } True) & \text{if cv_for_alpha_lasso_model_bool = } True \\
\end{cases}
$$

There are several additional parameters that can be adjusted in the geneRegulatNet function, which will be explained later in the *Default Parameters* section.  -->

<!-- ### Main Input: -->

<!-- * *edge_list*: A list of lists corresponding to a prior network involving the predictors (as nodes) and relationships among them as edges. We will utilize this prior network to constrain our machine learning model. For instance, this could be a Protein-Protein Interaction (PPI) network of interactions among the predictors. If  weights are missing for any edge, then the default_weight will be used for that respective edge. We assume that this network is undirected and thereby symmetric, so the user only needs to specify edges in 1 direction (and the other direction will be assumed automatically). 

For instance:

[[source<sub>1</sub>, target<sub>1</sub>, weight<sub>1</sub>], [source<sub>2</sub>, target<sub>2</sub>, weight<sub>2</sub>], ..., [source<sub>Z</sub>, target<sub>Z</sub>, weight<sub>Z</sub>]]. 

Where weight<sub>1</sub>, weight<sub>2</sub>, ..., weight<sub>Z</sub> are optional. If an edge is missing its respective edge weight, then the default edge weights will be utilized. 

The edge_list will be represented by:

| Source | Target |  Weight |
| --------- | ---------- | ---------- |
|source<sub>1</sub>   | target<sub>1</sub> | weight<sub>1</sub>|
|source<sub>2</sub>   | target<sub>2</sub> | weight<sub>2</sub> |
|...    | ... | ... |
|source<sub>Z</sub>    | target<sub>Z</sub> | weight<sub>Z</sub>|
|target<sub>1</sub>   | source<sub>1</sub> | weight<sub>1</sub> |
|target<sub>2</sub>    | source<sub>2</sub> | weight<sub>2</sub> |
|...    | ... | ... |
|target<sub>Z</sub>    | source<sub>Z</sub> | weight<sub>Z</sub> | -->

<!-- * *beta_network_val*:  A numerical value for $\beta_{network} \geq 0$.  -->

<!-- * *cv_for_alpha_lasso_model_bool*:
  - False (default): user wants to specify the value of $\alpha_{lasso}$
  - True: GRegulNet will perform cross-validation (CV) on training data to determine optimal $\alpha_{lasso}$ -->

<!-- $$ = \begin{cases}
  \text{if cv_for_alpha_lasso_model_bool = } False & \text{default: user wants to specify the value of }  \alpha_{lasso}  \\
  \text{if cv_for_alpha_lasso_model_bool = } True & \text{GRegulNet will perform cross-validation (CV) on training data to determine optimal } \alpha_{lasso} \\
\end{cases}
$$ -->

<!-- $$ = \begin{cases}
  \text{if cv_for_alpha_lasso_model_bool = } False & \text{default: user wants to specify the value of }  \alpha_{lasso}  \\
  \text{if cv_for_alpha_lasso_model_bool = } True & \text{GRegulNet will perform cross-validation (CV) on training data to determine optimal } \alpha_{lasso} \\
\end{cases}
$$ -->


<!-- ##### If *cv_for_alpha_lasso_model_bool* is False, we need to specify alpha_lasso_val $\alpha_{lasso}$ ##### -->

<!-- * *alpha_lasso_val*:  A numerical value for $\alpha_{lasso} \geq 0$. If *cv_for_alpha_lasso_model_bool* is False, the user is then advised to specify this $\alpha_{lasso}$ parameter (alpha_lasso_val). Otherwise, if no $\alpha_{lasso}$ value is specified, then the default value of $\alpha_{lasso} = 0.1$ will be used.  -->

### Main Inputs:

<!-- | Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| edge_list       | list of lists: [[source<sub>1</sub>, target<sub>1</sub>, weight<sub>1</sub>], ..., [source<sub>Z</sub>, target<sub>Z</sub>, weight<sub>Z</sub>]] | value needed |
| $\beta_{network}$  | Regularization parameter for network penalization | value needed |
| cv_for_alpha_lasso_model_bool  | Should GRegulNet perform Cross Validation to determine $\alpha_{lasso}$  | False |
| $\alpha_{lasso}$  | Regularization parameter for lasso | value needed if cv_for_alpha_lasso_model_bool = False; default: 0.1 | -->


| Parameter | Definition | 
| --------- | ---------- | 
| edge_list       | A list of lists corresponding to a prior network involving predictors (nodes) and relationships among them (edges): [[source<sub>1</sub>, target<sub>1</sub>, weight<sub>1</sub>], ..., [source<sub>Z</sub>, target<sub>Z</sub>, weight<sub>Z</sub>]]. Here, weight<sub>1</sub>, ..., weight<sub>Z</sub> are optional. | 
| beta_net: $\beta_{net}$  | Regularization parameter for network penalization: $\beta_{net} \geq 0$. | 
| cv_for_alpha | * False (default): user specifies value of $\alpha_{lasso}$ <br> * True: GRegulNet performs cross-validation (CV) on training data to determine optimal $\alpha_{lasso}$  | 
| alpha_lasso: $\alpha_{lasso}$  | A numerical regularization parameter for lasso needed if cv_for_alpha = False: $\alpha_{lasso} \geq 0$. |



<!-- | Parameter | Definition | 
| --------- | ---------- | 
| $X$ | Input numpy array matrix (list of lists) each list corresponds to a sample. Here, rows are samples and columns are predictors. | 
| $y$ | Input numpy array list with 1 value for each sample.|  -->


<!-- | Parameter | Definition | More information |
| --------- | ---------- | ---------- |
| edge_list       | A list of lists corresponding to a prior network involving predictors (nodes) and relationships among them (edges): [[source<sub>1</sub>, target<sub>1</sub>, weight<sub>1</sub>], ..., [source<sub>Z</sub>, target<sub>Z</sub>, weight<sub>Z</sub>]]. Here, weight<sub>1</sub>, ..., weight<sub>Z</sub> are optional. | This prior network constrains our model. We assume that this network is undirected and thereby symmetric, so the user only needs to specify edges in 1 direction (and other directions are assumed automatically). The default edge weight is utilized for any edge with a missing edge weight.|
| beta_network_val: $\beta_{network}$  | Regularization parameter for network penalization: $\beta_{network} \geq 0$. | Value needed, which scales the strength of network penalization |
| cv_for_alpha_lasso_model_bool  | Should GRegulNet perform Cross Validation to determine $\alpha_{lasso}$? | Default boolean value: False. <br>* False (default): user wants to specify the value of $\alpha_{lasso}$ <br> * True: GRegulNet will perform cross-validation (CV) on training data to determine optimal $\alpha_{lasso}$  |
| alpha_lasso_val: $\alpha_{lasso}$  | A numerical regularization parameter for lasso: $\alpha_{lasso} \geq 0$. | Value needed if cv_for_alpha_lasso_model_bool = False; default: 0.1 |
  -->

### Default Parameters: ###

* Parameters for the graph prior network:

| Parameter | Definition | 
| --------- | ---------- | 
| default_edge_weight  | Default edge weight ($w$) assigned to any edge with missing weight | 
| degree_pseudocount  | Pseudocount to add for the degree of each node in the network. |
| degree_threshold  | Edges with weight $w$ > degree_threshold are counted as 1 towards node degree (if *edge_vals_for_d is False*) |
<!-- | sqrt_w_for_d  | Sum $\sqrt{w}$ for a given node degree (if *edge_vals_for_d is True*) |
| square_w_for_d  | Sum $w^{2}$ for a given node degree (if *edge_vals_for_d is True*) | -->
 <!-- self_loops  | True: Add 1 to each degree ($d$) for each node in the network (for self-loops)| 
 | edge_vals_for_d  | True: edge weights $w$ used for node degree; False: threshold used | default: False| -->
<!-- | w_transform_for_d | To calculate degree for a given node, we can choose from 1 of 3 options (if *edge_vals_for_d is True*): <br> * "none": sum original $w$ <br> * "sqrt": sum $\sqrt{w}$ <br> * "square": sum $w^{2}$ |  -->

* Parameters for the network-based regularized model:

| Parameter | Definition | 
| --------- | ---------- | 
| use_net  | * True: use a prior graph network for regularization. <br> * False: fit a Lasso model on original $X$ and $y$ data (baseline). | 
| y_intercept | * True: y-intercept is fitted for the final GRegulNet model. <br> * False: no y-intercept is fitted (model coefficients are only for predictors)| 
| maxit  | the maximum # of iterations we will run Lasso regression model for (if *cv_for_alpha is False*) |
| num_cv_folds  | # of cross-validation (cv) folds we fit on training data during model building (if *cv_for_alpha is True*) |

### Details:

 We input an edge list of the prior graph network (constrains the model via network-based regularization) and a beta_network_val ($\beta_{net} \geq 0$, which scales the network-based regularization penalty). The user may specify the alpha_lasso_val ($\alpha_{lasso} \geq 0$) manually for the lasso regularization on the overall model (if *cv_for_alpha = False*) or GRegulNet may select an optimal $\alpha_{lasso}$ based on cross-validation (CV) on the training data (if *cv_for_alpha = True*). Then, **geneRegulatNet** builds an estimator object from the class GRegulNet that can then take in input $X$ and $y$ data: transforms them to $\tilde{X}$ and $\tilde{y}$, respectively, and use them to fit a Lasso regression model with a regularization value of $\alpha_{lasso}$. Ultimately, the trained GRegulNet model is more reflective of an underlying network structure among predictors and may be more biologically meaningful and interpretable. 

### Output Values: ###

* A fitted GRegulNet network-regularized linear model estimator object from the GRegulNet class. There are several methods we can call for our GRegulNet estimator object:

#### Methods:

* **fit($X$, $y$)**

Building and training the GRegulNet model with $X$ and $y$ data. 

| Parameter | Definition | 
| --------- | ---------- | 
| $X$ | Input numpy array matrix (list of lists) where each list corresponds to a sample used for training. Here, rows are samples and columns are predictors. | 
| $y$ | Input numpy array list for model training with 1 value for each sample.| 


We can retrieve our model coefficients and other attributes by calling these outputs:

| Output | Definition | 
| --------- | ---------- | 
| model_coef_df  | Pandas dataframe of the Lasso model coefficients for the predictors and y-intercept (if *y_intercept = True*) | 
| optimal_alpha  | If *cv_for_alpha = True*, returns the optimal $\alpha_{lasso}$ found by CV on training data | 
| all_params_list  | List of lists of the parameters used for GRegulNet model (defensive programming) | 
| params_df | Pandas dataframe of the parameters used for GRegulNet model (defensive programming) | 
| mse_train | Mean Square Error (MSE): predicted versus actual values | 

* **predict_y($X$)**

We can use our model to predict values for our response variable $y$. 

| Parameter | Definition | 
| --------- | ---------- | 
| $X$ | Input numpy array matrix (list of lists) where each list corresponds to a sample. Here, rows are samples and columns are predictors. | 

  - Returns:
  Numpy array of $\hat{y}$ predicted values for $y$.

* **test_mse($X$, $y$)**

We can evaluate our model performance capabilities on data like testing data using the Mean Squared Error (MSE) by calling the predict($X$, $y$) function. 

| Parameter | Definition | 
| --------- | ---------- | 
| $X$ | Numpy array matrix (list of lists) where each list corresponds to a sample. Here, rows are samples and columns are predictors. | 
| $y$ | Numpy array list for response variable with 1 value for each sample.| 

  - Returns:
  Numeric value corresponding to the Mean Square Error. 
  
  MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y_i})^2

## Demo (Toy Example) of GRegulNet:

Suppose we want to build a machine learning model to predict the gene expression level of our target gene (TG) $y$ based on the expression levels of 5 Transcription Factors (TFs): [TF<sub>1</sub>, $TF_{2}$, $TF_{3}$, $TF_{4}$, $TF_{5}$], which are our respective predictors [X<sub>1</sub>, $X_{2}$, $X_{3}$, $X_{4}$, $X_{5}$]. We generate 100 random samples (rows) of data where the Pearson correlations ($r$) of predictors with $y$ are *corrVals*: [cor(TF<sub>1</sub>, $y$) = 0.9, cor(TF<sub>2</sub>, $y$) = 0.5, cor(TF<sub>3</sub>, $y$) = 0.1, cor(TF<sub>4</sub>, $y$) = -0.2, cor(TF<sub>5</sub>, $y$) = -0.8]. The dimensions of $X$ are therefore 100 rows by 5 columns (predictors). More details about our *generate_dummy_data* function (and additional parameters we can adjust for) are in *Dummy_Data_Demo_Example.ipynb*. Our GRegulNet estimator also incorporates an **undirected prior graph network** of biological relationships among our 5 TFs based on a Protein-Protein Interaction (PPI) network, where higher edge weights $w$ indicate stronger interactions at the protein-level. 

```python
from gregulnet_classes import * # to load our package, GRegulNet

dummy_data = generate_dummy_data(corrVals = [0.9, 0.5, 0.1, -0.2, -0.8],
                   num_samples_M = 100,
                   train_data_percent = 70)

# 70 samples for training data (used to train and fit GRegulNet model)
X_train = dummy_data.get_X_train()
y_train = dummy_data.get_y_train()

# 30 samples for testing data
X_test = dummy_data.get_X_test()
y_test = dummy_data.get_y_test()

# prior network edge_list:
edge_list = [[1, 2, 0.9], [4, 5, 0.75], [1, 3], [1, 4], [1, 5], 
              [2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]
beta_network_val = 10 
# by default, cv_for_alpha is False, so alpha_lasso_val will be specified for the alpha_lasso parameter.
alpha_lasso_val = 0.01

# Building the network regularized regression model. 
gregulnet_demo = geneRegulatNet(edge_list = edge_list, 
                                beta_net = beta_network_val,
                                alpha_lasso = alpha_lasso_val)

# Fitting the gregulnet model on training data: X_train and y_train:
gregulnet_demo.fit(X_train, y_train)
```
  <!-- prior graph network used
  :) Please note that we count the number of edges with weight > 0.5 to get the degree for a given node.
  :) We also add 0.001 as a pseudocount to our degree value for each node.
  network used
  Training GRegulNet :) -->
    
  
<!-- ![png](README_python_files/README_python_12_1.png) -->
![png](output_12_1.png)



<!-- There is a particularly strong relationship between $TF_{1} \leftrightarrow TF_{2}$ of 0.9 and between $TF_{4} \leftrightarrow TF_{5}$ of 0.75. The remaining relationships among the other TFs is assumed to be the default (edge weight of 0.1). -->
<!-- Here, $gregulnet_{demo}$ is an object of the *GRegulNet* class. We fit a model using $X_{train}$ and $y_{train}$ data (70 samples). -->

To view and extract the predicted model coefficients for the predictors: 

<!-- ```python
gregulnet_demo.coef
```

    array([ 0.23655573,  0.11430656,  0.00148755, -0.03512912, -0.16009479]) -->


```python
gregulnet_demo.model_coef_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_intercept</th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0.236556</td>
      <td>0.114307</td>
      <td>0.001488</td>
      <td>-0.035129</td>
      <td>-0.160095</td>
    </tr>
  </tbody>
</table>
</div>



We can test the performance of our data on testing data (30 samples), to understand better the generalizability of our GRegulNet model on new, unseen, data. 


```python
pred_y_test = gregulnet_demo.predict_y(X_test) # predicted values for y_test
mse_test = gregulnet_demo.test_mse(X_test, y_test)

print(f"Please note that the testing Mean Square Error (MSE) is {mse_test}")
```

    Testing GRegulnet :)
    Please note that the testing Mean Square Error (MSE) is 0.17220844811456956
    

<!-- ### Comparison Demo: GRegulNet versus Baseline Model for Cross-Validation Alpha Lasso

We will use the same $X_{train}$, $y_{train}$, $X_{test}$, and $y_{test}$ data and same prior network here to compare illustrate the effectiveness of GRegulNet in terms of a lower testing MSE (relative to a baseline model that incorporates no prior network). For ease of comparison, we will select the optimal alpha_lasso for each model using cross validation (CV) on the training data (that is, *cv_for_alpha_lasso_model_bool* = True). This example also shows how to run **geneRegulatNet** when alpha_lasso is determined by CV. 

#### GRegulNet using Cross validation for Alpha Lasso 
```python
# geneRegulatNet where alpha_lasso is determined by cross-validation on training data: :)
gregulnet_cv_demo = geneRegulatNet(edge_list = edge_list, beta_network_val = 10,
                              alpha_lasso_val = alpha_lasso_val, 
                              cv_for_alpha_lasso_model_bool = True)
gregulnet_cv_demo.fit(X_train, y_train)
print(gregulnet_cv_demo.optimal_alpha)
gregulnet_cv_mse_test = gregulnet_cv_demo.predict(X_test, y_test)
print(f"Please note that the testing Mean Square Error (MSE) for GRegulNet-CV model is {gregulnet_cv_mse_test}")
gregulnet_cv_demo.model_coefficients_df
```

    prior graph network used
    :) Please note that we count the number of edges with weight > 0.5 to get the degree for a given node.
    :) We also add 0.001 as a pseudocount to our degree value for each node.
    
    network used
    Training GRegulNet :)
    Cross-Validation optimal alpha lasso: 0.0041235620913686235
    Testing GRegulnet :)
    Please note that the testing Mean Square Error (MSE) for GRegulNet-CV model is 0.020310913421979375
    

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_intercept</th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0.236283</td>
      <td>0.116051</td>
      <td>0.001487</td>
      <td>-0.037593</td>
      <td>-0.161468</td>
    </tr>
  </tbody>
</table>
</div>

    
<!-- ![png](output_27_2.png) -->
    
<!-- 
#### Baseline Model using Cross validation for Alpha Lasso 
```python
# baseline lasso model (no prior network). Optimal alpha_lasso determined by cross-validation
# on the training data: :)
baseline_demo = geneRegulatNet(edge_list = edge_list, beta_network_val = None,
                              alpha_lasso_val = alpha_lasso_val, 
                              cv_for_alpha_lasso_model_bool = True,
                              use_network = False)

baseline_demo.fit(X_train, y_train)
print(baseline_demo.optimal_alpha)
baseline_mse_test = baseline_demo.predict(X_test, y_test)
print(f"Please note that the testing Mean Square Error (MSE) for the baseline model is {baseline_mse_test}")
baseline_demo.model_coefficients_df
```

    baseline model (no prior network)
    Cross-Validation optimal alpha lasso: 0.022006210642838385
    Please note that the testing Mean Square Error (MSE) for the baseline model is 0.1630541856987722
    
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_intercept</th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0.256041</td>
      <td>0.036381</td>
      <td>0.076338</td>
      <td>0</td>
      <td>-0.208916</td>
    </tr>
  </tbody>
</table>
</div> --> 

## References

[1]: Caiyan Li, Hongzhe Li, Network-constrained regularization and variable selection for analysis of genomic data, Bioinformatics, Volume 24, Issue 9, May 2008, Pages 1175–1182, https://doi.org/10.1093/bioinformatics/btn081

[2]: Wilkinson, M.D., Dumontier, M., Aalbersberg, I.J. et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18

[3]: Zhang, Y., Akutsu, T., & Ching, W. K. (2016). Incorporating network topology and phylogenetic constraints for protein-protein interaction prediction. BMC bioinformatics, 17(1), 1-15. https://doi.org/10.1186/s12859-016-1310-4

[4]: Jia, C., Zhang, Y., Chen, K., & Zhang, S. (2018). A novel feature extraction method with improved consistency for identifying cell cycle regulated genes. Bioinformatics, 34(5), 896-903. https://doi.org/10.1093/bioinformatics/btx657

[5]: Lu, Y., Chen, X., & Hu, Z. (2017). Recognition of protein/gene names from text using an ensemble of classifiers and effective abbreviation resolution. BMC bioinformatics, 18(1), 1-11. https://doi.org/10.1186/s12859-017-1515-1

[6]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830.
