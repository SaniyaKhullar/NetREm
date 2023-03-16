---
layout: default
---

{% include mathjax.html %}

# GRegulNet

## Gene Regular(ized/atory) Network

#### By: Saniya Khullar, Xiang Huang, Daifeng Wang
##### Daifeng Wang Lab

## Summary

Regression techniques often face the challenge of dealing with correlated predictors, which can lead to unreliable model estimates. In biology, the variables are usually highly interconnected, and the relationships between them are complex and non-linear, making it difficult to obtain accurate regression models without incorporating prior knowledge. Traditional regression methods may not consider the potential impact of correlated predictors, including those representing Transcription Factors that regulate target genes in gene regulatory networks (GRNs), for instance. Network-constrained regularization addresses this issue by imposing constraints on the regression coefficients based on a network structure, ensuring that the model is consistent with the underlying biology and reducing the risk of overfitting to noise. In recent years, several software packages, including GRegulNet, have been developed that implement network-constrained regularization approaches for biological applications. These methods enable the identification of complex and non-linear relationships between variables, which traditional regression methods may miss. GRegulNet incorporates information from a prior network to improve predictions and is particularly useful for analyzing biological data. By leveraging the underlying network structure, network-constrained regularization methods can help identify relationships, highlight important nodes and edges in the network, provide insight into the underlying biological processes, and improve model accuracy and biological/clinical significance of the models. GRegulNet can be applied to other biological and non-biological applications, and for various types of network-based learning tasks where predictors are correlated or interconnected. In summary, network-constrained regularization is essential in biological applications, as it enables the construction of more accurate and interpretable models that incorporate prior knowledge of the biological network structure. GRegulNet is a critical tool that allows for the incorporation of multiple types of network data, including protein-protein interaction networks, GRNs, and metabolic networks, leading to better understanding of complex biological systems. Overall, incorporating network-constrained regularization using tools such as GRegulNet is a critical step towards improving the accuracy and interpretability of regression models in biological and other applications.

## Pipeline

*Pipeline image of Gregulnet*

Alternate names:
* LassoRegNet
* LassoNet

## Hardware Requirements

The analysis is based on Python 3.10.6. Please note that larger prior graph networks may require more memory, space, and time. We anticipate that you would only need a standard computer with enough RAM to support the operations. A Linux system with 32 GB RAM and 32GB storage would be enough to support GRegulNet. 

## Software Requirements


Please open a terminal window or command prompt window and navigate to your project directory using the command line or terminal.

Please ensure you have cloned or downloaded our GRegulNet Github code and package.  We have provided a *requirements.txt* file, which lists the key packages and respective versions needed. To install these packages properly, please run the following command in the terminal or command prompt window to install the packages specified in the *requirements.txt* file:

**pip install -r requirements.txt**

This command will install all the packages listed in the *requirements.txt* file, including any specified versions or dependencies.

In short, please note that we need to import the following Python packages needed to run our code:

* matplotlib.pyplot, networkx, numpy, numpy.typing, os, pandas, plotly.express, random, scipy, sklearn, sys, tqdm, warnings

To install these packages manually, please run *pip install [package]* or *pip3 install [package]* in the terminal or run *conda install [package]* in the Anaconda prompt.

## Functions in the GRegulNet pipeline


<!-- 
# * parameter 1: edge_list
# * beta_network_val
# * alpha_lasso_val, beta_network_val.  -->

Please note that our package, GRegulNet, is run by the following function **geneRegulatNet** in Python.

The function, **geneRegulatNet**, inputs the edge list of the prior graph network (to be used to constrain the model via network-based regularization) and a beta_network_val ($\beta_{network} \geq 0$) that scales the network-based regularization penalty. The user has 2 options with respect to the lasso regularization on the overall model: 
* default option: the user may  specify an alpha_lasso_val ($\alpha_{lasso} \geq 0$) manually (if *cv_for_alpha_lasso_model_bool = False*). If no alpha_lasso_val is specified, 0.1 will be used as the default. 
* alternative option: the user may opt for GRegulNet to select the alpha_lasso_val ($\alpha_{lasso} \geq 0$) based on cross-validation on the training data (if *cv_for_alpha_lasso_model_bool = True*)

Ultimately, this function builds an estimator object from the class GRegulNet. This estimator is then able to take in input $X$ and $y$ data and transform those data inputs (to $\tilde{X}$ and $\tilde{y}$, respectively) based on the prior network edge lists and $\beta_{network}$ value. Next, the estimator can fit a Lasso regression model on $\tilde{X}$ and $\tilde{y}$ using the regularization value of alpha_lasso_val ($\alpha_{lasso} \geq 0$), which is either provided by the user (if *cv_for_alpha_lasso_model_bool = False*) or determined by GRegulNet (based on cross-validation on training data, if *cv_for_alpha_lasso_model_bool = True*). Overall, the trained and fitted models are more reflective of an underlying network structure among predictors and may be more biologically meaningful and interpretable. 


**geneRegulatNet**:

The basic version of geneRegulatNet requires that the user is aware of at least the following 4 parameters, which we list and explain in the *Main Input* section. That is, *geneRegulatNet(edge_list, beta_network_val, cv_for_alpha_lasso_model_bool, alpha_lasso_val)*. At minimum, the user needs to specify just 3 parameters. 

$$
\begin{cases}
  \text{geneRegulatNet(edge_list, } \beta_{network}, \text{cv_for_alpha_lasso_model_bool = } False, \alpha_{lasso}\text{)} & \text{if cv_for_alpha_lasso_model_bool = } False \\
  \text{geneRegulatNet(edge_list, } \beta_{network}, \text{cv_for_alpha_lasso_model_bool = } True) & \text{if cv_for_alpha_lasso_model_bool = } True \\
\end{cases}
$$

There are several additional parameters that can be adjusted in the geneRegulatNet function, which will be explained later in the *Default Parameters* section. 

### Main Input:

* *edge_list*: A list of lists corresponding to a prior network involving the predictors (as nodes) and relationships among them as edges. We will utilize this prior network to constrain our machine learning model. For instance, this could be a Protein-Protein Interaction (PPI) network of interactions among the predictors. If the weights are missing for any edge, then the default_weight will be used for that respective edge. We assume that this network is undirected and thereby symmetric, so the user only needs to specify edges in 1 direction (and the other direction will be assumed automatically). 

For instance:
<img src="https://render.githubusercontent.com/render/math?math=[[[source$_{1}$, target$_{1}$, weight$_{1}$], [source$_{2}$, target$_{2}$, weight$_{2}$], ..., [source$_{Z}$, target$_{Z}$, weight$_{Z}$]]]" alt="[[[source$_{1}$, target$_{1}$, weight$_{1}$], [source$_{2}$, target$_{2}$, weight$_{2}$], ..., [source$_{Z}$, target$_{Z}$, weight$_{Z}$]]]">

[[source$_{1}$, target$_{1}$, weight$_{1}$], [source$_{2}$, target$_{2}$, weight$_{2}$], ..., [source$_{Z}$, target$_{Z}$, weight$_{Z}$]]. 

Where weight$_{1}$, weight$_{2}$, ..., weight$_{3}$ are optional. If an edge is missing its respective edge weight, then the default edge weights will be utilized. 

The edge_list will be represented by:

| Source | Target |  Weight |
| --------- | ---------- | ---------- |
|source$_{1}$    | target$_{1}$ | weight$_{1}$ |
|source$_{2}$    | target$_{2}$ | weight$_{2}$ |
|...    | ... | ... |
|source$_{Z}$    | target$_{Z}$ | weight$_{Z}$ |
|target$_{1}$    | source$_{1}$ | weight$_{1}$ |
|target$_{2}$    | source$_{2}$ | weight$_{2}$ |
|...    | ... | ... |
|target$_{Z}$    | source$_{Z}$ | weight$_{Z}$ |

* *beta_network_val*:  A numerical value for $\beta_{network} \geq 0$. 

<!-- * *cv_for_alpha_lasso_model_bool*: A boolean value (True or False). This parameter asks the user if the user wants GRegulNet to perform cross-validation (CV) on the training data to determine the optimal alpha_lasso_val ($\alpha_{lasso} \geq 0$). The default value is False, which means that the user wants to specify the value of $\alpha_{lasso}$.  -->

* *cv_for_alpha_lasso_model_bool*:

$$ = \begin{cases}
  \text{if cv_for_alpha_lasso_model_bool = } False & \text{default: user wants to specify the value of }  \alpha_{lasso}  \\
  \text{if cv_for_alpha_lasso_model_bool = } True & \text{GRegulNet will perform cross-validation (CV) on training data to determine optimal } \alpha_{lasso} \\
\end{cases}
$$



<!-- $$ = \begin{cases}
  \text{default: user wants to specify the value of }  \alpha_{lasso} & \text{if cv_for_alpha_lasso_model_bool = } False \\
  \text{GRegulNet will perform cross-validation (CV) on training data to determine optimal } \alpha_{lasso} & \text{if cv_for_alpha_lasso_model_bool = } True \\
\end{cases}
$$
 -->
##### If *cv_for_alpha_lasso_model_bool* is False, we need to specify alpha_lasso_val $\alpha_{lasso}$ #####

* *alpha_lasso_val*:  A numerical value for $\alpha_{lasso} \geq 0$. Thus, if *cv_for_alpha_lasso_model_bool* is False, the user is then advised to specify this $\alpha_{lasso}$ parameter (alpha_lasso_val). Otherwise, please note that if no $\alpha_{lasso}$ value is specified, then the default value of $\alpha_{lasso} = 0.1$ will be used. 


### Summary of main inputs needed:
| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| edge_list       | list of lists: [[source$_{1}$, target$_{1}$, weight$_{1}$], ..., [source$_{Z}$, target$_{Z}$, weight$_{Z}$]] | value needed |
| $\beta_{network}$  | Regularization parameter for network penalization | value needed |
| cv_for_alpha_lasso_model_bool  | Should GRegulNet perform Cross Validation to determine $\alpha_{lasso}$  | False |
| $\alpha_{lasso}$  | Regularization parameter for lasso | value needed if cv_for_alpha_lasso_model_bool = False; default: 0.1 |

 
### Default parameters: ###

Please note these parameters that can be adjusted as needed for user needs and specifications. 

* Parameters for the graph prior network:

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| default_edge_weight  | If an edge is missing an edge weight, this is weight assigned to that edge | 0.1 |
| consider_self_loops  | For Degree Matrix D. True: Add 1 to each degree (for self-loops)| False|
| pseudocount_for_diagonal_matrix  | For Degree Matrix D. Pseudocount to add for each degree (node). | 0.001 |
| use_edge_weight_values_for_degrees_bool  | For Degree Matrix D. True: edge weights used; False: threshold used | False|


If *use_edge_weight_values_for_degrees_bool is False*, we will use a threshold to assign degrees:

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| threshold_for_degree  | For Degree Matrix D. Edges with weight > threshold_for_degree are counted as 1 towards degree | 0.5 |


If *use_edge_weight_values_for_degrees_bool is True*, we can use the edge weights $w$ directly, $\sqrt{w}$, or $w^{2}$ for the degree:

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| square_root_weights_for_degree_sum_bool  | Sum square root of edge weights $\sqrt{w}$ for degree for a given node | False |
| squaring_weights_for_degree_sum_bool  | Sum squared edge weights $w^{2}$ for degree for a given node | False |

* Parameters for the network-based regularized model:

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| use_network  | If False, we will run a standard Lasso regression model on the original $X$ and $y$ data (baseline). | True |
| fit_y_intercept_bool  | Should a y-intercept be fitted for the final model by GRegulNet | False |


if *cv_for_alpha_lasso_model_bool is False* (the default):

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| max_lasso_iterations  | the maximum # of iterations we will run Lasso regression model | 10000 |


If *cv_for_alpha_lasso_model_bool is True*:

| Parameter | Definition | Default |
| --------- | ---------- | ---------- |
| num_cv_folds  | the # of cross-validation (cv) folds we fit on training data when building model | 5 |


### Output ###

* A Fitted Estimator from the GRegulNet class.

There are several attributes available for objects in the GRegulNet class. 

We can fit our GRegulNet estimator on $X$ and $y$ training data and retrieve coefficients for the GRegulNet network-regularized linear model. Then, we can evaluate our model performance capabilities on testing data.  We evaluate our model predictive performance using the Mean Squared Error (MSE). 






## Demo (Toy Example) of GRegulNet:

Suppose we have gene expression data for 5 Transcription Factors (TFs), [$TF_{1}$, $TF_{2}$, $TF_{3}$, $TF_{4}$, and $TF_{5}$] that are our respective predictors [$X_{1}$, $X_{2}$, $X_{3}$, $X_{4}$, and $X_{5}$]. We also have gene expression data for our target gene (TG), our response variable $y$.  We want to build a model to predict the expression of our $y$ based on the gene expression data of these 5 TFs. Our GRegulNet estimator will also incorporate a prior biological network of relationships among the TFs based on a Protein-Protein Interaction (PPI) network. 


Here, we build the GRegulNet estimator based on an **undirected prior graph network** of the relationships among our 5 Transcription Factors (TFs). As shown below, there is a particularly strong relationship between $TF_{1} \leftrightarrow TF_{2}$ of 0.9 and between $TF_{4} \leftrightarrow TF_{5}$ of 0.75. We specify the remaining relationships among the other TFs to be the default edge weight of 0.1.



```python
from gregulnetClasses import * # to load our package, GRegulNet
edge_list = [[1, 2, 0.9], [4, 5, 0.75], [1, 3], [1, 4], [1, 5], 
              [2, 3], [2, 4], [2, 5], [3, 4], [3, 5]]
beta_network_val = 10
# by default, cv_for_alpha_lasso_model_bool is False, so alpha_lasso_val will be specified.
alpha_lasso_val = 0.01

# Building the network regularized regression model. 
gregulnet_demo = geneRegulatNet(edge_list = edge_list, beta_network_val = beta_network_val,
                          alpha_lasso_val = alpha_lasso_val)
```

    :) Please note that we count the number of edges with weight > 0.5 to get the degree for a given node.
    :) We also add 0.001 as a pseudocount to our degree value for each node.
    
    


    
![png](output_12_1.png)
    


We can specify our $X$ and $y$ data to train our GRegulNet object using the *DemoDataBuilderXandY* class. In our demo, we utilize the *demo_dict* dictionary below to encode values for 3 parameters for our demo. Our class generates random data based on *num_samples_M* and *corrVals*. We specify *num_samples_M* is 100 samples. Further, we want the correlations of each of our predictors with the $y$ variable to be as provided by *corrVals*: [cor(TF$_{1}$, y) = 0.9, cor(TF$_{2}$, y) = 0.5, cor(TF$_{3}$, y) = 0.1, cor(TF$_{4}$, y) = -0.2, cor(TF$_{5}$, y) = -0.8]. Since *same_train_and_test_data_bool* is False, we will separate the data with 70% for training and 30% for testing.  There are additional parameters that we can adjust for our demo, but we wanted to retain simplicity. Please note that we explain more details about this class in *DemoDataBuilderXandY_explanation.ipynb*.
    



```python
demo_dict = {"num_samples_M": 100,
            "corrVals": [0.9, 0.5, 0.1, -0.2, -0.8],
            "same_train_and_test_data_bool": False}

dummy_data = DemoDataBuilderXandY(**demo_dict)

X_train = dummy_data.X_train
y_train = dummy_data.y_train

# we can view the y_train data as a neat dataframe:
y_train_df = pd.DataFrame(y_train, columns = ["y"])
print(f"Training y (response) data: {y_train_df.shape[0]} rows. The first 5 rows:")
print(y_train_df.head())

# we can view the X_train data as a neat dataframe:
X_train_df = pd.DataFrame(X_train, columns = gregulnet_demo.tf_names_list)
print(f"\nTraining X (predictors) data: {X_train_df.shape[0]} rows for {X_train_df.shape[1]} predictors. The first 10 rows:")
X_train_df.head(10)
```

    :) Please note that since we hold out 30.0% of our 100 samples for testing, we have:
    :) X_train = 70 rows (samples) and 5 columns (N = 5 predictors) for training.
    :) X_test = 30 rows (samples) and 5 columns (N = 5 predictors) for testing.
    :) y_train = 70 corresponding rows (samples) for training.
    :) y_test = 30 corresponding rows (samples) for testing.
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 4995.60it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 5000.36it/s]
    100%|██████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 4996.79it/s]

    Training y (response) data: 70 rows. The first 5 rows:
              y
    0  0.050186
    1 -0.081154
    2 -0.622764
    3  0.783163
    4  1.295948
    
    Training X (predictors) data: 70 rows for 5 predictors. The first 10 rows:
    

    
    




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <td>0.902250</td>
      <td>0.343490</td>
      <td>-1.317465</td>
      <td>1.404691</td>
      <td>0.745724</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.936549</td>
      <td>1.110822</td>
      <td>-0.696995</td>
      <td>1.109920</td>
      <td>0.028946</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000611</td>
      <td>2.197287</td>
      <td>-0.074302</td>
      <td>0.389102</td>
      <td>2.369909</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.418952</td>
      <td>-0.594274</td>
      <td>0.705211</td>
      <td>0.316235</td>
      <td>-0.986833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.944057</td>
      <td>0.095329</td>
      <td>0.838374</td>
      <td>-0.469584</td>
      <td>-0.271817</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-3.036204</td>
      <td>-2.121818</td>
      <td>0.001455</td>
      <td>2.002789</td>
      <td>1.886874</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.718175</td>
      <td>-1.053622</td>
      <td>0.738562</td>
      <td>-0.547582</td>
      <td>0.026048</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-2.090047</td>
      <td>-1.448126</td>
      <td>0.368563</td>
      <td>1.855596</td>
      <td>0.799289</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-3.258704</td>
      <td>-0.372644</td>
      <td>1.294874</td>
      <td>0.118021</td>
      <td>1.888439</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.137793</td>
      <td>0.381576</td>
      <td>0.373323</td>
      <td>-1.683666</td>
      <td>-1.021895</td>
    </tr>
  </tbody>
</table>
</div>



To see all of the parameters (specified and default) that are used for the GRegulNet model:



```python
gregulnet_demo.parameters_df
```




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parameter</th>
      <th>data type</th>
      <th>description</th>
      <th>value</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>default_edge_weight</td>
      <td>&gt;= 0</td>
      <td>edge weight for any edge with missing weight info</td>
      <td>0.1</td>
      <td>PriorGraphNetwork</td>
    </tr>
    <tr>
      <th>1</th>
      <td>consider_self_loops</td>
      <td>boolean</td>
      <td>add 1 to the degree for each node (based on se...</td>
      <td>False</td>
      <td>PriorGraphNetwork</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pseudocount_for_diagonal_matrix</td>
      <td>&gt;= 0</td>
      <td>to ensure that no nodes have 0 degree value in...</td>
      <td>0.001</td>
      <td>PriorGraphNetwork</td>
    </tr>
    <tr>
      <th>3</th>
      <td>use_edge_weight_values_for_degrees_bool</td>
      <td>boolean</td>
      <td>if False, we use a threshold instead to derive...</td>
      <td>False</td>
      <td>PriorGraphNetwork</td>
    </tr>
    <tr>
      <th>4</th>
      <td>threshold_for_degree</td>
      <td>&gt;= 0</td>
      <td>edge weights &gt; 0.5 are counted as 1 for the de...</td>
      <td>0.5</td>
      <td>PriorGraphNetwork</td>
    </tr>
    <tr>
      <th>5</th>
      <td>model_type</td>
      <td>set of options</td>
      <td>which model type should be used for geneRegula...</td>
      <td>Lasso</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>6</th>
      <td>max_lasso_iterations</td>
      <td>&gt;= 1</td>
      <td>the maximum # of iterations for Lasso</td>
      <td>Lasso</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>7</th>
      <td>use_cross_validation_for_model_bool</td>
      <td>boolean</td>
      <td>should we use cross validation for training th...</td>
      <td>False</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>8</th>
      <td>alpha_lasso</td>
      <td>&gt;= 0</td>
      <td>value for alpha for the lasso problem</td>
      <td>0.01</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>9</th>
      <td>use_network</td>
      <td>boolean</td>
      <td>perform network regularization using a network...</td>
      <td>True</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>10</th>
      <td>beta_network</td>
      <td>&gt;= 0</td>
      <td>value of beta for the network regularization p...</td>
      <td>10</td>
      <td>GRegulNet</td>
    </tr>
    <tr>
      <th>11</th>
      <td>fit_y_intercept_bool</td>
      <td>boolean</td>
      <td>fit a y-intercept for our regression problem</td>
      <td>False</td>
      <td>GRegulNet</td>
    </tr>
  </tbody>
</table>
</div>



Here, $gregulnet_{demo}$, is an object of the *GRegulNet* class. We fit a model using the $X_{train}$ and $y_{train}$ data sets.


```python
gregulnet_demo.fit(X_train, y_train)
```

    network used
    Training GRegulNet :)
    




    <gregulnetClasses.GRegulNet at 0x1ede632e830>



To view and extract the predicted model coefficients for the predictors: 


```python
gregulnet_demo.coef
```




    array([ 0.23655573,  0.11430656,  0.00148755, -0.03512912, -0.16009479])




```python
gregulnet_demo.model_coefficients_df
```




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
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



We can test the performance of our data on new, unseen data, to understand better the generalizability of our GRegulNet model on new data. 


```python
y_test = dummy_data.y_test 
X_test = dummy_data.X_test

# we can view the y_train data as a neat dataframe:
y_test_df = pd.DataFrame(y_test, columns = ["y"])
print(f"Testing y (response) data: {y_test_df.shape[0]} rows. The first 5 rows:")
print(y_test_df.head())

# we can view the X_test data as a neat dataframe:
X_test_df = pd.DataFrame(X_test, columns = gregulnet_demo.tf_names_list)
print(f"\nTesting X (predictors) data: {X_test_df.shape[0]} rows for {X_test_df.shape[1]} predictors. The first 10 rows:")
X_test_df.head(10)
```

    Testing y (response) data: 30 rows. The first 5 rows:
              y
    0 -0.198628
    1 -2.488737
    2 -0.587804
    3  0.287809
    4  0.444063
    
    Testing X (predictors) data: 30 rows for 5 predictors. The first 10 rows:
    




<div>
<!-- <style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style> -->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
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
      <td>0.263202</td>
      <td>-0.995175</td>
      <td>0.248797</td>
      <td>-0.755467</td>
      <td>-1.484214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-7.034347</td>
      <td>-0.720368</td>
      <td>-0.535171</td>
      <td>0.095236</td>
      <td>0.607330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.259201</td>
      <td>-0.928826</td>
      <td>0.051216</td>
      <td>-0.999568</td>
      <td>2.240629</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.804365</td>
      <td>-0.698938</td>
      <td>-0.616339</td>
      <td>-1.323494</td>
      <td>-1.537090</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.675801</td>
      <td>-0.790465</td>
      <td>0.002748</td>
      <td>-0.852372</td>
      <td>-0.471373</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.388396</td>
      <td>2.192212</td>
      <td>-0.118921</td>
      <td>-0.892500</td>
      <td>-4.234699</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.504579</td>
      <td>-1.491449</td>
      <td>-0.685401</td>
      <td>-1.485232</td>
      <td>0.816566</td>
    </tr>
    <tr>
      <th>7</th>
      <td>-4.194619</td>
      <td>-2.003116</td>
      <td>-0.370393</td>
      <td>-0.719803</td>
      <td>2.062496</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.284898</td>
      <td>0.633291</td>
      <td>-1.400780</td>
      <td>-0.729766</td>
      <td>0.913916</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.296446</td>
      <td>-0.674408</td>
      <td>-0.109297</td>
      <td>-0.035722</td>
      <td>0.045178</td>
    </tr>
  </tbody>
</table>
</div>




```python
mse_test = gregulnet_demo.predict(X_test, y_test)
print(f"Please note that the testing Mean Square Error (MSE) is {mse_test}")
```

    Testing GRegulnet :)
    Please note that the testing Mean Square Error (MSE) is 0.020152051044508176
    

## References

[1]: Wilkinson, M.D., Dumontier, M., Aalbersberg, I.J. et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18

[2]: Zhang, Y., Akutsu, T., & Ching, W. K. (2016). Incorporating network topology and phylogenetic constraints for protein-protein interaction prediction. BMC bioinformatics, 17(1), 1-15. https://doi.org/10.1186/s12859-016-1310-4

[3]: Jia, C., Zhang, Y., Chen, K., & Zhang, S. (2018). A novel feature extraction method with improved consistency for identifying cell cycle regulated genes. Bioinformatics, 34(5), 896-903. https://doi.org/10.1093/bioinformatics/btx657

[4]: Lu, Y., Chen, X., & Hu, Z. (2017). Recognition of protein/gene names from text using an ensemble of classifiers and effective abbreviation resolution. BMC bioinformatics, 18(1), 1-11. https://doi.org/10.1186/s12859-017-1515-1

[5]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830.
