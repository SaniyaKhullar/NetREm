<!-- ---
layout: default
---

{% include mathjax.html %} -->

# NetREm
## Network regression embeddings reveal cell-type transcription factor interactions for gene regulation
<!-- ##### GRegNet Gene Regular(ized/atory) Network -->

### By: Saniya Khullar, Xiang Huang, Raghu Ramesh, John Svaren, Daifeng Wang
[Daifeng Wang Lab](https://daifengwanglab.org/) <br>

## Simulation used in the paper



## Demo (Toy Example) of NetREm:
Our goal is to build a machine learning model to predict the gene expression levels of our target gene (TG) $y$ based on the gene expression levels of $N = 5$ Transcription Factors (TFs) [TF<sub>1</sub>, $TF_{2}$, $TF_{3}$, $TF_{4}$, $TF_{5}$] in a particular cell-type. Assume the gene expression values for each TF are [X<sub>1</sub>, $X_{2}$, $X_{3}$, $X_{4}$, $X_{5}$], respectively. We generate $M = 100,000$ random samples (rows) of data where the Pearson correlations ($r$) between gene expression of each TF ($X$) with gene expression of TG $y$ as *corrVals*: [cor(TF<sub>1</sub>, $y$) = 0.9, cor(TF<sub>2</sub>, $y$) = 0.5, cor(TF<sub>3</sub>, $y$) = 0.3, cor(TF<sub>4</sub>, $y$) = -0.2, cor(TF<sub>5</sub>, $y$) = -0.8]. 

The dimensions of $X$ are therefore 100,000 rows by 5 columns (predictors). More details about our *generate_dummy_data* function (and additional parameters we can adjust for) are in [Dummy_Data_Demo_Example.ipynb](https://github.com/SaniyaKhullar/NetREm/blob/main/Dummy_Data_Demo_Example.ipynb). Our NetREm estimator also incorporates a constraint of an **undirected weighted prior graph network** of biological relationships among only 5 TFs based on a weighted Protein-Protein Interaction (PPI) network ([TF<sub>1</sub>, $TF_{2}$, $TF_{3}$, $TF_{4}$, $TF_{5}$]), where higher edge weights $w$ indicate stronger biological interactions at the protein-level.

The code for this demo example is [demo_toy.py](https://github.com/SaniyaKhullar/NetREm/blob/main/demo/demo_toy.py) in the *demo* folder.

```python 
from DemoDataBuilderXandY import generate_dummy_data
from Netrem_model_builder import netrem
import PriorGraphNetwork as graph
import error_metrics as em 
import essential_functions as ef
import netrem_evaluation_functions as nm_eval

dummy_data = generate_dummy_data(corrVals = [0.9, 0.5, 0.3, -0.2, -0.8],
                                 num_samples_M = 100000,
                                 train_data_percent = 70)
```
The Python console or Jupyter notebook will  print out the following:

    same_train_test_data = False
    Please note that since we hold out 30.0% of our 100 samples for testing, we have:
    X_train = 70 rows (samples) and 6 columns (N = 6 predictors) for training.
    X_test = 30 rows (samples) and 6 columns (N = 6 predictors) for testing.
    y_train = 70 corresponding rows (samples) for training.
    y_test = 30 corresponding rows (samples) for testing.

The $X$ data should be in the form of a Pandas dataframe as below:

```python
X_df = dummy_data.X_df
X_df.head()
```
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
      <th>TF6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.020840</td>
      <td>0.594445</td>
      <td>-1.443012</td>
      <td>-0.688777</td>
      <td>0.900770</td>
      <td>-2.643671</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.224776</td>
      <td>-0.270632</td>
      <td>-0.557771</td>
      <td>-0.305574</td>
      <td>0.054708</td>
      <td>-1.054197</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.746721</td>
      <td>1.502236</td>
      <td>2.043813</td>
      <td>1.252975</td>
      <td>2.082159</td>
      <td>1.227615</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.558130</td>
      <td>1.290771</td>
      <td>-1.230527</td>
      <td>-0.678410</td>
      <td>0.630084</td>
      <td>-1.508758</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-2.181462</td>
      <td>-0.657229</td>
      <td>-2.880186</td>
      <td>-1.629470</td>
      <td>0.268042</td>
      <td>1.207254</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_df = dummy_data.y_df
y_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.601721</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.151619</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.359462</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.222055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.775868</td>
    </tr>
  </tbody>
</table>
</div>

```python
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
netrem_demo = netrem(edge_list = edge_list, 
                     beta_net = beta_network_val,
                     alpha_lasso = alpha_lasso_val,
                     view_network = True)

# Fitting the NetREm model on training data: X_train and y_train:
netrem_demo.fit(X_train, y_train)
```

    
![png](output_3_1.png)
    



    
![png](output_3_2.png)
    


   1 new node(s) added to network based on gene expression data ['TF6']
    

    
![png](output_3_5.png)

![png](netrem_estimator.PNG)

<!-- There is a particularly strong relationship between $TF_{1} \leftrightarrow TF_{2}$ of 0.9 and between $TF_{4} \leftrightarrow TF_{5}$ of 0.75. The remaining relationships among the other TFs is assumed to be the default (edge weight of 0.1). -->
<!-- Here, $gregulnet_{demo}$ is an object of the *GRegulNet* class. We fit a model using $X_{train}$ and $y_{train}$ data (70 samples). -->

To view and extract the predicted model coefficients for the predictors: 

<!-- ```python
gregulnet_demo.coef
```

    array([ 0.23655573,  0.11430656,  0.00148755, -0.03512912, -0.16009479]) -->


```python
netrem_demo.model_coef_df
```

<!-- <div>
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
</div> -->

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
      <th>TF6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>0.309776</td>
      <td>0.112297</td>
      <td>0.001116</td>
      <td>-0.073603</td>
      <td>-0.21665</td>
      <td>0.000375</td>
    </tr>
  </tbody>
</table>
</div>

In the context of gene regulation (in biology), we predict that predictors with negative NetREm coefficients for target gene (TG) $y$ may be repressors (their activity focuses on reducing expression of $y$) and those with positive coefficients for $y$ may be activators. 

To view the TG-specific TF-TF interactome that NetREm learned for this target gene $y$, in our given cell-type, we can view the `B_interaction_df`.  

```python
netrem_demo.B_interaction_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
      <th>TF6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TF1</th>
      <td>4.512391</td>
      <td>0.856197</td>
      <td>-0.515540</td>
      <td>-0.262458</td>
      <td>-2.315411</td>
      <td>-1.166994</td>
    </tr>
    <tr>
      <th>TF2</th>
      <td>0.856197</td>
      <td>1.644936</td>
      <td>-0.342697</td>
      <td>0.169874</td>
      <td>-0.570895</td>
      <td>-0.494075</td>
    </tr>
    <tr>
      <th>TF3</th>
      <td>-0.515540</td>
      <td>-0.342697</td>
      <td>84.228863</td>
      <td>-0.470847</td>
      <td>-0.618672</td>
      <td>-16.642297</td>
    </tr>
    <tr>
      <th>TF4</th>
      <td>-0.262458</td>
      <td>0.169874</td>
      <td>-0.470847</td>
      <td>1.218198</td>
      <td>0.070268</td>
      <td>-0.619841</td>
    </tr>
    <tr>
      <th>TF5</th>
      <td>-2.315411</td>
      <td>-0.570895</td>
      <td>-0.618672</td>
      <td>0.070268</td>
      <td>2.505441</td>
      <td>-0.163531</td>
    </tr>
    <tr>
      <th>TF6</th>
      <td>-1.166994</td>
      <td>-0.494075</td>
      <td>-16.642297</td>
      <td>-0.619841</td>
      <td>-0.163531</td>
      <td>84.577403</td>
    </tr>
  </tbody>
</table>
</div>

We predict that positive scores between predictors (e.g. $TF_1$ and $TF_2$ have a score of 0.856197) tend to imply potential cooperativity (e.g. cobinding) among them. Negative scores may suggest antagonistic activity between the TFs (e.g. $TF_1$ and $TF_5$ have a score of -2.315411), where these TFs may compete to regulate the $TG$ through biological mechanisms that may be investigated further through experiments.


We can test the performance of our data on testing data $X_{test}$ ($M = 30$ samples), to understand better the generalizability of our NetREm model on new, unseen, data. 


```python
pred_y_test = netrem_demo.predict(X_test) # predicted values for y_test
mse_test = netrem_demo.test_mse(X_test, y_test)

print(f"The testing Mean Square Error (MSE) is {mse_test}")
```

    The testing Mean Square Error (MSE) is 0.10939471847175668


We can analyze more metrics about our NetREm model results as below: 

```python
netrem_demo.final_corr_vs_coef_df
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>info</th>
      <th>input_data</th>
      <th>TF1</th>
      <th>TF2</th>
      <th>TF3</th>
      <th>TF4</th>
      <th>TF5</th>
      <th>TF6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>network regression coeff. with y: y</td>
      <td>X_train</td>
      <td>0.309776</td>
      <td>0.112297</td>
      <td>0.001116</td>
      <td>-0.073603</td>
      <td>-0.21665</td>
      <td>0.000375</td>
    </tr>
    <tr>
      <th>0</th>
      <td>corr (r) with y: y</td>
      <td>X_train</td>
      <td>0.90244</td>
      <td>0.440681</td>
      <td>0.058722</td>
      <td>-0.163072</td>
      <td>-0.814564</td>
      <td>-0.252551</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Absolute Value NetREm Coefficient Ranking</td>
      <td>X_train</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>


In the context of gene regulation, our results may thereby be interpreted in the [following way](https://github.com/SaniyaKhullar/NetREm/blob/main/netrem_final_demo.png).

Nonetheless, NetREm can be applied to solve a suite of regression problems where there is an underlying connection among the predictors and their correlation with one another may be utilized jointly for the predictive task rather than discarded. 

We also provide a suite of evaluation functions and explanations of more advanced functionalities in our [User Guide](https://github.com/SaniyaKhullar/NetREm/blob/main/user_guide/).





## References

[1]: Caiyan Li, Hongzhe Li, Network-constrained regularization and variable selection for analysis of genomic data, Bioinformatics, Volume 24, Issue 9, May 2008, Pages 1175–1182, https://doi.org/10.1093/bioinformatics/btn081

[2]: Wilkinson, M.D., Dumontier, M., Aalbersberg, I.J. et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18

[3]: Zhang, Y., Akutsu, T., & Ching, W. K. (2016). Incorporating network topology and phylogenetic constraints for protein-protein interaction prediction. BMC bioinformatics, 17(1), 1-15. https://doi.org/10.1186/s12859-016-1310-4

[4]: Jia, C., Zhang, Y., Chen, K., & Zhang, S. (2018). A novel feature extraction method with improved consistency for identifying cell cycle regulated genes. Bioinformatics, 34(5), 896-903. https://doi.org/10.1093/bioinformatics/btx657

[5]: Lu, Y., Chen, X., & Hu, Z. (2017). Recognition of protein/gene names from text using an ensemble of classifiers and effective abbreviation resolution. BMC bioinformatics, 18(1), 1-11. https://doi.org/10.1186/s12859-017-1515-1

[6]: Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12(Oct), 2825-2830.
