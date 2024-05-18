# By Saniya Khullar :)
# Please note the code to select the list of candidate Transcription Factors (TFs) for a given target gene (TG) in the cell-type
# for the Alzheimer's disease (AD) versus controls application
gc()
library(dplyr)
library(stringr)
# user parameters :)

info_type = "AD" # Control
cell_type = "Microglia"

if (cell_type == "Microglia"){
  abbrev = "Mic"
} else if (cell_type == "Excitatory_Neuron") {
  abbrev = "Ex"
} else if (cell_type == "Inhibitory_Neuron") {
  abbrev = "In"
} else if (cell_type == "OPCs") {
  abbrev = "Opc"
} else if (cell_type == "Pericytes") {
  abbrev = "Per"
} else if (cell_type == "Astrocytes") {
  abbrev = "Ast"
} else if (cell_type == "Endothelial") {
  abbrev = "End"
} else if (cell_type == "Endo_BBB") {
  abbrev = "End"
} else if (cell_type == "Oligodendrocytes") {
  abbrev = "Oli"
}
print(":) by Saniya")
print(abbrev)
print(info_type)
gene_expression_folder_FP = paste0("C://Users//saniy//Documents//netrem_application2//Gene_Expression//train_test//")
gc()
# https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-3450-3
library(dplyr)
library(stringr)
# user parameters :)

if (cell_type == "Microglia"){
  abbrev = "Mic"
} else if (cell_type == "Excitatory_Neuron") {
  abbrev = "Ex"
} else if (cell_type == "Inhibitory_Neuron") {
  abbrev = "In"
} else if (cell_type == "OPCs") {
  abbrev = "Opc"
} else if (cell_type == "Pericytes") {
  abbrev = "Per"
} else if (cell_type == "Astrocytes") {
  abbrev = "Ast"
} else if (cell_type == "Endothelial") {
  abbrev = "End"
} else if (cell_type == "Endo_BBB") {
  abbrev = "End"
} else if (cell_type == "Oligodendrocytes") {
  abbrev = "Oli"
}
print(":) by Saniya")
print(abbrev)
print(info_type)
gene_expression_folder_FP = paste0("E://final_preprocessed_datasets//Gene_Expression//train_test//")
gexpr_RData_FN_train = paste0(gene_expression_folder_FP, "MIT_", info_type, "_",
                        abbrev, "_imputed_gexpr_train.RData")
library(stringr)
gexpr_RData_FN_test = paste0(gene_expression_folder_FP, "MIT_", info_type, "_",
                              abbrev, "_imputed_gexpr_test.RData")
print(gexpr_RData_FN_train)
print(gexpr_RData_FN_test)
load(gexpr_RData_FN_train)
load(gexpr_RData_FN_test)

train_data[1:4, 1:5]
train_data = train_data[,-ncol(train_data)]
test_data = test_data[,-ncol(train_data)]

X_train_all = train_data
X_test_all = test_data
y_train_all = train_data
y_test_all = test_data


global_TF_removal_based_on_gene_expression_values = 40 # please note we initially remove TFs with expression levels below
# the 40th percentile overall (amongst all other genes/TFs in the dataset) in training data

min_human_gene_expression_percentile = 40 # minimum expression percentile in training data (for humans)
min_percentile_to_use = 75 # for filtering step 2 monalisa reference network of predicted TF-TG relationships 
low_num_TFs_for_model = 10
high_num_TFs_for_model = 80

###### other_file_paths
tiny_ppi_df_FP = "E://final_preprocessed_datasets//input_netrem_2024//final_filtered_PPI_MIT_AD_Control_CellTypeData_BySaniya.csv" #"D://updatedMOST_PPI_for_usage_bySaniya_complexes_and_interactions.csv" #updatedMOST_PPI_for_usage_bySaniya_complexes_and_interactions.csv" # C://Users//saniy//Downloads//netrem_folder//preprocessed_string_human_ppi_network_edge_list_v11.5_bySaniya_withCORUMandContextHumanPPI_mostUpdated.csv" #"C://Users//saniy//Downloads//netrem_folder//preprocessed_string_human_ppi_network_edge_list_v11.5_bySaniya_withCORUM_mostUpdated.csv"
tf_complexes_FP = "C://Users//saniy//Documents//updatedNEWER_complexes_to_consider.csv" #updated_complexes_to_consider.csv" #C://Users//saniy//Downloads//updated_TF_complexes_with_CORUMandContextualPPI.csv" # "C://Users//saniy//Downloads//updated_TF_complexes_with_CORUM.csv"
gene_metrics_FP = paste0("E://final_preprocessed_datasets//Monalisa_binding//", cell_type, "_target_gene_metrics_from_cobinding_work_by_SaniyaNEW.csv") # D://netrem_july//monalisa//target_gene_metrics_from_cobinding_work_Schwann_cells_by_SaniyaNEW.csv" #target_gene_metrics_from_cobinding_work_Schwann_cells_by_Saniya.csv"
scgrnom_step2_FP = paste0("E://final_preprocessed_datasets//scgrnom_binding//scgrnom_", cell_type, "_most_updated.parquet")# "D://netrem_july//scgrnom_schwann_combined_results_July2023_bySaniya.csv"

tf_coloc_FP = paste0("E://final_preprocessed_datasets//input_netrem_2024//",
                     cell_type, "final_newest_organized_tf_nonoverlapping_monaLisaMinScore8_TFcolocalization_ppiFilterWithKegg_NEW.csv") # "D://netrem_july//final_newest_organized_tf_nonoverlapping_schwannButUpTo_7celltypes_monaLisaMinScore8_TFcolocalization_bySaniyaK_withKeggAndPPIFilter.csv"# "C://Users//saniy//Documents//organized_tf_nonoverlapping_schwannButUpTo_7celltypes_monaLisaMinScore7point5_TF_updated_colocalization_bySaniyaK_filteredByPPI_withKegg.csv" #"D://netrem_july//final_newest_organized_tf_nonoverlapping_schwannButUpTo_7celltypes_monaLisaMinScore8_TFcolocalization_bySaniyaK.csv" #C://Users//saniy//Documents//organized_tf_nonoverlapping_schwannButUpTo_7celltypes_monaLisaMinScore7point5_TF_updated_colocalization_bySaniyaK_filteredByPPI.csv"

tf_molSim_FP = paste0("E://final_preprocessed_datasets//input_netrem_2024//", 
                      cell_type, "newerest_withKegg_ppiFiltering_tf_sim_mf_", cell_type, "_NEW.csv") # "C://Users//saniy//Documents//newerest_withKegg_ppiFiltering_tf_sim_mf_Schwann.csv" #newerest_ppiFiltering_tf_sim_mf_Schwann.csv" #newerest_tf_sim_mf_Schwann.csv" #melted_tf_mf_sim_df.csv"


jac_sim_binding_FP = paste0("E://final_preprocessed_datasets//input_netrem_2024//",
                            cell_type, "edge_list_tf_cobind_jaccard_sim_raw_atacseq_regulatory_region_overlaps_ppiFilterWithKegg_NEW.csv")  # "C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya_ppiFiltering_withKegg.csv"# "C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya_ppiFiltering.csv" #"C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya.csv"

cobinding_metrics_FP =  paste0("E://final_preprocessed_datasets//most_updated_final_datasets//", cell_type, "//",
                               cell_type, "_COMBO_overall_metrics_for_TF_binding_monalisa_adults_bySaniya.parquet")  # "C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya_ppiFiltering_withKegg.csv"# "C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya_ppiFiltering.csv" #"C://Users//saniy//Documents//edge_list_tf_cobinding_jaccard_similarity_regulatory_region_overlaps_Schwann_rawATACseq_scgrnomStep2_bySaniya.csv"

library(arrow)


# dataFrame[rows, columns]
X_train_all[is.na(X_train_all)] <- 0
X_test_all[is.na(X_test_all)] <- 0

X_train_all[is.na(X_train_all)] <- 0
X_test_all[is.na(X_test_all)] <- 0

gene_expression_genes = colnames(X_train_all)#[-1]
gene_expression_genes = gene_expression_genes[order(gene_expression_genes)]
head(gene_expression_genes)

print(paste0(":) Please note that Saniya had held out 30% of the data for testing."))
print(paste0(":) # of gene expression genes for ", cell_type, " = " , length(gene_expression_genes)))
print(paste0(":) X_train dimensions (rows = # of cell samples, columns = # of genes): ", nrow(X_train_all)))
print(paste0(":) X_test dimensions (rows = # of cell samples, columns = # of genes): ", nrow(X_test_all)))

#################################################

filtered_data = X_train_all

head(filtered_data)
X_train_all[1:4, 1:4]

gene_expression_avg = apply(filtered_data, 2, mean)
gene_expression_avg_no_na <- replace(gene_expression_avg, is.na(gene_expression_avg), 0)
expression_threshold = quantile(gene_expression_avg_no_na, global_TF_removal_based_on_gene_expression_values/100)
# Replace missing values with 0

print(paste0(":) Please note that we will find the ", global_TF_removal_based_on_gene_expression_values, 
             "% expression threshold in the entire training human gene expression data: ", expression_threshold))
print("We will use that minimum threshold for filtering for TFs in the training data, instead of for filtering TGs.")
gene_expression_avg_df <- data.frame(avg_expression = gene_expression_avg,
                                     gene = names(gene_expression_avg))
genes_below_threshold_df <- gene_expression_avg_df[gene_expression_avg_df$avg_expression <= expression_threshold, ]

human_genes_below_threshold <- genes_below_threshold_df$gene
print(length(human_genes_below_threshold))

common_low_express_genes = unique(human_genes_below_threshold) #intersect(human_genes_below_threshold, mouse_genes_below_threshold)
print(paste0(":) # of lowly expressed genes in training human gene expression data: ", 
             length(human_genes_below_threshold)))

print(paste0(":) # of common lowly expressed genes in both training human gene expression data and mouse gene expression data: ", 
             length(common_low_express_genes)))
genes_below_threshold = unique(human_genes_below_threshold) #c(human_genes_below_threshold, mouse_genes_below_threshold))
print(paste0(":) # of overall lowly expressed genes in training human gene expression data: ", 
             length(genes_below_threshold)))
print(paste0(":) Thus, please note that we will remove ANY TFs that are in this list of ", length(genes_below_threshold), 
             " genes."))
length(genes_below_threshold)


###########

human_gene_expression_avg_train_df = gene_expression_avg_df
human_gene_expression_avg_train_df$organism = "human"
human_gene_expression_avg_train_df$cell_type = cell_type
human_gene_expression_avg_train_df$info_type = info_type
human_gene_expression_avg_train_df$data_source = "training_X_data"
head(human_gene_expression_avg_train_df)
human_gene_expression_avg_train_df = human_gene_expression_avg_train_df[order(human_gene_expression_avg_train_df$avg_expression,
                                                                              decreasing = TRUE),]
human_gene_expression_avg_train_df$rank_num = rank(-1 * human_gene_expression_avg_train_df$avg_expression)
head(human_gene_expression_avg_train_df)

###########################################################
major_gene_df = read.csv(gene_metrics_FP, header = TRUE)[,-1]
head(major_gene_df)
num_cobinding_genes = length(unique(major_gene_df$gene))
print(paste0(":) Please note that there are ", num_cobinding_genes, 
             " target genes (TGs) identified by Saniya's cobinding method"))
major_gene_df = major_gene_df[which(major_gene_df$gene %in% gene_expression_genes),]
updated_num_cobinding_genes = length(unique(major_gene_df$gene))
print(paste0(":) # of genes dropped: ", num_cobinding_genes - updated_num_cobinding_genes))
print(paste0(":) After filtering for genes found in our gene expression data, please note that there are ",
             updated_num_cobinding_genes, " target genes identified by Saniya's cobinding method")) # 13418 
num_cobinding_genes = length(unique(major_gene_df$gene))
print(paste0(":) Please note that there are ", num_cobinding_genes, 
             " target genes identified by Saniya's cobinding method")) # 12490 

# complexes
tf_complexes_df = read.csv(tf_complexes_FP, header = TRUE)[,-1]  # 1139483
print(paste0(":) # of rows BEFORE filtering TF complexes dataset: ", nrow(tf_complexes_df)))
tf_complexes_df = tf_complexes_df[which(tf_complexes_df$TF1 %in% gene_expression_genes),]
tf_complexes_df = tf_complexes_df[which(tf_complexes_df$TF2 %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression dataset
tf_complexes_df = tf_complexes_df[-which(tf_complexes_df$TF1 %in% genes_below_threshold),]
tf_complexes_df = tf_complexes_df[-which(tf_complexes_df$TF2 %in% genes_below_threshold),]
print(paste0(":) # of rows AFTER filtering TF complexes: ", nrow(tf_complexes_df)))
tf_complexes_df$TF_TF_link = paste0(tf_complexes_df$TF1, "_", tf_complexes_df$TF2)
head(tf_complexes_df)

# Gene Ontology Molecular Function similarity of TFs can be helpful :)
melted_tf_mf_sim_df = read.csv(tf_molSim_FP, header = TRUE)[,-1] #.drop(columns = ["Unnamed: 0"])
print(paste0(":) # of rows BEFORE filtering TF-TF molecular similarity dataset by Saniya: ", nrow(melted_tf_mf_sim_df)))  # 110454
head(melted_tf_mf_sim_df)
melted_tf_mf_sim_df = melted_tf_mf_sim_df[which(melted_tf_mf_sim_df$TF1 %in% gene_expression_genes),]
melted_tf_mf_sim_df = melted_tf_mf_sim_df[which(melted_tf_mf_sim_df$TF2 %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression datasets
melted_tf_mf_sim_df = melted_tf_mf_sim_df[-which(melted_tf_mf_sim_df$TF1 %in% genes_below_threshold),]
melted_tf_mf_sim_df = melted_tf_mf_sim_df[-which(melted_tf_mf_sim_df$TF2 %in% genes_below_threshold),]
melted_tf_mf_sim_df  = melted_tf_mf_sim_df[order(melted_tf_mf_sim_df$weight, 
                                                 decreasing = TRUE),]
print(paste0(":) # of rows AFTER filtering TF-TF molecular similarity dataset by Saniya: ", nrow(melted_tf_mf_sim_df))) # 43716


# for TFs that have experimentally-found complexes they bind to, we can utilize the binding data results
# to help us select better the TFs to add to the machine learning model (Based on TFs they form complexes with
# that they share strong colocalization in our cell-type(s) of interest with)
complex_combo_df = inner_join(tf_complexes_df, melted_tf_mf_sim_df)
complex_combo_df = complex_combo_df[order(complex_combo_df$weight, decreasing = TRUE),] 

complex_combo_df2 = complex_combo_df
complex_combo_df2$TF_TF_link = paste0(complex_combo_df2$TF2, "_", complex_combo_df2$TF1)
colnames(complex_combo_df2)[1:2] = c("TF2", "TF1")
head(complex_combo_df2)
complex_combo_df = unique(rbind(complex_combo_df, complex_combo_df2))
complex_combo_df2 = NULL
complex_combo_df_approach0 = complex_combo_df
complex_combo_df_approach0 = complex_combo_df_approach0[which(complex_combo_df_approach0$TF1 %in% gene_expression_genes),]
complex_combo_df_approach0 = complex_combo_df_approach0[which(complex_combo_df_approach0$TF2 %in% gene_expression_genes),]
head(complex_combo_df_approach0)
complex_combo_df_approach0 = complex_combo_df_approach0[,c(1, 2, 5,6)]
complex_combo_df_approach0 = unique(complex_combo_df_approach0)
complex_combo_df_approach0 = complex_combo_df_approach0[order(complex_combo_df_approach0$weight, decreasing = TRUE),] 

tfs_in_complex_list0 = unique(complex_combo_df_approach0$TF1)
length(tfs_in_complex_list0) # 362



###############################################################################################
# Filtering the PPI...
###############################################################################################

# Approach 1 for finding which TFs may exhibit co-binding with each other (colocalization)
# way to add in more TFs as predictors to the model: for a given TF, we look at TFs that share the highest "count" with it (that is, the highest # of common regions of non-overlapped motif binding across 
# cell-type-specific raw scATAC-seq regions found in at most 1 of 7 similar cell-types)


colocTF_df = read.csv(tf_coloc_FP, header = TRUE)[,-1] #columns = ["Unnamed: 0"])
head(colocTF_df)
print(paste0(":) # of rows BEFORE filtering Monalisa Colocalization dataset: ", nrow(colocTF_df))) # 164674
colocTF_df = colocTF_df[which(colocTF_df$TF1 %in% gene_expression_genes),]
colocTF_df = colocTF_df[which(colocTF_df$TF2 %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression dataset
colocTF_df = colocTF_df[-which(colocTF_df$TF1 %in% genes_below_threshold),]
colocTF_df = colocTF_df[-which(colocTF_df$TF2 %in% genes_below_threshold),]
print(paste0(":) # of rows AFTER filtering Monalisa Colocalization dataset: ", nrow(colocTF_df)))  # 74420
colocTF_df  = colocTF_df[order(colocTF_df$JacSim, decreasing = TRUE),]
head(colocTF_df)
row.names(colocTF_df) = NULL


# for TFs that have experimentally-found complexes they bind to, we can utilize the binding data results
# to help us select better the TFs to add to the machine learning model (Based on TFs they form complexes with
# that they share strong colocalization in our cell-type(s) of interest with)
tf_complexes_df = tf_complexes_df[,c(1, 2)]
complex_combo_df = inner_join(tf_complexes_df, colocTF_df)
complex_combo_df = complex_combo_df[order(complex_combo_df$count, decreasing = TRUE),] 

complex_combo_df2 = complex_combo_df
complex_combo_df2$TF_TF_link = paste0(complex_combo_df2$TF2, "_", complex_combo_df2$TF1)
colnames(complex_combo_df2)[1:2] = c("TF2", "TF1")
head(complex_combo_df2)
complex_combo_df = unique(rbind(complex_combo_df, complex_combo_df2))
complex_combo_df2 = NULL
complex_combo_df_approach1 = complex_combo_df
complex_combo_df_approach1 = complex_combo_df_approach1[which(complex_combo_df_approach1$TF1 %in% gene_expression_genes),]
complex_combo_df_approach1 = complex_combo_df_approach1[which(complex_combo_df_approach1$TF2 %in% gene_expression_genes),]
head(complex_combo_df_approach1)
#complex_combo_df_approach1 = complex_combo_df_approach1[,c(1, 2, 3, 4,9, 10)]
complex_combo_df_approach1 = unique(complex_combo_df_approach1)
complex_combo_df_approach1 = complex_combo_df_approach1[order(complex_combo_df_approach1$JacSim, decreasing = TRUE),] 

tfs_in_complex_list1 = unique(complex_combo_df_approach1$TF1)
length(tfs_in_complex_list1) # 330



# Approach 2 for finding which TFs may exhibit co-binding with each other (colocalization)
# another way to add in more TFs as predictors to the model: for a given TF, we look at TFs that share the highest
# "value" with it (that is, the highest Jaccard Similarity value based on binding to the same regulatory regions in the 
# raw General Schwann Cell scATACseq data. Here, Saniya did NOT consider overlapping versus non-overlapping motif positions. 
# The motif database was also smaller: JASPAR2022)
js = read.csv(jac_sim_binding_FP, header = TRUE)[,-1]
colnames(js)[1:2] = c("TF1", "TF2")
js = js[order(js$Jaccard.Similarity, decreasing = TRUE),] 
dim(js)
js_mini = js[which(js$TF1 %in% gene_expression_genes),]
js_mini = js_mini[which(js_mini$TF2 %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression dataset
js_mini = js_mini[-which(js_mini$TF1 %in% genes_below_threshold),]
js_mini = js_mini[-which(js_mini$TF2 %in% genes_below_threshold),]
dim(js_mini)
js_mini2 = js_mini
colnames(js_mini2)[1:2] = c("TF2", "TF1")

js_mini = unique(rbind(js_mini, js_mini2))
dim(js_mini)
js_mini2 = NULL
js_mini$TF_TF_link = paste0(js_mini$TF1, "_", js_mini$TF2)
js_mini = js_mini[order(js_mini$Jaccard.Similarity, decreasing = TRUE),] 

js_mini



# for TFs that have experimentally-found complexes they bind to, we can utilize the binding data results
# to help us select better the TFs to add to the machine learning model (Based on TFs they form complexes with
# that they share strong colocalization in our cell-type(s) of interest with)
complex_combo_df = inner_join(tf_complexes_df, js_mini)
complex_combo_df = complex_combo_df[order(complex_combo_df$Jaccard.Similarity, decreasing = TRUE),] 

complex_combo_df2 = complex_combo_df
complex_combo_df2$TF_TF_link = paste0(complex_combo_df2$TF2, "_", complex_combo_df2$TF1)
colnames(complex_combo_df2)[1:2] = c("TF2", "TF1")
head(complex_combo_df2)
complex_combo_df = unique(rbind(complex_combo_df, complex_combo_df2))
complex_combo_df2 = NULL
complex_combo_df_approach2 = complex_combo_df
complex_combo_df_approach2 = complex_combo_df_approach2[which(complex_combo_df_approach2$TF1 %in% gene_expression_genes),]
complex_combo_df_approach2 = complex_combo_df_approach2[which(complex_combo_df_approach2$TF2 %in% gene_expression_genes),]
complex_combo_df_approach2 = complex_combo_df_approach2[order(complex_combo_df_approach2$Jaccard.Similarity, decreasing = TRUE),] 
head(complex_combo_df_approach2)

tfs_in_complex_list2 = unique(complex_combo_df_approach2$TF1)
length(tfs_in_complex_list2) # 237


#################################################
### Input Reference network:

# This is the input cobinding prediction of TFs that are predicted to regulate a given
# target gene (TG). Saniya has captured various metrics. Sometimes a gene (Target Gene: TG) 
# will have some scATACseq raw regions in Schwann Cells that are found in only Schwann Cells 
# (or at most 6 other similar cell-types), and then, cell-type-specific metrics can be
# used to prioritize TFs predicted to bind to those cell-type-specific regions. 
# Other times, a gene will NOT have any cell-type-specific regions, and then we can 
# look at the behavior of the TFs overall.
cobinding_metrics_FP <- gsub("//", "\\\\", cobinding_metrics_FP)
monalisa_cobinding_df = read_parquet(cobinding_metrics_FP) 
head(monalisa_cobinding_df)
monalisa_cobinding_df = as.data.frame(monalisa_cobinding_df)
#monalisa_cobinding_df = read.csv(cobinding_metrics_FP, header = TRUE)[,-1]#.drop(columns = ["Unnamed: 0"]) # 9064113 rows × 27 columns
dim(monalisa_cobinding_df)
library(dplyr)
complex_rows_df = monalisa_cobinding_df[grep("[::]", monalisa_cobinding_df$TF),] 
dim(complex_rows_df) # 854774     28
# Filter the rows based on the conditions
filtered_df <- complex_rows_df %>%
  mutate(TF1 = ifelse(grepl("::", TF), sub("::.*", "", TF), TF),
         TF2 = ifelse(grepl("::", TF), sub(".*::", "", TF), TF)) %>%
  filter(TF1 %in% gene_expression_genes & TF2 %in% gene_expression_genes 
         & !(TF1 %in% genes_below_threshold) & !(TF2 %in% genes_below_threshold))
monalisa_cobinding_df$TF_TG
dim(filtered_df)
head(filtered_df) # 334727     30
#setdiff(unique(monalisa_cobinding_df$TF), unique(filtered_df$TF))
# monalisa_cobinding_df = monalisa_cobinding_df[-grep("[::]", monalisa_cobinding_df$TF),]
print(paste0(":) # of rows BEFORE filtering Monalisa TF-TG Reference dataset: ", nrow(monalisa_cobinding_df))) # 164674

to_add_df = monalisa_cobinding_df[which(monalisa_cobinding_df$TF_TG %in% filtered_df$TF_TG),]
monalisa_cobinding_df = monalisa_cobinding_df[which(monalisa_cobinding_df$gene %in% gene_expression_genes),]
monalisa_cobinding_df = monalisa_cobinding_df[which(monalisa_cobinding_df$TF %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression dataset
monalisa_cobinding_df = monalisa_cobinding_df[-which(monalisa_cobinding_df$TF %in% genes_below_threshold),]
monalisa_cobinding_df = monalisa_cobinding_df[which(monalisa_cobinding_df$TF != monalisa_cobinding_df$gene),]
dim(monalisa_cobinding_df) # 4002110      28
monalisa_cobinding_df = rbind(monalisa_cobinding_df, to_add_df)
dim(monalisa_cobinding_df) # 4336837      28
monalisa_cobinding_df = monalisa_cobinding_df[which(monalisa_cobinding_df$TF != monalisa_cobinding_df$gene),]
dim(monalisa_cobinding_df) # 4336837      28
monalisa_cobinding_df = unique(monalisa_cobinding_df)
print(paste0(":) # of rows AFTER filtering Monalisa TF-TG Reference dataset: ", nrow(monalisa_cobinding_df))) # 4002110
head(monalisa_cobinding_df) # 10156551 to 6134222 


scgrnom_step2_FP <- gsub("//", "\\\\", scgrnom_step2_FP)
scgrnom_step2_df = read_parquet(scgrnom_step2_FP) 
scgrnom_step2_df = as.data.frame(scgrnom_step2_df)
# scgrnom_step2_df = read.csv(scgrnom_step2_FP, header = TRUE)[,-1] # 
print(paste0(":) # of rows BEFORE filtering scgrnom Step 2 TF-TG Reference dataset: ", nrow(scgrnom_step2_df))) # 45464821
# scgrnom_step2_df = scgrnom_step2_df[which(scgrnom_step2_df$TF != scgrnom_step2_df$TG),]
scgrnom_step2_df = scgrnom_step2_df[which(scgrnom_step2_df$TF != scgrnom_step2_df$gene),]
scgrnom_step2_df = scgrnom_step2_df[which(scgrnom_step2_df$gene %in% gene_expression_genes),]
#scgrnom_step2_df = scgrnom_step2_df[which(scgrnom_step2_df$TG %in% gene_expression_genes),]
scgrnom_step2_df = scgrnom_step2_df[which(scgrnom_step2_df$TF %in% gene_expression_genes),]
# removing any TFs that have low expression in the gene expression dataset
scgrnom_step2_df = scgrnom_step2_df[-which(scgrnom_step2_df$TF %in% genes_below_threshold),]
print(paste0(":) # of rows AFTER filtering scgrnom Step 2 TF-TG Reference dataset: ", nrow(scgrnom_step2_df))) # 8145251
head(scgrnom_step2_df) # 24156608


cell_type_specific_TGs = unique(major_gene_df[which(major_gene_df$percent_cell_type_specific_regions > 0),]$gene)
length(cell_type_specific_TGs) # 3260

non_cell_type_specific_TGs = unique(major_gene_df[which(major_gene_df$percent_cell_type_specific_regions == 0),]$gene)
length(non_cell_type_specific_TGs) # 9230

total_cobinding_TGs = unique(major_gene_df$gene)
num_cobinding_TGs = length(total_cobinding_TGs)

total_cobinding_TGs = total_cobinding_TGs[order(total_cobinding_TGs)]
head(total_cobinding_TGs)


outRData = paste0("E://final_preprocessed_datasets//netremRData//netremRData_bySaniya_",
                  info_type, "_", cell_type, "_March2024.RData")
save.image(outRData) 
genes_and_tfs_list = list() # 1389, 7592

scgrnom_step2_df = cbind(scgrnom_step2_df$TF, scgrnom_step2_df$gene)
scgrnom_step2_df = data.frame(scgrnom_step2_df)
colnames(scgrnom_step2_df) = c("TF", "gene")
head(scgrnom_step2_df)
dim(scgrnom_step2_df) # 21337666        2
scgrnom_step2_df = unique(scgrnom_step2_df)
dim(scgrnom_step2_df) # 21337666        2

rm(gene_expression_avg_df)
rm(train_data)
rm(filtered_data)
rm(y_test_all)
rm(y_train_all)
rm(test_data)
rm(X_test_all)
rm(filtered_df)
pNum = 1
for (gene_num in 1:length(total_cobinding_TGs)){
  print(paste0(":) gene_num = ", gene_num, " of ", 
               length(total_cobinding_TGs)))
  tg = total_cobinding_TGs[gene_num]
  tg
  
  rows1 = which(monalisa_cobinding_df$gene == tg)
  tfs_for_the_tg = monalisa_cobinding_df[rows1,]
  
  monalisa_tfs = unique(tfs_for_the_tg$TF)
  found_complexes = FALSE
  complexes_found_vec = monalisa_tfs[grep("[::]", monalisa_tfs)]
  if (length(complexes_found_vec) > 0){
    found_complexes = TRUE
  }
  monalisa_tfs = unlist(strsplit(monalisa_tfs, "[::]"))
  
  
  head(tfs_for_the_tg)
  print(paste0(":) # of monalisa TFs for TG ", tg, ": ", length(monalisa_tfs)))
  
  rows2 = which(scgrnom_step2_df$gene == tg)
  other_tfs = unique(scgrnom_step2_df[rows2,]$TF)
  print(paste0(":) # of scgrnom part 2 TFs for TG ", tg, ": ", length(other_tfs)))
  
  common_tfs = intersect(monalisa_tfs, other_tfs)
  cell_type_specific_bool = FALSE # does this TG have Schwann cell-type-specific peaks (found in at most 1 of the 7 key cell types)
  min_percentile_to_use_overall = min_percentile_to_use
  if (tg %in% cell_type_specific_TGs){
    cell_type_specific_bool = TRUE
    min_percentile_to_use_overall = min_percentile_to_use + 5
  }
  tfs_list_for_tg = monalisa_tfs
  
  full_tfs = unique(c(monalisa_tfs, other_tfs))
  complexes_to_add_vec = unique(tf_complexes_df[which(tf_complexes_df$TF1 %in% full_tfs),]$TF2)
  full_tfs = intersect(colnames(X_train_all), unique(c(full_tfs, complexes_to_add_vec)))
  X_tfs_train = X_train_all[full_tfs]
  
  
  gene_expression_avg = apply(X_tfs_train, 2, mean)
  gene_expression_avg_no_na <- replace(gene_expression_avg, is.na(gene_expression_avg), 0)
  expression_threshold = quantile(gene_expression_avg_no_na, min_human_gene_expression_percentile/100)
  # Replace missing values with 0
  

  gene_expression_avg_df <- data.frame(avg_expression = gene_expression_avg,
                                       gene = names(gene_expression_avg))
  genes_above_threshold_df <- gene_expression_avg_df[gene_expression_avg_df$avg_expression >= expression_threshold, ]
  
  TFs_for_TG_above_threshold <- genes_above_threshold_df$gene
  print(length(TFs_for_TG_above_threshold))
  
  
  tfs_for_the_tg = unique(tfs_for_the_tg[which(tfs_for_the_tg$TF %in% TFs_for_TG_above_threshold),])
  
  common_tfs = intersect(common_tfs, unique(tfs_for_the_tg$TF)) # found in scgrnom and monalisa
  
  if (cell_type_specific_bool == TRUE){
    final_percentile_to_use = min_percentile_to_use + 5
    
  } else {
    final_percentile_to_use = min_percentile_to_use
  }
  
  thresh3 = quantile(as.numeric(tfs_for_the_tg$num_overall_TG_regions_bound_by_TF), final_percentile_to_use/100)
  tfs_for_the_tg_p3 = tfs_for_the_tg[which(tfs_for_the_tg$num_overall_TG_regions_bound_by_TF 
                                           >= thresh3),]
  
  thresh4 = quantile(as.numeric(tfs_for_the_tg$overall_maxMotif_score_median), 
                     final_percentile_to_use/100)
  tfs_for_the_tg_p4 = tfs_for_the_tg[which(tfs_for_the_tg$overall_maxMotif_score_median 
                                           >= thresh4),]
  
  thresh5 = quantile(as.numeric(tfs_for_the_tg$overall_maxMotif_score_mean), 
                     final_percentile_to_use/100)
  tfs_for_the_tg_p5 = tfs_for_the_tg[which(tfs_for_the_tg$overall_maxMotif_score_mean 
                                           >= thresh5),]
  
  
  shared_TFs = intersect(tfs_for_the_tg_p3$TF, tfs_for_the_tg_p4$TF)
  shared_TFs = intersect(shared_TFs, tfs_for_the_tg_p5$TF)
  
  
  if (cell_type_specific_bool == TRUE){
    thresh1 = quantile(as.numeric(tfs_for_the_tg$cellTypeSpecific_maxMotif_score_median), final_percentile_to_use/100)
    tfs_for_the_tg_p1 = tfs_for_the_tg[which(tfs_for_the_tg$cellTypeSpecific_maxMotif_score_median 
                                             >= thresh1),]
    
    thresh2 = quantile(as.numeric(tfs_for_the_tg$percent_TG_cell_type_specific_regions_bound_by_TF), 
                       final_percentile_to_use/100)
    tfs_for_the_tg_p2 = tfs_for_the_tg[which(tfs_for_the_tg$percent_TG_cell_type_specific_regions_bound_by_TF 
                                             >= thresh2),]
    
    thresh6 = quantile(as.numeric(tfs_for_the_tg$cellTypeSpecific_maxMotif_score_mean), 
                       final_percentile_to_use/100)
    tfs_for_the_tg_p6 = tfs_for_the_tg[which(tfs_for_the_tg$cellTypeSpecific_maxMotif_score_mean 
                                             >= thresh6),]
    
    shared_TFs = intersect(shared_TFs, tfs_for_the_tg_p6$TF)
    shared_TFs = intersect(shared_TFs, tfs_for_the_tg_p1$TF)
    shared_TFs = intersect(shared_TFs, tfs_for_the_tg_p2$TF)
  }
  
  best_tfs = unique(c(shared_TFs, common_tfs))

  low_TFs_bool = TRUE
  high_TFs_bool = FALSE
  if (length(best_tfs) >= low_num_TFs_for_model){
    low_TFs_bool = FALSE
  } else if (length(best_tfs) >= high_num_TFs_for_model){
    high_TFs_bool = TRUE
  }
  
  tfs_added_vec = c()
  for (k in 1:length(best_tfs)){
    
    tf_name = best_tfs[k]
    # Molecular Function TF - TF similarity
    if (tf_name %in% tfs_in_complex_list0){
      melty = complex_combo_df_approach0
    } else{
      melty = melted_tf_mf_sim_df
    }
    
    melty = melty[which(melty$TF1 == tf_name),]
    melty = melty[-which(melty$TF2 %in% best_tfs),] 
    max_TF_score = max(melty$weight)
    melty = unique(melty[which(melty$weight == max_TF_score),]$TF2)
    
    if (tf_name %in% tfs_in_complex_list1){# then, we filter for the TF based on other TFs it forms complex with
      minner = complex_combo_df_approach1
    } else { 
      minner = colocTF_df
    }
    
    minner = minner[which(minner$TF1 == tf_name),]
    minner = minner[-which(minner$TF2 %in% best_tfs),] 
    maxVal = max(as.numeric(minner$JacSim))
    max_minner = unique(minner[which(as.numeric(minner$JacSim) == maxVal),]$TF2)
    
    if (tf_name %in% tfs_in_complex_list2){ # then, we filter for the TF based on other TFs it forms complex with
      jacy = complex_combo_df_approach2
    } else{
      jacy = js_mini
    }
    
    jacy = jacy[which(jacy$TF1 == tf_name),]
    jacy = jacy[-which(jacy$TF2 %in% best_tfs),] 
    
    maxJacy = max(as.numeric(jacy$Jaccard.Similarity))
    max_Jacy = unique(minner[which(as.numeric(jacy$Jaccard.Similarity) == maxJacy),]$TF2)
    
    tfs_added_vec = unique(c(tfs_added_vec, melty, max_minner, max_Jacy))
  }
  
  tfs_added_vec <- as.character(na.omit(tfs_added_vec))
  
  
  if (found_complexes){ # adding any TFs found to operate in a complex from monalisa
    to_check_complexes = paste(best_tfs, collapse = "|")
    best_complex_vec = complexes_found_vec[grep(paste(to_check_complexes, collapse = "|"), complexes_found_vec)]
    
    # best_complex_vec = complexes_found_vec[grep(paste(best_complex_vec, collapse = "|"), complexes_found_vec)]
    best_complex_vec = unlist(strsplit(best_complex_vec, "[::]"))
    best_complex_vec <- best_complex_vec[nzchar(best_complex_vec)]
    best_complex_vec = unique(best_complex_vec)
    best_tfs = unique(c(best_tfs, best_complex_vec))
  }
  
  tfs_for_tg_final_vec = unique(c(best_tfs, tfs_added_vec))
  
  print(paste0(":) FINAL # of TFs for TG: ", length(tfs_for_tg_final_vec)))
  print("##############################################")
  print("")
  tg_col = rep(tg, length(tfs_for_tg_final_vec))
  df_to_add = data.frame(tg_col, tfs_for_tg_final_vec)
  colnames(df_to_add) = c("TG", "TF")
  head(df_to_add)
  dim(df_to_add)
  
  genes_and_tfs_list[[gene_num]] = df_to_add
  
  if (gene_num %% 100 == 0){
    intermed_path = paste0("intermediate_tfs_for_tgs_", info_type, "_", cell_type, ".RData")
    save(genes_and_tfs_list, gene_num, file = intermed_path)
    
  }
}
genes_and_tfs_df = do.call(rbind, genes_and_tfs_list)
outCSV = paste0("E://final_preprocessed_datasets//netremRData//netrem_genes_and_tfs_df_bySaniya_", info_type, "_", cell_type, "_March2024_p", pNum, ".csv")
write.csv(genes_and_tfs_df, outCSV)