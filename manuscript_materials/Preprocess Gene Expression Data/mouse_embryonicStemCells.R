fp = "C://Users//saniy//Downloads//mouse//mESCs_combined_mouse.csv"
data <- read.csv(fp)[,-1]
data[1:5, 1:5]
row.names(data) = data$CellId
data = data[,-1]
data[1:5, 1:5]
# Xkr4 Rp1 Sox17 Mrpl15 Lypla1
# ggattggtacagaacgtg    0   0     0      5     15
# ccgtaagtcggcggattg    0   0     0      3     28
# caccacagtaaaaattgg    0   0     0      0      2
# caagtcgagtgaacggac    0   0     0      5     10
# ggacgagagcttgactcg    0   0     0     15      7

data = t(data)
data[1:5, 1:5]
# ggattggtacagaacgtg ccgtaagtcggcggattg caccacagtaaaaattgg caagtcgagtgaacggac
# Xkr4                    0                  0                  0                  0
# Rp1                     0                  0                  0                  0
# Sox17                   0                  0                  0                  0
# Mrpl15                  5                  3                  0                  5
# Lypla1                 15                 28                  2                 10
# ggacgagagcttgactcg
# Xkr4                    0
# Rp1                     0
# Sox17                   0
# Mrpl15                 15
# Lypla1                  7
#Seurat expects cell barcodes as columns, and features (genes) as rows.
library("Seurat")
seurat_object <- CreateSeuratObject(counts = data)
seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)
# Performing log-normalization
# 0%   10   20   30   40   50   60   70   80   90   100%
#   [----|----|----|----|----|----|----|----|----|----|
#      **************************************************|
seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst")
# Calculating gene variances
# 0%   10   20   30   40   50   60   70   80   90   100%
#   [----|----|----|----|----|----|----|----|----|----|
#      **************************************************|
#      Calculating feature variances of standardized and clipped values
#    0%   10   20   30   40   50   60   70   80   90   100%
#      [----|----|----|----|----|----|----|----|----|----|
#         **************************************************|
seurat_object <- ScaleData(seurat_object, features = rownames(seurat_object))
# Centering and scaling data matrix
# |===========================================================================================| 100%
expr  = seurat_object@assays$RNA@data 
exprMat = as.matrix(expr) # t(as.matrix(expr))
dim(exprMat) # [1] 24421  1080 ; rows = genes, columns = samples
#selected_feat = rownames(exprMat)[Matrix::rowSums(exprMat)>0] # making sure the gene is expressed positively# control samples
#exprMat = exprMat[which(row.names(exprMat) %in% selected_feat),]
#exprMat = exprMat[-1,]
dim(expr)
dim(exprMat) # 17441  1080
exprMat[1:5, 1:5]
write.csv(exprMat, "mouse_ESCs_logNormalizedAndScaled_updatedAndRedoneBySaniya.csv")

uhoh = read.csv("mouse_ESCs_logNormalizedAndScaled_updatedAndRedoneBySaniya.csv", header = TRUE)


protein.info.v12.0 <- read.delim("C:/Users/saniy/Downloads/mouse/10090.protein.info.v12.0.txt", header = TRUE)
head(protein.info.v12.0)



protein.links.v12.0 <- read.delim("C:/Users/saniy/Downloads/mouse/10090.protein.links.v12.0.txt", sep = " ", header = TRUE)
head(protein.links.v12.0)
# X.string_protein_id
p1_df = protein.info.v12.0[,c(1, 2)]
colnames(p1_df) = c(colnames(protein.links.v12.0)[1], "Mouse_Gene_1")

p2_df = protein.info.v12.0[,c(1, 2)]
#colnames(p2_df)[1] = colnames(protein.links.v12.0)[2]
colnames(p2_df) = c(colnames(protein.links.v12.0)[2], "Mouse_Gene_2")


head(p1_df)
head(p1_df)

library(dplyr)
df1 = inner_join(protein.links.v12.0, p1_df)#, on = "protein1")
dim(protein.links.v12.0) #[1] 12684354        3
dim(df1) #[1]  12684354        4

df1 = inner_join(df1, p2_df)
dim(df1) #[1]  12684354        5

head(df1)
df1$gene1_gene2 = paste0(df1$Mouse_Gene_1, "_", df1$Mouse_Gene_2)
head(df1)

# Step 1: Standardize gene pairs
# Sort the gene names in each pair so that gene1_gene2 is the same as gene2_gene1
df1$standardized_gene_pair <- apply(df1[, c("Mouse_Gene_1", "Mouse_Gene_2")], 1, function(x) paste(sort(x), collapse = "_"))

write.csv(df1, "mouse_updated_links.csv")

# Step 2: Calculate the average combined_score for each standardized gene pair
average_scores <- df1 %>%
  group_by(standardized_gene_pair) %>%
  summarize(average_combined_score = mean(combined_score))

# View the result
head(average_scores)




#seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst")
# seurat_object <- ScaleData(seurat_object)
# seurat_object <- RunPCA(seurat_object, features = VariableFeatures(object = seurat_object))
seurat_object

















#########################################################
protein.info.v11.0 <- read.delim("C:/Users/saniy/Downloads/mouse/10090.protein.info.v12.0.txt", header = TRUE)#read.delim("C://Users//saniy//Downloads//10090.protein.info.v11.0.txt", sep = "\t", header = TRUE)
head(protein.info.v11.0)



protein.links.v11.0 <- read.delim("C:/Users/saniy/Downloads/10090.protein.links.v11.0.txt", sep = " ", header = TRUE)
head(protein.links.v11.0)
# X.string_protein_id
p1_df = protein.info.v11.0[,c(1, 2)]
colnames(p1_df) = c(colnames(protein.links.v11.0)[1], "Mouse_Gene_1")

p2_df = protein.info.v11.0[,c(1, 2)]
#colnames(p2_df)[1] = colnames(protein.links.v11.0)[2]
colnames(p2_df) = c(colnames(protein.links.v11.0)[2], "Mouse_Gene_2")


head(p1_df)
head(p1_df)

library(dplyr)
df1 = inner_join(protein.links.v11.0, p1_df)#, on = "protein1")
dim(protein.links.v11.0) #[1] 12,684,354        3 --> 11,944,806        3
dim(df1) #[1]  12684354        4

df1 = inner_join(df1, p2_df)
dim(df1) #[1]  12684354        5

head(df1)
df1$gene1_gene2 = paste0(df1$Mouse_Gene_1, "_", df1$Mouse_Gene_2)
head(df1)

# Step 1: Standardize gene pairs
# Sort the gene names in each pair so that gene1_gene2 is the same as gene2_gene1
#df1$standardized_gene_pair <- apply(df1[, c("Mouse_Gene_1", "Mouse_Gene_2")], 1, function(x) paste(sort(x), collapse = "_"))
#head(df1)
write.csv(df1, "mouse_updated_links_v11.csv")

# Step 2: Calculate the average combined_score for each standardized gene pair
average_scores <- df1 %>%
  group_by(standardized_gene_pair) %>%
  summarize(avg_combo_score = mean(combined_score))
write.csv(average_scores, "average_ppi_links_old_stringdb_mouse_v11.csv")


average_scores <- df1 %>%
  group_by(standardized_gene_pair) %>%
  summarize(average_combined_score = mean(combined_score))

# View the result
head(average_scores)