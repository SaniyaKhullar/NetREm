fp = "C://Users//saniy//Downloads//Buenrostro_Hematopoiesis//Buenrostro_Hematopoiesis//rawdata//buenrostro_2018_scRNAseq.txt"

data_set = read.csv(fp, sep = "\t", row.names = 1)
head(data_set)

hscs_df = data_set[,grep("hsc", colnames(data_set))] # 2,268
View(hscs_df)
# #Seurat expects cell barcodes as columns, and features (genes) as rows.

hscs_df[1:5, 1:5]
data_set[1:5, 1:5]
library("Seurat")
seurat_object <- CreateSeuratObject(counts = hscs_df)
seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)
seurat_object <- FindVariableFeatures(seurat_object, selection.method = "vst")
seurat_object <- ScaleData(seurat_object, features = rownames(seurat_object))

expr  = seurat_object@assays$RNA@data 
exprMat = as.matrix(expr) #t(as.matrix(expr))
dim(exprMat) # 12558 genes (rows) for 2268 columns (samples)

selected_feat = rownames(exprMat)[Matrix::rowSums(exprMat)>3] # making sure the genes are expressed in at least 3 control samples
#selected_feat = rownames(exprMat)[Matrix::rowSums(exprMat)>0] # making sure the gene is expressed positively# control samples
exprMat = exprMat[which(row.names(exprMat) %in% selected_feat),]
dim(exprMat) # [1] 12223  2268

write.csv(exprMat, "buenrostro_et_al_hematopoeisis_hsc_processed_scRNAseq_bySaniya.csv")
