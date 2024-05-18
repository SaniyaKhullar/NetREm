
protein.info.v12.0 <- read.delim("C:/Users/saniy/Downloads/9606.protein.links.v12.0.txt/9606.protein.info.v12.0.txt", header = TRUE)
head(protein.info.v12.0)
protein.links.v12.0 <- read.delim("C://Users//saniy//Downloads//9606.protein.links.v12.0.txt//9606.protein.links.v12.0.txt", sep = " ", header = TRUE)
head(protein.links.v12.0)
p1_df = protein.info.v12.0[,c(1, 2)]
colnames(p1_df) = c(colnames(protein.links.v12.0)[1], "Human_Gene_1")

p2_df = protein.info.v12.0[,c(1, 2)]
colnames(p2_df) = c(colnames(protein.links.v12.0)[2], "Human_Gene_2")
head(p1_df)
head(p1_df)
library(dplyr)
df1 = inner_join(protein.links.v12.0, p1_df)#, on = "protein1")
dim(protein.links.v12.0) #[1] [1] 13715404        3
dim(df1) #[1] [1] 13715404        3
df1 = inner_join(df1, p2_df)
dim(df1) #[1]  13715404        5

head(df1)
df1$gene1_gene2 = paste0(df1$Human_Gene_1, "_", df1$Human_Gene_2)
head(df1)
write.csv(df1, "human_v12_links.csv")