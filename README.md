# MVMTMDA
Codes and datasets for the paper entitle "Predicting microRNA-disease associations from lncRNA-microRNA interactions via multi-view multi-task learning"

To run the python codes, there are several python package needed to be installed, which include Tensorflow, numpy and math.

The codes includes two files, DataSet.py and Model.py. The former is to deal with data processing while the later is to predict the scores for each pair of MDA. The outputs contain three files: 1) s_MDA.txt is a 799x268 matrix for scoring all the MDAs; 2) s_LMI.txt is a 541X268 scoring matrix for LMI; 3) s_MicroRNA_Embedding.txt presents the embedding features for 268 types of microRNAs.

The Data folder contains four files. 
1) LMI_dataset.xlsx describes the LMI collected from the LncRNASNP.v2 database. The file shows the matching ID for each type of lncRNA and microRNA in tables of "lncRNA_ID" and "microRNA_ID".
2) MDA_dataset.xlsx describes the MDA collected from the HMDD.v3 database, giving the matching ID of diseases in "disease ID" table.
3) The rest two files, "lncRNA-miRNA_id.txt" and "miRNA-disease_id.txt", are the id list of known LMI and MDA and are used as the inputs for our codes.

Guidelines for use: Step 1) add the LMI and MDA data for any query microRNA in "lncRNA-miRNA_id.txt" and "miRNA-disease_id.txt", resepectively; Step 2) run the model files; Step 3) obtain the predicted scores of the query microRNA to all disease by inquiring the ID in "MDA_dataset.xlsx" files.
