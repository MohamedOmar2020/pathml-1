

# Test if the pipeline function actually works
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from MultiRun import multi_run
import pandas as pd

#######


dirpath = r"data/noolan"

# Get the file names
from os import listdir
file_list = listdir(dirpath)

## Just use the .tif files
target_files = []

for fname in file_list:
    if fname.endswith('tif'):
        target_files.append(fname)

print(target_files)

#####################
## Iterate over the relevant input filenames and run the pipeline function
# Initialize empty dictionaries
all_results = {}

# Iterate over files and run pipeline for each
for fname in target_files:
    results = multi_run(dirpath, fname)
    all_results[fname] = results


## concatenate the results in 1 file
import anndata as ad
#combined = ad.concat(all_results, label="dataset")
adata_combined = ad.concat(all_results, join="outer", label="origin")

# Read channel names
chanelnames = pd.read_csv("data/noolan/channelNames.txt", header = None, dtype = str, low_memory=False)

adata_combined.var_names = chanelnames.values
adata_combined.var_names_make_unique()

## Save the combined adata file for further use
adata_combined.write_loom("data/adata_combined.loom", write_obsm_varm=True)
#adata_combined = ad.read_loom("data/adata_combined.loom", obs_names="obs_names", sparse=False)

#################################################
## Proceed with the single cell workflow
import scanpy as sc
sc.pl.violin(adata_combined, keys = ['0','24','60'])
sc.pp.log1p(adata_combined)

sc.pp.scale(adata_combined, max_value=10)
sc.tl.pca(adata_combined, svd_solver='arpack')
sc.pp.neighbors(adata_combined, n_neighbors=10, n_pcs=10)
sc.tl.umap(adata_combined)
sc.pl.umap(adata_combined, color=['0','33','60'])
sc.pl.umap(adata_combined, color='origin')

sc.tl.leiden(adata_combined, resolution = 0.15)
sc.pl.umap(adata_combined, color='leiden')
sc.tl.rank_genes_groups(adata_combined, 'leiden', method='t-test')
sc.pl.rank_genes_groups_dotplot(adata_combined, groupby='leiden', vmax=5, n_genes=5)

##############################################
import squidpy as sq
sc.set_figure_params(dpi=300, frameon=True, figsize=(8,8), format='png')
sc.pl.spatial(adata_combined, color='leiden', spot_size=15)
sc.pl.spatial(adata_combined, color="leiden",groups=["2","4"],spot_size=15)


sq.gr.spatial_neighbors(adata_combined)
sq.gr.nhood_enrichment(adata_combined, cluster_key="leiden")
sq.pl.nhood_enrichment(adata_combined, cluster_key="leiden", method="ward")
plt.show()

sq.gr.co_occurrence(adata_combined, cluster_key="leiden")
sq.pl.co_occurrence(adata_combined, cluster_key="leiden")
plt.show()