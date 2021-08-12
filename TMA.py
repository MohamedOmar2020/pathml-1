

# load libraries and data
from pathml.core.slide_data import VectraSlide
from pathml.core.slide_data import CODEXSlide
from pathml.core.slide_data import MultiparametricSlide
from pathml.core import types

from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import SegmentMIF, QuantifyMIF, CollapseRunsCODEX

import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
from deepcell.utils.plot_utils import make_outline_overlay
from deepcell.utils.plot_utils import create_rgb_image

import scanpy as sc
import squidpy as sq
sc.set_figure_params(dpi=200, frameon=True, figsize=(12,9), format='png')


slidedata = CODEXSlide('/Volumes/Mohamed/TMA_A/bestFocus/reg033_X01_Y01_Z09.tif')

# These tif are of the form (x,y,z,c,t) but t is being used to denote cycles
# 17 z-slices, 4 channels per 23 cycles, 70 regions
slidedata.slide.shape

img = slidedata.slide.extract_region(location = (0,0), size = (1920, 1440))
img.shape


pipe = Pipeline([
    CollapseRunsCODEX(z=0),
    SegmentMIF(model='mesmer', nuclear_channel=0, cytoplasm_channel=29, image_resolution=0.377442),
    QuantifyMIF(segmentation_mask='cell_segmentation')
])

client = Client(n_workers = 10)
slidedata.run(pipe, distributed = False, client = client, tile_size=1000, tile_pad=False, overwrite_existing_tiles=True)

#######
img = slidedata.tiles[0].image

for i in range(92):
    plt.imshow(img[:,:,i])
    plt.show()

########
def plot(slidedata, tile, channel1, channel2):
    image = np.expand_dims(slidedata.tiles[tile].image, axis=0)
    nuc_segmentation_predictions = np.expand_dims(slidedata.tiles[tile].masks['nuclear_segmentation'], axis=0)
    cell_segmentation_predictions = np.expand_dims(slidedata.tiles[tile].masks['cell_segmentation'], axis=0)
    #nuc_cytoplasm = np.expand_dims(np.concatenate((image[:,:,:,channel1,0], image[:,:,:,channel2,0]), axis=2), axis=0)
    nuc_cytoplasm = np.stack((image[:,:,:,channel1], image[:,:,:,channel2]), axis=-1)
    rgb_images = create_rgb_image(nuc_cytoplasm, channel_colors=['blue', 'green'])
    overlay_nuc = make_outline_overlay(rgb_data=rgb_images, predictions=nuc_segmentation_predictions)
    overlay_cell = make_outline_overlay(rgb_data=rgb_images, predictions=cell_segmentation_predictions)
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(rgb_images[0, ...])
    ax[1].imshow(overlay_cell[0, ...])
    ax[0].set_title('Raw data')
    ax[1].set_title('Cell Predictions')
    plt.show()

# DAPI + Syp
plot(slidedata, tile=0, channel1=0, channel2=29)

# DAPI + CD44
plot(slidedata, tile=0, channel1=29, channel2=13)

################################################
## explore the single-cell quantification of the imaging data.
#adata = slidedata.tiles[0].counts.to_memory().copy('path')

adata = slidedata.counts.to_memory().copy('path')
adata
adata.X
adata.obs
adata.var

################################################
## Scanpy workflow
import scanpy as sc
sc.pl.violin(adata, keys = ['0','24','60', '29', '33'])
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=10)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['0','29', '33'])

sc.tl.leiden(adata, resolution = 0.15)
sc.pl.umap(adata, color='leiden')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups_dotplot(adata, groupby='leiden', vmax=5, n_genes=2)

##############################################
import squidpy as sq
sc.set_figure_params(dpi=300, frameon=True, figsize=(15,10), format='png')
sc.pl.spatial(adata, color='leiden', spot_size=15)
sc.pl.spatial(adata, color="leiden", groups=["2","4"], spot_size=15)





