
# load libraries and data
from pathml.core.slide_data import VectraSlide
from pathml.core.slide_dataset import SlideDataset
from pathml.core.slide_dataset import VectraSlideDataset

from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import SegmentMIF, QuantifyMIF, CollapseRunsCODEX

import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client
from deepcell.utils.plot_utils import make_outline_overlay
from deepcell.utils.plot_utils import create_rgb_image

import scanpy as sc
import squidpy as sq
from pathlib import Path

# assuming that all WSIs are in a single directory, all with .svs file extension
data_dir = Path("data/noolan/")
slidedata_paths = list(data_dir.glob("*.tif"))

# create a list of SlideData objects by loading each path
slidedata_list = [VectraSlide(p) for p in slidedata_paths]

# initialize a SlideDataset
dataset = SlideDataset(slidedata_list)
