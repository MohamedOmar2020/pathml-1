
import matplotlib.pyplot as plt
from pathlib import Path
from pathml.core.slide_dataset import SlideDataset
from pathml.core.slide_data import HESlide
import h5py
from pathml.core.h5path import read

## Load the slide
wsi = HESlide("data/CMU-1.svs", name = "example")

# How big is the slide at different levels (resolutions)? ## Level 0 is the highest resolution
for i in range(3):
    print(f"level {i}:\t{wsi.slide.get_image_shape(level = i)}")


# See a thumbnail of the slide
thumbnail = wsi.slide.get_thumbnail(size = (500, 500))

plt.imshow(thumbnail)
plt.axis("off")
plt.show()


# Look at some specific regions
region0 = wsi.slide.extract_region(location = (32000, 8000), size = (2000, 2000))
region1 = wsi.slide.extract_region(location = (33000, 8500), size = (500, 500))
region2 = wsi.slide.extract_region(location = (33000, 8500), size = (100, 100))

fix, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 8))

ax[0].imshow(region0)
ax[1].imshow(region1)
ax[2].imshow(region2)

for a in ax: a.axis("off")
plt.show()

###################################################################
## Define the pipeline:
from pathml.preprocessing.pipeline import Pipeline
from pathml.preprocessing.transforms import BoxBlur, TissueDetectionHE

## Blurring: A larger kernel width yields a more blurred result for all blurring transforms.
## TissueDetectionHE: is a Transform for detecting regions of tissue from an H&E image. It is composed by applying a sequence of other Transforms: first a median blur, then binary thresholding, then morphological opening and closing, and finally foreground detection.

pipeline = Pipeline([
    BoxBlur(kernel_size=15),
    TissueDetectionHE(mask_name = "tissue", min_region_size=500, threshold=30, outer_contours_only=True)
])

########################################################
## Run the pipline
wsi.run(pipeline)

## Look at the results
print(f"Total number of tiles extracted: {len(wsi.tiles)}")

fix, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 7.5))

for c, tile_index in enumerate([60, 75, 100]):
    t = wsi.tiles[tile_index]
    ax[0, c].imshow(t.image)
    ax[1, c].imshow(t.masks["tissue"])

ax[0, 0].set_ylabel("Tile Image", fontsize = 20)
ax[1, 0].set_ylabel("Detected Tissue", fontsize = 20)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout()
plt.show()


wsi.tiles.h5manager.h5.keys()

########################################################
## Save to disk
wsi.write('data/CMU1slidedataobject.h5path')


## Look at the h5 file
f = h5py.File('data/CMU1slidedataobject.h5path', 'r')
f.keys()






