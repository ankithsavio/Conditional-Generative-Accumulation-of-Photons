# Conditional GAP for Image Inverse Problems

This repo contains 3 folders corresponding to the modules presented in the thesis. 

The Folder contains necessary files to train and run the CGAP for their respective inverse problem.

BinomDataset: Is the GAP forward process that creates training pairs for the model.

CGAP_UNET: Is the modified Conditional UNET.

Inference: Is the photon sampling process for Geneartive or Diversity Denoising.

Inference_cascade: Is the photon sampling process for the Cascaded models, it adopts shifting through multiple cascade based on noise level.

CGAP-Train: Notebook to train the CGAP model.

CGAP-Test: Notebook to test the CGAP model.

The Full CGAP model can be downloaded from the following [link](https://www.kaggle.com/models/weedoo/full-conditional-gap-models/)


The Cascaded CGAP model are to be downloaded separately : [Inpainting](https://www.kaggle.com/models/asanchithsavio/inpaint-cascade-m40to2.5/), [Colorization](https://www.kaggle.com/models/asanchithsavio/color-cascade/), [Super Resolution](https://www.kaggle.com/models/asanchithsavio/super-res-cascase-256-256/)

Training Dataset : [FFHQ 256x256](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only)

Full models can be trained directly using the Train Notebook.
Cascade model should be trained for different minpsnr and maxpsnr. [minpsnr, maxpsnr]
Model 1: [-40, -30], Model 2: [-30, -20], Model 3: [-20, -10], Model 4: [-10, 0], Model 5: [0, 10].