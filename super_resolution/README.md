# Super resolution
This module includes codes implemented for the super resoltuion experiments in our paper.

## Shallow models
#### Introduction
We downloaded the matlab implementations for the shallow models from [this page](https://naotoyokoya.com/Publications.html).
This research is published in the following paper:
>N. Yokoya, C. Grohnfeldt, and J. Chanussot, ”Hyperspectral and multispectral data fusion: a comparative review of the recent literature,” IEEE Geoscience and Remote Sensing Magazine, vol. 5, no. 2, pp. 29-56, 2017.

The modified script "shallow_models/augsburg_experiments.m" can reproduce the experiments that we report in our paper.

#### Environment
The implementations of shallow models are tested with Matlab 2020b lisenced by TU München.

## Deep models
#### Introduction
We fetched the original implementations of deep learning models from [this repo](https://github.com/hw2hwei/SSRNET).
The related research is published in the following paper:
> X. Zhang, W. Huang, Q. Wang, and X. Li, “SSR-NET: Spatial-Spectral Reconstruction Network for Hyperspectral and Multispectral Image Fusion,”  IEEE Transactions on Geoscience and Remote Sensing (T-GRS), 2020.

"main_ResTNet.py" and "main.py" respectively reproduce the experiments of ResTNet and SSRNET which are reported in our paper.

#### Environment
**Step 1:** Install conda (version 4.9.1, where the code were tested)
[Please refer to the anaconda documentation](https://docs.anaconda.com/anaconda/install/)

**Step 2:** Create the identical conda env as our testing env
```
cd deep_models
conda env create -f environment.yml #Create env from yml file
conda activate augsburg_superresolution #activate the conda env
```

**Step 3:** Run the experiments

SSR-NET experiments 
```
cd deep_models/SSRNET-master/
python main.py
```
ResTFNet experiments
```
cd deep_models/SSRNET-master/
python main_ResTNet.py
```
The trained models are saved in:
```
deep_models/trained_model/
```

