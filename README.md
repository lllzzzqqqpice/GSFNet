# **Geometric and Semantic Fusion Network "GSFNet"**

## Introduction
Dense matching and semantic segmentation are critical yet challenging tasks in the processing of Very High Resolution (VHR) satellite imagery, particularly within complex urban environments. Traditional approaches often treat these tasks independently, failing to address their inherent limitations: semantic segmentation struggles to distinguish spectrally similar objects , while dense matching falters in low-texture regions where photometric consistency breaks down. To address these complementary challenges, we propose the Geometric and Semantic Fusion Network

##  Use
### Environment

 - Python 3.8
 - TensorFlow 2.10.0

### Install
numpy 1.22.0

scipy 1.4.1

pillow 9.5.0
### Data Preparation
Download [US3D Datasets](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)
### Training for US3D
`python GSFNet.py`
