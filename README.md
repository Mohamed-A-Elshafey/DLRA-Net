# DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution

## Table of Contents

1. [Introduction](#introduction)
2. [Repository Files Description](#Repository Files Description)
3. [Usage](#usage)
4. [Contact](#Contact)
5. [Reference](#Reference)
   
## Introduction

Hyperspectral Images (HSIs) provide detailed scene insights using extensive spectral bands, crucial for material discrimination and earth observation with substantial costs and low spatial resolution. Recently, Convolutional Neural Networks (CNNs) are common choice for Spectral Super-Resolution (SSR) from Multispectral Images (MSIs). However, they often fail to simultaneously exploit pixel-level noise degradation of MSIs and complex contextual spatial-spectral characteristics of HSIs. In this paper, a Deep Local Residual Attention Network with Contextual Refinement Network (DLRA-Net) is proposed to integrate local low-rank spectral and global contextual priors for improved SSR. Specifically, SSR is unfolded into Contextual-attention Refinement Module (CRM) and Dual Local Residual Attention Module (DLRAM). CRM is proposed to adaptively learn complex contextual priors to guide the convolution layer weights for improved spatial restorations. While DLRAM captures deep refined texture details to enhance contextual priors representations for recovering HSIs. Moreover, lateral fusion strategy is designed to integrate the obtained priors among DLRAMs for faster network convergence. Experimental results on natural-scene datasets with practical noise patterns confirm exceptional DLRA-Net performance with relatively small model size. DLRA-Net demonstrates Maximum Relative Improvements (MRI) between 9.71 and 58.58% in Mean Relative Absolute Error (MRAE) with reduced parameters between 52.18 and 85.85%. Besides, a practical RS-HSI dataset is generated for evaluations showing MRI between 8.64 and 50.56% in MRAE. Furthermore, experiments with HSI classifiers indicate improved performance of reconstructed RS-HSIs compared to RS-MSIs, with MRI in Overall Accuracy (OA) between 7.10 and 15.27%. Lastly, a detailed ablation study assesses model complexity and runtime.

## Repository Files Description
```
DLRA-Net/code/
├── Training                              
├── Testing                 
└── Valid Datasets            
```
#### 1- Training
The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. It utilizes both CNNs and LSTMs within a cascading architecture to efficiently process image data by leveraging spatial and spectral feature extraction methods. The proposed hybrid (CNN-LSTM) model combines the Spatial feature extraction capability of CNNs with the shared output feature and sequential context representation of LSTMs (Temporal feature extraction) to extract SPATIAL features to create an effective latent compact representation. The convolution layer is responsible for extracting spatial features from the input image, The proposed CNN output is refined forward via a flatten layer, which converts all of the resulting multidimensional arrays into a single long continuous linear vector from pooled feature maps, the input of three layers of stacked LSTM cells, respectively. An LSTM cell comprises three gates- input, output, and forget. The sequential latent representation represents the compressed data

#### 2- Testing     
The proposed model has two main components: a forward network(encoder) and a Backward network (decoder). We focused especially on separating spectral-spatial feature extraction blocks, which form the core of the SSFE network. The spectral and spatial features are combined into a spatial-spectral feature representation. The outputs of these blocks are then (Feature Fusion) concatenated and fed into a Downsampling Stage. The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. The proposed hybrid SSFE model merges one directional CNN as a spatial block and LSTM as a spectral block in parallel paths, in which the CNN path focuses on spatial feature extraction, whereas the LSTM path is dedicated to spectral feature extraction.

#### 3- Valid Datasets      
The propsoed model architecture is presented for both RGB and Multispectral EuroSat dataset. CNNs are adept at extracting spatial features from RGB images where spectral details are less critical. In contrast, for multispectral image compression, a standard CNN may ignore vital spectral information that is essential to these types of data. To address the mentioned issue, we propose a two-directional CNN approach ( this method allows the convolutional kernel to independently extract spatial features along the two parallel pathways, in which spatial features are extracted from two different directions, and makes full use of the correlations between rows and between columns of each pixel.)  With the characteristics of the sliding window mechanism of the CNN, it’s possible to capture integrated spatial features by simply altering the movement direction of the kernel, given the relative nature of the image tensor arrangement and the movement of the convolution kernel, transposing the image tensor is adopted as an alternative approach.

## 3 - Usage
To implement and verify these models, you need to specify the dataset, the model name, and the path to the model's weights.

### You can download the RGB Dataset from this link: 
https://www.kaggle.com/datasets/apollo2506/eurosat-dataset?select=EuroSATallBands

### You can download the Multispectral Dataset from this link:
https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset

## 4 - Contact

Mohamed Ahmed Badr, Researcher at Avionics Engineering Department, Military Technical College, Cairo, Egypt, m.badr1086@gmail.com

Ahmed Fathy Elrewainy, Assistant Professor, Avionics Engineering Department, Military Technical College, Cairo, Egypt, ahmed.elrewainy@mtc.edu.eg

Mohamed Abdelmoneim Taha Elshafey, Associate Professor, Head of Computer Engineering and Artificial Intelligence Department, Military Technical College, Cairo, Egypt, m.shafey@mtc.edu.eg ; mohamed.elshafey@ieee.org

## 5 - Reference

```
@article{doi:10.2514/1.I011445,
author = {Badr, Mohamed Ahmed and Elrewainy, Ahmed Fathy and Elshafey, Mohamed Abdelmoneim Taha Elshafey},
title = {Hybrid Spatial–Spectral Autoencoder Models for Lossy Satellite Image Compression},
journal = {Journal of Aerospace Information Systems},
volume = {0},
number = {0},
pages = {1-22},
year = {0},
doi = {10.2514/1.I011445},
URL = {https://doi.org/10.2514/1.I011445}
}
```

```
Badr, MA, Elrewainy, AF, and Elshafey, MAT. "Hybrid Spatial–Spectral Autoencoder Models for Lossy Satellite Image Compression." Journal of Aerospace Information Systems (2024): 1-22.
```


https://arc.aiaa.org/doi/10.2514/1.I011445
