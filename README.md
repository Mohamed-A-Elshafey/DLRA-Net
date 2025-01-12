# DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution
[International Journal of Computer Vision](https://link.springer.com/article/10.1007/s11263-024-02238-w)

[Ahmed R. El-gabri](https://orcid.org/0000-0002-9802-5040), [Hussein A. Aly](https://orcid.org/0000-0001-5604-7252), [Tarek S. Ghoniemy](https://orcid.org/0000-0003-4919-4232) and [Mohamed A. Elshafey](https://orcid.org/0000-0002-1517-8878)

<hr />

## Table of Contents

1. [Introduction](#introduction)
2. [DLRA-Net Model](#DLRA-Net-Model)
3. [Repository Files Description](#Repository-Files-Description)
4. [Usage](#usage)
5. [Reference](#Reference)
   
## Introduction

Hyperspectral Images (HSIs) provide detailed scene insights using extensive spectral bands, crucial for material discrimination and earth observation with substantial costs and low spatial resolution. Recently, Convolutional Neural Networks (CNNs) are common choice for Spectral Super-Resolution (SSR) from Multispectral Images (MSIs). However, they often fail to simultaneously exploit pixel-level noise degradation of MSIs and complex contextual spatial-spectral characteristics of HSIs. In this paper, a Deep Local Residual Attention Network with Contextual Refinement Network (DLRA-Net) is proposed to integrate local low-rank spectral and global contextual priors for improved SSR. Specifically, SSR is unfolded into Contextual-attention Refinement Module (CRM) and Dual Local Residual Attention Module (DLRAM). CRM is proposed to adaptively learn complex contextual priors to guide the convolution layer weights for improved spatial restorations. While DLRAM captures deep refined texture details to enhance contextual priors representations for recovering HSIs. Moreover, lateral fusion strategy is designed to integrate the obtained priors among DLRAMs for faster network convergence. Experimental results on natural-scene datasets with practical noise patterns confirm exceptional DLRA-Net performance with relatively small model size. DLRA-Net demonstrates Maximum Relative Improvements (MRI) between 9.71 and 58.58% in Mean Relative Absolute Error (MRAE) with reduced parameters between 52.18 and 85.85%. Besides, a practical RS-HSI dataset is generated for evaluations showing MRI between 8.64 and 50.56% in MRAE. Furthermore, experiments with HSI classifiers indicate improved performance of reconstructed RS-HSIs compared to RS-MSIs, with MRI in Overall Accuracy (OA) between 7.10 and 15.27%. Lastly, a detailed ablation study assesses model complexity and runtime.



## DLRA-Net Model

![DLRANet](https://github.com/user-attachments/assets/130c4a30-8de2-4369-a089-3125ddf246eb)

Architecture of DLRA-Net integrating local spectral correlations and global contextual priors for improved SSR. CRM learns complex contextual priors to remove the undesired artifacts. Then, DRAMs progressively build up the HSI by capturing local and global contexts. Rm-1 and Rm represents lateral fusion connections between DLRAMs while red downarrow and red uparrow indicate downsampling and upsampling, respectively



![crm](https://github.com/user-attachments/assets/c8522dc9-78aa-4761-9766-8b0543fcc33f)



The architecture CAM. CAM utilizes an average pooling across spatial dimensions H x W then introduced to a shared network followed by a Sigmoid layer to learn channel attention map.



![mc](https://github.com/user-attachments/assets/65f3c965-a2a7-4464-a8e4-06e1357234c0)



The architecture of MC module with dilated convolution layers with d dilation rates are adopted to capture contextual information at different scales

![rimp](https://github.com/user-attachments/assets/03191689-e765-4a5c-b1dc-74b6d92c79b8)

Relative improvements of the proposed model

## Repository Files Description
```
DLRA-Net/code/
├── Training      Code for Training Stage                             
└── Testing       Code for Testing Stage                     
```

## Usage
DLRA-Net is assessed in the NTIRE2022 “Spectral Recovery” track where RGB images were recovered using known CSS. Every RGB image was independently normalized by its maximum value, contaminated with a more realistic unknown noise model, and compressed. This results in severely damaged images where the same object may have different spectrum representations across all the scenes. It comprises 900 training and 50 validation data pairs.

You can download the NTIRE2022 Dataset from this link: 
https://vcg.seas.harvard.edu/publications/ntire-2022


## Reference

To cite this paper
```
El-gabri, A.R., Aly, H.A., Ghoniemy, T.S. et al. DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution. Int J Comput Vis (2024). https://doi.org/10.1007/s11263-024-02238-w
```

https://link.springer.com/article/10.1007/s11263-024-02238-w

