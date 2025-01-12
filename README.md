# DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution
[International Journal of Computer Vision](https://link.springer.com/article/10.1007/s11263-024-02238-w)

[Ahmed R. El-gabri](https://orcid.org/0000-0002-9802-5040), [Hussein A. Aly](https://orcid.org/0000-0001-5604-7252), [Tarek S. Ghoniemy](https://orcid.org/0000-0003-4919-4232) and [Mohamed A. Elshafey](https://orcid.org/0000-0002-1517-8878)

<hr />

## Table of Contents

1. [Introduction](#introduction)
2. [DLRA-Net Network Architecture](#DLRA-Net-Network-Architecture)
3. [Data Preparation](#Data-Preparation)
4. [Citation](#Citation)
   
## Introduction

Hyperspectral Images (HSIs) provide detailed scene insights using extensive spectral bands, crucial for material discrimination and earth observation with substantial costs and low spatial resolution. Recently, Convolutional Neural Networks (CNNs) are common choice for Spectral Super-Resolution (SSR) from Multispectral Images (MSIs). However, they often fail to simultaneously exploit pixel-level noise degradation of MSIs and complex contextual spatial-spectral characteristics of HSIs. In this paper, a Deep Local Residual Attention Network with Contextual Refinement Network (DLRA-Net) is proposed to integrate local low-rank spectral and global contextual priors for improved SSR. Specifically, SSR is unfolded into Contextual-attention Refinement Module (CRM) and Dual Local Residual Attention Module (DLRAM). CRM is proposed to adaptively learn complex contextual priors to guide the convolution layer weights for improved spatial restorations. While DLRAM captures deep refined texture details to enhance contextual priors representations for recovering HSIs. Moreover, lateral fusion strategy is designed to integrate the obtained priors among DLRAMs for faster network convergence. Experimental results on natural-scene datasets with practical noise patterns confirm exceptional DLRA-Net performance with relatively small model size. DLRA-Net demonstrates Maximum Relative Improvements (MRI) between 9.71 and 58.58% in Mean Relative Absolute Error (MRAE) with reduced parameters between 52.18 and 85.85%. Besides, a practical RS-HSI dataset is generated for evaluations showing MRI between 8.64 and 50.56% in MRAE. Furthermore, experiments with HSI classifiers indicate improved performance of reconstructed RS-HSIs compared to RS-MSIs, with MRI in Overall Accuracy (OA) between 7.10 and 15.27%. Lastly, a detailed ablation study assesses model complexity and runtime.



## DLRA-Net Network Architecture

![DLRANet](https://github.com/user-attachments/assets/130c4a30-8de2-4369-a089-3125ddf246eb)

Architecture of DLRA-Net integrating local spectral correlations and global contextual priors for improved SSR. CRM learns complex contextual priors to remove the undesired artifacts. Then, DRAMs progressively build up the HSI by capturing local and global contexts. Rm-1 and Rm represents lateral fusion connections between DLRAMs while red downarrow and red uparrow indicate downsampling and upsampling, respectively



![crm](https://github.com/user-attachments/assets/c8522dc9-78aa-4761-9766-8b0543fcc33f)



The architecture CAM. CAM utilizes an average pooling across spatial dimensions H x W then introduced to a shared network followed by a Sigmoid layer to learn channel attention map.



![mc](https://github.com/user-attachments/assets/65f3c965-a2a7-4464-a8e4-06e1357234c0)



The architecture of MC module with dilated convolution layers with d dilation rates are adopted to capture contextual information at different scales

![rimp](https://github.com/user-attachments/assets/03191689-e765-4a5c-b1dc-74b6d92c79b8)

Relative improvements of the proposed model
                  

## Data Preparation:

- Download the training spectral images ([Google Drive](https://drive.google.com/file/d/1FQBfDd248dCKClR-BpX5V2drSbeyhKcq/view))
- Download the training RGB images ([Google Drive](https://drive.google.com/file/d/1A4GUXhVc5k5d_79gNvokEtVPG290qVkd/view))
- Download  the validation spectral images ([Google Drive](https://drive.google.com/file/d/12QY8LHab3gzljZc3V6UyHgBee48wh9un/view))
- Download the validation RGB images ([Google Drive](https://drive.google.com/file/d/19vBR_8Il1qcaEZsK42aGfvg5lCuvLh1A/view))

Put all downloaded files to `/DLRA-Net-master/Dataset/`, and this repo is collected as the following form:
 ```shell
	|--DLRA-Net-master
		|--figures
		|--test
		|--train  
		|--Dataset 
            |--Train_spectral
				|--ARAD_1K_0001.mat
				|--ARAD_1K_0002.mat
				： 
				|--ARAD_1K_0900.mat
			|--Train_RGB
				|--ARAD_1K_0001.jpg
				|--ARAD_1K_0002.jpg
				： 
				|--ARAD_1K_0900.jpg
			|--Valid_soectral
				|--ARAD_1K_0901.mat
				|--ARAD_1K_0902.mat
				： 
				|--ARAD_1K_0950.mat
			|--Valid_RGB
				|--ARAD_1K_0901.jpg
				|--ARAD_1K_0902.jpg
				： 
				|--ARAD_1K_0950.jpg
```
1. #### Data Preprocess.
```shell
cd /DLRA-Net-master/train/
# Getting the prepared train data by run:
python train_data_preprocess.py --data_path '../Dataset' --patch_size 128 --stride 64 --train_data_path './dataset/Train'

# Getting the prepared valid data by run:
python valid_data_preprocess.py --data_path '../Dataset' --valid_data_path './dataset/Valid'
```
2. #### Training.
```shell
python main.py
```
The data generated during training will be recorded in `/RealWorldResults/`.

3. #### Test.
```shell
cd /DLRA-Net-master/test/
python test.py --RGB_dir '../Dataset/Valid_RGB' --model_dir './model/model.pth' --result_dir './test_results'

# The PSNR, SSIM, SAM, ERGAS, MRAE and RMSE indicators can be obtained by run:
python compute_mrae.py --path_rec './test_results' --path_gt '../Dataset/Valid_spectral'
```
## Citation

If you find this code helpful, please kindly cite:
```shell
# DLRA-Net
@article{el2024dlra,
  title={DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution},
  author={El-gabri, Ahmed R and Aly, Hussein A and Ghoniemy, Tarek S and Elshafey, Mohamed A},
  journal={International Journal of Computer Vision},
  pages={1--33},
  year={2024},
  publisher={Springer}
}
```
El-gabri, A.R., Aly, H.A., Ghoniemy, T.S. et al. DLRA-Net: Deep Local Residual Attention Network with Contextual Refinement for Spectral Super-Resolution. Int J Comput Vis (2024). https://doi.org/10.1007/s11263-024-02238-w
```

https://link.springer.com/article/10.1007/s11263-024-02238-w

