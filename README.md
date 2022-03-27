# PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI2020-Oral)

> **Authors:** 
> [Deng-Ping Fan](https://dpfan.net/), 
> [Ge-Peng Ji](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
> [Tao Zhou](https://taozh2017.github.io/),
> [Geng Chen](https://www.researchgate.net/profile/Geng_Chen13), 
> [Huazhu Fu](http://hzfu.github.io/), 
> [Jianbing Shen](http://iitlab.bit.edu.cn/mcislab/~shenjianbing), and 
> [Ling Shao](http://www.inceptioniai.org/).

## 1. Preface

- This repository provides code for "_**PraNet: Parallel Reverse Attention Network for Polyp Segmentation**_" MICCAI-2020. 
([paper](https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_26) | [中文版](http://dpfan.net/wp-content/uploads/MICCAI20_PraNet_Chinese.pdf))

- If you have any questions about our paper, feel free to contact me. And if you are using PraNet 
or evaluation toolbox for your research, please cite this paper ([BibTeX](#4-citation)).


### 1.1. :fire: NEWS :fire:

- [2022/03/27] :boom: Our new task about Video Polyp Segmentation (VPS) has been released. [ProjectLink](https://github.com/GewelsJI/VPS)/ [PDF]().

- [2021/12/26] :boom: PraNet模型在[Jittor Developer Conference 2021](https://cg.cs.tsinghua.edu.cn/jittor/news/2021-12-27-15-27-00-00-jdc1/)中荣获「最具影响力计图论文（应用）奖」

- [2021/09/07] The Jittor convertion of PraNet ([inference code](https://github.com/DengPingFan/PraNet/tree/master/jittor)) is available right now. It has robust inference efficiency compared to PyTorch version, please enjoy it. Many thanks to Yu-Cheng Chou for the excellent conversion from pytorch framework.

- [2021/09/05] The Tensorflow (Keras) implementation of PraNet (ResNet50/MobileNetV2 version) is released in [github-link](https://github.com/Thehunk1206/PRANet-Polyps-Segmentation). Thanks Tauhid Khan.

- [2021/08/18] Improved version (PraNet-V2) has been released: https://github.com/DengPingFan/Polyp-PVT.

- [2021/04/23] We update the results on four [Camouflaged Object Detection (COD)](https://github.com/DengPingFan/SINet) testing dataset (i.e., COD10K, NC4K, CAMO, and CHAMELEON) of our PraNet, which is the retained on COD dataset from scratch. Download links at google drive are avaliable here: [result](https://drive.google.com/file/d/1h1sXnZA3uIeRXe9eUsH8Vp9i40VylauB/view?usp=sharing), [model weight](https://drive.google.com/file/d/1epdeolFS_JC8D8Pm_r0TaUJM-Qo4v49c/view?usp=sharing), [evaluation results](https://drive.google.com/file/d/1hY_S0-o5rezsBZCUegpDtAAmhy8jpW5N/view?usp=sharing).

- [2021/01/21] :boom: Our PraNet has been used as the base segmentation model of [Prof. Michael I. Jordan](https://scholar.google.com/citations?user=yxUduqMAAAAJ&hl=zh-CN) et al's recent work (Distribution-Free, Risk-Controlling Prediction Sets, [Journal of the ACM 2021](https://arxiv.org/pdf/2101.02703.pdf)).

- [2021/01/10] :boom: Our PraNet achieved the Top-1 ranking on the camouflaged object detection task ([link](http://dpfan.net/camouflage)). 

- [2020/09/18] Upload the pre-computed maps.

- [2020/05/28] Upload pre-trained weights.

- [2020/06/24] Release training/testing code.

- [2020/03/24] Create repository.


### 1.2. Table of Contents

- [PraNet: Parallel Reverse Attention Network for Polyp Segmentation (MICCAI 2020)](#pranet--parallel-reverse-attention-network-for-polyp-segmentation--miccai-2020-)
  * [1. Preface](#1-preface)
    + [1.1. :fire: NEWS :fire:](#11--fire--news--fire-)
    + [1.2. Table of Contents](#12-table-of-contents)
    + [1.3. State-of-the-art approaches](#13-SOTAs)
  * [2. Overview](#2-overview)
    + [2.1. Introduction](#21-introduction)
    + [2.2. Framework Overview](#22-framework-overview)
    + [2.3. Qualitative Results](#23-qualitative-results)
  * [3. Proposed Baseline](#3-proposed-baseline)
    + [3.1 Training/Testing](#31-training-testing)
    + [3.2 Evaluating your trained model:](#32-evaluating-your-trained-model-)
    + [3.3 Pre-computed maps:](#33-pre-computed-maps)
  * [4. Citation](#4-citation)
  * [5. TODO LIST](#5-todo-list)
  * [6. FAQ](#6-faq)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

### 1.3. State-of-the-art Approaches  
1. "Selective feature aggregation network with area-boundary constraints for polyp segmentation." IEEE Transactions on Medical Imaging, 2019.
doi: https://link.springer.com/chapter/10.1007/978-3-030-32239-7_34 
2. "PraNet: Parallel Reverse Attention Network for Polyp Segmentation" IEEE Transactions on Medical Imaging, 2020.
doi: https://link.springer.com/chapter/10.1007%2F978-3-030-59725-2_26
3. "Hardnet-mseg: A simple encoder-decoder polyp segmentation neural network that achieves over 0.9 mean dice and 86 fps" arXiv, 2021
doi: https://arxiv.org/pdf/2101.07172.pdf
4. "TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation" arXiv, 2021.
doi: https://arxiv.org/pdf/2102.08005.pdf
5. continue updating...


## 2. Overview

### 2.1. Introduction

Colonoscopy is an effective technique for detecting colorectal polyps, which are highly related to colorectal cancer. 
In clinical practice, segmenting polyps from colonoscopy images is of great importance since it provides valuable 
information for diagnosis and surgery. However, accurate polyp segmentation is a challenging task, for two major reasons:
(i) the same type of polyps has a diversity of size, color and texture; and
(ii) the boundary between a polyp and its surrounding mucosa is not sharp. 

To address these challenges, we propose a parallel reverse attention network (PraNet) for accurate polyp segmentation in colonoscopy
images. Specifically, we first aggregate the features in high-level layers using a parallel partial decoder (PPD). 
Based on the combined feature, we then generate a global map as the initial guidance area for the following components. 
In addition, we mine the boundary cues using a reverse attention (RA) module, which is able to establish the relationship between
areas and boundary cues. Thanks to the recurrent cooperation mechanism between areas and boundaries, 
our PraNet is capable of calibrating any misaligned predictions, improving the segmentation accuracy. 

Quantitative and qualitative evaluations on five challenging datasets across six
metrics show that our PraNet improves the segmentation accuracy significantly, and presents a number of advantages in terms of generalizability,
and real-time segmentation efficiency (∼50fps).

### 2.2. Framework Overview

<p align="center">
    <img src="imgs/framework-final-min.png"/> <br />
    <em> 
    Figure 1: Overview of the proposed PraNet, which consists of three reverse attention 
    modules with a parallel partial decoder connection. See § 2 in the paper for details.
    </em>
</p>

### 2.3. Qualitative Results

<p align="center">
    <img src="imgs/qualitative_results.png"/> <br />
    <em> 
    Figure 2: Qualitative Results.
    </em>
</p>

## 3. Proposed Baseline

### 3.1. Training/Testing

The training and testing experiments are conducted using [PyTorch](https://github.com/pytorch/pytorch) with 
a single GeForce RTX TITAN GPU of 24 GB Memory.

> Note that our model also supports low memory GPU, which means you can lower the batch size


1. Configuring your environment (Prerequisites):
   
    Note that PraNet is only tested on Ubuntu OS with the following environments. 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n PraNet python=3.6`.
    
    + Installing necessary packages: PyTorch 1.1

1. Downloading necessary data:

    + downloading testing dataset and move it into `./data/TestDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing).
    
    + downloading training dataset and move it into `./data/TrainDataset/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing).
    
    + downloading pretrained weights and move it into `snapshots/PraNet_Res2Net/PraNet-19.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1pUE99SUQHTLxS9rabLGe_XTDwfS6wXEw/view?usp=sharing).
    
    + downloading Res2Net weights [download link (Google Drive)](https://drive.google.com/file/d/1_1N-cx1UpRQo7Ybsjno1PAg4KE1T9e5J/view?usp=sharing).
   
1. Training Configuration:

    + Assigning your costumed path, like `--train_save` and `--train_path` in `MyTrain.py`.
    
    + Just enjoy it!

1. Testing Configuration:

    + After you download all the pre-trained model and testing dataset, just run `MyTest.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).
    
    + Just enjoy it!

### 3.2 Evaluating your trained model:

Matlab: One-key evaluation is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)), 
please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.
The complete evaluation toolbox (including data, map, eval code, and res): [new link](https://drive.google.com/file/d/1bnlz7nfJ9hhYsMLFSBr9smcI7k7p0pVy/view?usp=sharing). 

Python: Please refer to the work of ACMMM2021 https://github.com/plemeri/UACANet

### 3.3 Pre-computed maps: 
They can be found in [download link](https://drive.google.com/file/d/1tW0OOxPSuhfSbMijaMPwRDPElW1qQywz/view?usp=sharing).


## 4. Citation

Please cite our paper if you find the work useful: 

    @article{fan2020pra,
    title={PraNet: Parallel Reverse Attention Network for Polyp Segmentation},
    author={Fan, Deng-Ping and Ji, Ge-Peng and Zhou, Tao and Chen, Geng and Fu, Huazhu and Shen, Jianbing and Shao, Ling},
    journal={MICCAI},
    year={2020}
    }

## 5. TODO LIST

> If you want to improve the usability or any piece of advice, please feel free to contact me directly ([E-mail](gepengai.ji@gmail.com)).

- [ ] Support `NVIDIA APEX` training.

- [ ] Support different backbones (
VGGNet, 
ResNet, 
[ResNeXt](https://github.com/facebookresearch/ResNeXt),
[iResNet](https://github.com/iduta/iresnet), 
and 
[ResNeSt](https://github.com/zhanghang1989/ResNeSt) 
etc.)

- [ ] Support distributed training.

- [ ] Support lightweight architecture and real-time inference, like MobileNet, SqueezeNet.

- [ ] Add more comprehensive competitors.

## 6. FAQ

1. If the image cannot be loaded in the page (mostly in the domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)

## 7. License

The source code is free for research and education use only. Any comercial use should get formal permission first.

---

**[⬆ back to top](#0-preface)**
