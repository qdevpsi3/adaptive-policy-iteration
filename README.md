<h1 align="center" style="margin-top: 0px;"> <b>Adaptive Approximate Policy Iteration</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=arXiv:2002.03069&color=b31b1b)](https://arxiv.org/abs/2002.03069)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=JAX&color=27A59A)](https://github.com/google/jax)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/adaptive-policy-iteration/blob/main/notebooks/deepsea.ipynb)
</div>

## **Description**
This repository contains an <ins>unofficial</ins> implementation of the <ins>Adaptive Approximate Policy Iteration</ins> and its application to the <ins>DeepSea</ins> environment as in :

- Paper : **Adaptive Approximate Policy Iteration**
- Authors : **B. Hao, N. Lazic, Y. Abbasi-Yadkori, P. Joulani, C. Szepesvari**
- Date : **2021**

## **Details**
- Environment : **DeepSea environment** *(Paper, Page 7)* using `bsuite`
- Features : **One-hot encoding** *(Paper, Page 7)*
- Evaluation method : **least-squares Monte Carlo** *(Paper, Page 7)* using `JAX`
- Agent : **AAPI** *(Paper, Algorithm 1)* using `JAX`
## **Usage**
To run the experiments :

- Option 1 : Open in [Colab](https://colab.research.google.com/github/qdevpsi3/adaptive-policy-iteration/blob/main/notebooks/deepsea.ipynb). 
- Option 2 : Run on local machine. First, you need to clone this repository and execute the following commands to install the required packages :
```
$ cd adaptive-policy-iteration
$ pip install -r requirements.txt
```
You can run an experiment using the following command :
```
$ cd src
$ python deepsea.py
```

<p align="center">
<img src="./notebooks/deepsea_results.png" width="500">
</p>