# EmerGNN
The code of EmerGNN, which is designed for predicting interactions between emerging and existing drugs.

## Installation
This section describes the installation process based on PyTorch. Firstly, you need to install PyTorch and CUDA. After that, you can proceed to install the necessary requirements. The running environment is a Linux server with Ubuntu and an NVIDIA GeForce RTX 3090 GPU. The CUDA version is 11.3.1. We provide the exact versions of packages we use to run this code as follows. Note that other versions may also work. Thanks for their contribution!
```
# Create conda environment
conda create -n emergnn python==3.8
# Install pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
# Install nvcc
conda install https://anaconda.org/conda-forge/cudatoolkit-dev/11.3.1/download/linux-64/cudatoolkit-dev-11.3.1-py38h497a2fe_0.tar.bz2
# Install other necessary requirements 
pip install -r requirements.txt
```



## Running scripts

First `cd` into the corresponding directory, i.e., DrugBank or TWOSIDES. 

Our data is uploaded on the [one drive link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/yzhangee_connect_ust_hk/Elpl9vADdy9Hn9i-IUiruvQB3vNaKjITos5KSXr76coWOg?e=YOBUZf). Download `data-DB.zip` into the DrugBank file folder or `data-TS.zip` for TWOSIDES file folder. Then, unzip the compressed data to get a folder with name `data` for use.



Once the data is ready, you can run the following scripts to reproduce the results:

```
python -W ignore evaluate.py --dataset=S1_1 --n_epoch=40 --epoch_per_test=2 --gpu=0
```

The three settings, i.e. (S0) interaction between exsiting drugs, (S1) interaction between emerging drug and existing drug, and (S2) interaction between emerging drugs, are provided. The datasets for S1 and S2 settings are created by ourself, while the S0 setting uses the original dataset from SumGNN (Yu et. al. 2021).



You can also try the following script for hyper-parameter tuning, which may lead to even better performance.

```
python -W ignore tune_hyperms.py --dataset=S1_1 --n_epoch=20 --epoch_per_test=2 --gpu=0
```



In our environment, results of both datasets in one round can be obtained within 1 day. Running on TWOSIDES can be faster than DrugBank since TWOSIDES is smaller.

