# DETIRE
Version 1.1

Authors: Yan Miao, Fu Liu, Yun Liu

Maintainer: Yan Miao miaoyan17@mails.jlu.edu.cn

# Description
DETIRE (a hybrid Deep lEarning model for idenTifying vIral sequences fRom mEtagenomes) is a deep learning based virus identification method that could detect viruses directly from metagenomes. DETIRE is a two-stage architecture, containing a graph convolutional network (GCN) based sequence embedder and a two-path deep learning model. First, every sequence is cut into several 3-mer fragments, which are then successively input to the GCN-based sequence embedder to train the representations of all 3-mer fragments. After that, these embedded fragments are then fed into the CNN-path and BiLSTM-path to learn their features, respectively. Finally, by two dense layers and a softmax layer, a pair scores are generated, and the higher score determines which type the input sequence is. 

# Dependences
To utilize RNN-VirSeeker on Windows, Python packages "sklearn", "numpy" and "matplotlib" are needed to be previously installed.

In convenience, download Anaconda from https://repo.anaconda.com/archive/, which contains most of needed packages.

To insatll Keras, start "cmd.exe" and enter <br>
```
pip install keras
```
Our codes were all edited by Python 3.6.5 with Keras 1.2.0.
