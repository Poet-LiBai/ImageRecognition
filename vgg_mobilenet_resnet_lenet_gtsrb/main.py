import tensorflow as tf
from tensorflow.keras import Model,Sequential,Input
from tensorflow.keras.layers import Conv2D,MaxPool2D,GlobalAveragePooling2D,Flatten,Dense,ZeroPadding2D,DepthwiseConv2D
from tensorflow.keras.layers import Activation,BatchNormalization,Dropout,Softmax,Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import datetime,os,time
import h5py
import argsparse

mobilenet_cfg = [['s1',64],['s2',128],['s1',128],['s2',256],['s1',256],['s2',512],['s1',512],['s1',512],['s1',512],['s1',512],['s1',512],
                 #['s2',1024],['s2',1024]
                 ]
#类别数
num_classes = 10

##***实验对比条件***##
#是否使用预训练模型
pretrained = False
#是否使用数据增强
data_augmentation = True
#权重参数初始化
#kernel_init = 
#权重正则化
kernel_regularizer = None
#是否通过fc_cfg的'D'，来表示使用dropout ,如[256,'D',128,'D']
fc_cfg = [256,128]
#batch_size
batch_size = 128
#输入大小
input_shape_select = (32,32,3)
#激活函数
activation = 'relu'
#BN层使用
batch_normalization = True
#优化算法
optimizer = 'adam'
#损失函数
loss_function = 'sparse_categorical_crossentropy'
#评估指标
metrics = 'accuracy'


# weight_decay = 5e-4
# learning_rate = 1e-2
# dropout_rate = 0.5
# epoch_num = 200

