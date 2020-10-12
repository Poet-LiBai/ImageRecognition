from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, ZeroPadding2D, ReLU
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Input, Model

cfg = [['s1', 64], ['s2', 128], ['s1', 128], ['s2', 256], ['s1', 256], ['s2', 512], ['s1', 512], ['s1', 512],
       ['s1', 512], ['s1', 512], ['s1', 512],
       # ['s2',1024],['s2',1024]
       ]


# 第一层的
# PCBA---ZeroPaddingConv2DBatchNormalizationActivation
def ZeroConvBn(x, num_filters):
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), strides=(2, 2), padding='valid', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(ReLU(6))(x)
    return x


# 深度可分离卷积
# DBA---DepthwiseBatchNormalizationActivation
def DepthwiseConvBn(x, **kwargs):
    newkwargs = {'use_bias': False, 'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    newkwargs.update(kwargs)
    x = DepthwiseConv2D(kernel_size=(3, 3), **newkwargs)(x)
    x = BatchNormalization()(x)
    x = Activation(ReLU(6))(x)
    return x


# 点卷积模块
# PBA---PointwiseBatchNormalizationActivation
def PointwiseConvBn(x, num_filters):
    x = Conv2D(num_filters, kernel_size=1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(ReLU(6))(x)
    return x


# 深度可分离卷积和点卷积组合
# Depthwise_Pointwise_Block
def DPBlock(x, out_channels, **kwargs):
    x = DepthwiseConvBn(x, **kwargs)
    x = PointwiseConvBn(x, out_channels)
    return x


# 对于步长strides=2的模块，下采样前要补0才能使输出特征图大小刚好是输入的一半,
# padding = ((top_pad, bottom_pad), (left_pad, right_pad))决定在哪个边缘补0
def ZeroDPBlock(x, out_channels):
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = DPBlock(x, out_channels, strides=(2, 2))
    return x


# 以输出特征图大小相同的层组合成一个模块
def ZeroDPBlock2(x, num_filters):
    x = ZeroPadding2D(padding=((1, 0), (1, 0)))(x)
    x = DPBlock(x, num_filters, strides=(2, 2))
    x = DPBlock(x, num_filters * 2)
    return x


# 连续4个循环（论文的图是5个循环，我把第1个循环放到DPDP的一个模块中了）
def DPBlock4(x, num_filters):
    x = DPBlock(x, num_filters)
    x = DPBlock(x, num_filters)
    x = DPBlock(x, num_filters)
    x = DPBlock(x, num_filters)
    return x


def body(x, class_num):
    x = ZeroConvBn(x, 32)

    x = DPBlock(x, 64)

    x = ZeroDPBlock2(x, 64)
    x = ZeroDPBlock2(x, 128)
    x = ZeroDPBlock2(x, 256)

    # x = DPBlock4(x, 512)

    # x = ZeroDPBlock(x, 1024)
    # x = ZeroDPBlock(x, 1024)

    x = GlobalAveragePooling2D()(x)
    x = Dense(class_num, activation='softmax')(x)
    return x


def mobilenet_model(class_num, input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    outputs = body(inputs, class_num)
    return Model(inputs, outputs)


mobilenet = mobilenet_model(10)
mobilenet.summary()
