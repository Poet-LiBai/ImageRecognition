resnet_cfg = {
    'resnet18': {'block_num': [2, 2, 2, 2], 'block_channels': [[64, 64], [128, 128], [256, 256], [512, 512]]},
    'resnet34': {'block_num': [3, 4, 6, 3], 'block_channels': [[64, 64], [128, 128], [256, 256], [512, 512]]},
    'resnet50': {'block_num': [3, 4, 6, 3],
                 'block_channels': [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]},
    'resnet101': {'block_num': [3, 4, 23, 3],
                  'block_channels': [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]},
    'resnet152': {'block_num': [3, 8, 36, 3],
                  'block_channels': [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]}
}


def BasicBlock(out_channels, layer_num, strides=1, activation='relu', padding='same', kernel_regularizer=None, bn=True):
    # out_channel [64,64],[128,128]
    block_list = []
    n = 0
    for k_size in [3, 3]:

        # n是block里的第几层conv
        n += 1

        # 使用bn就无需use_bias
        if bn:
            use_bias = False
        else:
            use_bias = True
        if n == 1:
            strides = strides
        else:
            strides = 1

        block_list += [Conv2D(out_channels[n - 1], kernel_size=k_size, strides=strides, padding=padding,
                              kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                              name='basicblock_' + str(layer_num) + '_conv1')]

        # 定义是否添加bn层
        if bn:
            block_list += [BatchNormalization(name='basicblock_' + str(layer_num) + '_bn' + str(n))]
        # basicblock的第2层是在shortcut之后才添加激活函数
        if n != 2:
            block_list += [
                Activation(activation=activation, name='basicblock_' + str(layer_num) + '_activation' + str(n))]

    return block_list


def BottleBlock(out_channels, layer_num, strides=1, activation='relu', padding='same', kernel_regularizer=None,
                bn=True):
    # out_channel [64,64,128]
    block_list = []
    n = 0
    for k_size in [1, 3, 1]:

        # n是单个block里的第几层conv
        n += 1

        # 使用bn就无需use_bias
        if bn:
            use_bias = False
        else:
            use_bias = True

        if n == 1:
            strides = strides
        else:
            strides = 1

        block_list += [Conv2D(out_channels[n - 1], kernel_size=k_size, strides=strides, padding=padding,
                              kernel_regularizer=kernel_regularizer, use_bias=use_bias,
                              name='bottleblock_' + str(layer_num) + '_conv1')]

        if bn:
            block_list += [BatchNormalization(name='bottleblock_' + str(layer_num) + '_bn' + str(n))]
        # bottleblock的第3层是在shortcut之后才添加激活函数
        if n != 3:
            block_list += [
                Activation(activation=activation, name='bottleblock_' + str(layer_num) + '_activation') + str(n)]

    return block_list


# 加上跳跃连接
def skip_connect(x_, block, cfg, layer_num, strides=1, activation='relu', padding='same', kernel_regularizer=None,
                 bn=True):
    # skip connect H(x)=F(x)+x
    skip = block(cfg, layer_num=layer_num, strides=strides, activation=activation, padding=padding,
                 kernel_regularizer=kernel_regularizer, bn=bn)
    input_data = x_
    x_ = Sequential(skip, name='identify_block_' + str(layer_num))(input_data)

    # if x.shape[-1] != input_data.shape[-1]:
    #   input_data = MaxPool2D(pool_size=(2,2),strides=2,padding='valid')(input_data)
    #   zero_pad = tf.zeros_like(input_data)
    #   input_data = tf.concat([input_data,zero_pad],axis=-1)

    # x = add([x,input_data])
    x_ = Activation(activation=activation, name='add_after_activation_' + str(layer_num))(x_)

    return x_


def resnet_state_layer(x, block_type, block_num, block_channels, activation='relu', kernel_regularizer=None, bn=True):
    state_layer_input = x
    block_channels_idx = -1
    layer_num = 0

    for n in block_num:

        # resnet_cfg字典中'block_channels'列表索引
        block_channels_idx += 1

        for m in range(n):
            # layer_num表示整个网络的第几层，除了第1个block,其他block的第1层strides=2
            layer_num += 1
            if layer_num != 1 and m == 0:
                strides = 2

            else:
                strides = 1
            # we
            state_layer_output = skip_connect(state_layer_input, block_type, block_channels[block_channels_idx],
                                              layer_num, strides=strides, activation=activation, padding=padding,
                                              kernel_regularizer=kernel_regularizer, bn=bn)
            state_layer_input = state_layer_output

    return state_layer_output


def resnet_input_layer(x, kernel_size=(7, 7), activation='relu', padding='valid', kernel_regularizer=None, bn=True):
    x = ZeroPadding2D(padding=3, name='zeropadding_1')(x)
    x = Conv2D(filters=64, kernel_size=kernel_size, strides=2, padding=padding, kernel_regularizer=kernel_regularizer,
               name='conv2d_1')(x)
    if bn:
        x = BatchNormalization(name='bn_1')(x)
    x = Activation(activation=activation, name='activation_1')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding='same', name='maxpool_1')(x)

    return x


def resnet_header_layer(x, num_classes=1000):
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    return x


x = Input(shape=(224, 224, 3), name='input')
model_input = resnet_input_layer(x)
model_body = resnet_state_layer(model_input, BasicBlock, resnet_cfg['resnet18']['block_num'],
                                resnet_cfg['resnet18']['block_channels'])
model_output = resnet_header_layer(model_body)

model_2 = Model(x, model_output)
model_2.summary()
