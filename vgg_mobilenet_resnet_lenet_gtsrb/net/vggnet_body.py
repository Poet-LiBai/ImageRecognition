from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout

vgg_cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


# 以下各个模块的定义函数返回的都是层列表list,因为Sequential([])传入的参数是列表形式


# 定义conv_bn_act模块，可易于调整是否使用bn层和更改激活函数
def conv_bn_act(x, filters):
    x = Conv2D(filters, (3, 3), 1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x


def vgg_state(x, vgg_cfg_select):
    for filters in vgg_cfg_select:
        if filters != 'M':
            x = conv_bn_act(x, filters)
        else:
            x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    return x


def vgg_net(vgg_cfg_select, class_num):
    inputs = Input(shape=(32, 32, 3))
    x = vgg_state(inputs, vgg_cfg_select)
    x = Dense(4096)(x)
    x = Dropout(0.5)(x)
    x = Dense(4096)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(class_num)(x)

    return Model(inputs, outputs)


vgg_model = vgg_net(vgg_cfg['11'], class_num=80)

vgg_model.summary()
# vgg_model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
# history = vgg_model.fit(train_data, validation_data=test_data, epochs=50, callbacks=callbacks)
# history_eval = vgg_model.evaluate(test_data)
