import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, GlobalAvgPool2D, Activation, LeakyReLU


class LeNet5(Model):

    def __init__(self, activation):
        super(LeNet5, self).__init__(name='LeNet5')

        self.conv2d_1 = Conv2D(32, (3, 3), 1, activation=activation)
        self.MaxPool2d_1 = MaxPool2D((2, 2), 2)
        self.conv2d_2 = Conv2D(64, (3, 3), 1, activation=activation)
        self.MaxPool2d_2 = MaxPool2D((2, 2), 2)
        self.conv2d_3 = Conv2D(128, (3, 3), 1, activation=activation)
        self.global_avg = GlobalAvgPool2D()
        self.dense_4 = Dense(84, activation=activation)
        self.dense_5 = Dense(43, activation='softmax')

    def call(self, x, **kwargs):
        x = self.conv2d_1(x)
        x = self.MaxPool2d_1(x)
        x = self.conv2d_2(x)
        x = self.MaxPool2d_2(x)
        x = self.conv2d_3(x)
        x = self.global_avg(x)
        x = self.dense_4(x)
        x = self.dense_5(x)

        return x


def lenet_model(class_num, act='relu', **kwargs):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), **kwargs))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation(act))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), **kwargs))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation(act))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (3, 3), **kwargs))
    # model.add(tf.keras.layers.BatchNormalization())
    model.add(Activation(act))
    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(84, **kwargs))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))

    return model
