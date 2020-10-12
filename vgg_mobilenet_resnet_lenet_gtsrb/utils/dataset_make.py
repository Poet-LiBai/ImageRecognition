import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def keras_image_generator(x_trains, y_trains, x_tests, y_tests, batch_size):
    image_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=45, width_shift_range=.15, height_shift_range=.15,
                                   horizontal_flip=True, zoom_range=0.5, validation_split=0.1)
    train_ds = image_gen.flow(x_trains, y_trains, batch_size=batch_size, shuffle=True)
    test_ds = ImageDataGenerator(rescale=1. / 255).flow(x_tests, y_tests)
    data, labels = next(train_ds)
    print(data[0].shape, data[0].dtype)
    plt.imshow(data[1])
    plt.show()
    return train_ds, test_ds


def io_read_decode(filepath, labels):
    images = tf.io.read_file(filepath)
    images = tf.io.decode_jpeg(images)
    return images, labels


def image_resize(images, labels, target_size=32):
    # images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize(images, (target_size, target_size), method='lanczos5') / 255.0
    return images, labels


def image_resize_aug(images, label, target_size=32):
    # images = tf.image.per_image_standardization(images)
    # images = tf.image.rgb_to_grayscale(images)
    images = tf.image.resize(images, (target_size, target_size), method='lanczos5') / 255.0
    images = tf.image.random_brightness(images, 0.1)
    # images = tf.image.random_contrast(images, 0.1, 0.3)
    # images = tf.image.resize_with_crop_or_pad(images, target_size + 6, target_size + 6)
    # Random crop back to the original size
    # images = tf.image.random_crop(images, size=[target_size, target_size, 3])
    # images = tf.image.random_hue(images, 0.2)
    # images = tf.image.random_saturation(images, 0.1, 0.3)
    # images = tf.image.random_jpeg_quality(images, 85, 100)
    return images, label


class TFDataSlices:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def train_data(self, x_trains, y_trains):
        train_ds = tf.data.Dataset.from_tensor_slices((x_trains, y_trains)) \
            .map(io_read_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .cache().map(image_resize_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .shuffle(buffer_size=len(x_trains)) \
            .repeat().batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds

    def test_data(self, x_tests, y_tests):
        test_ds = tf.data.Dataset.from_tensor_slices((x_tests, y_tests)) \
            .map(io_read_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache() \
            .map(image_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return test_ds

    def test_data_no_io(self, x_tests, y_tests):
        test_ds = tf.data.Dataset.from_tensor_slices((x_tests, y_tests)) \
            .map(image_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache() \
            .shuffle(buffer_size=len(y_tests)).batch(self.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return test_ds
