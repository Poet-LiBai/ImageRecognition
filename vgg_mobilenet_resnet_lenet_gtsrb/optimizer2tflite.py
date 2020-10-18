import numpy as np
import tensorflow as tf
from utils.dataset_make import TFDataSlices
from label_transform.gtsrb_csv_to_list_path import gtsrb_list_path

BATCH_SIZE = 128

x_train, y_train, x_valid, y_valid, x_test, y_test = gtsrb_list_path()
train_ds = TFDataSlices(BATCH_SIZE).train_data(x_train, y_train)
valid_ds = TFDataSlices(BATCH_SIZE).test_data(x_valid, y_valid)
test_ds = TFDataSlices(BATCH_SIZE).test_data(x_test, y_test)
# print(int(len(x_test) / BATCH_SIZE))
# random_skip = np.random.randint(int(len(x_test) / BATCH_SIZE)) - 1
# predict_images, true_labels = test_ds.skip(random_skip).as_numpy_iterator().next()
# flags.DEFINE_boolean('convert', False, 'select overwrite to convert')
# flags.DEFINE_string('model_path', "model_h5_save/traffic_sign.h5", 'saved model file path')
# flags.DEFINE_string('tflite_path', "model_h5_save/traffic_sign.tflite", 'tflite output path')
# flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32)')
# flags.DEFINE_string('dataset', "/Volumes/Elements/data/coco_dataset/coco/5k.txt", 'path to dataset')
#
# FLAGS = flags.FLAGS


# def convert():
#     if FLAGS.convert:
#         model = tf.keras.models.load_model(FLAGS.model_path)
#         # model2.summary()
#         converter = tf.lite.TFLiteConverter.from_keras_model(model)
#         tflite_model = converter.convert()
#         with open(FLAGS.tflite_output, 'wb') as file:
#             file.write(tflite_model)


def demo():
    interpreter = tf.lite.Interpreter(model_path="model_h5_save/traffic_sign.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    print('---input_details---\n', input_details)
    output_details = interpreter.get_output_details()
    print('---output_details---\n', output_details)

    input_shape = input_details[0]['shape']

    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

    print('---output_data---\n', output_data)


def evaluate_model():
    interpreter = tf.lite.Interpreter(model_path="model_h5_save/traffic_sign.tflite")
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on ever y image in the "test" dataset.
    prediction_result = []
    labels_list = []
    n = 0
    for predict_images, true_labels in test_ds:
        n += 1
        for i, test_image in enumerate(predict_images):
            if i % 128 == 0:
                print('Evaluated on {m} epochs so far.'.format(m=n))
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_result.append(digit)
            labels_list.append(np.asarray(true_labels)[i])

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_result)
    labels_array = np.array(labels_list)
    # print(prediction_digits)
    # print(labels_list)
    accuracy = np.mean(prediction_digits == labels_array)
    print('TFLite test_accuracy:', accuracy)


# def data_show():
#     print(test_ds)
#     print(predict_images)
#     print(predict_images.shape)
#     print(true_labels.shape)


if __name__ == '__main__':
    # convert()
    # demo()
    # data_show()
    evaluate_model()
