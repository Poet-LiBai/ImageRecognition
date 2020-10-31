import argparse
import zipfile
import time
import progressbar
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from label_transform.gtsrb_csv_to_list_path import gtsrb_list_path
from utils.dataset_make import TFDataSlices

BATCH_SIZE = 128

# gtsrtb_list_path 返回的是images,labels的路径列表
x_train, y_train, x_valid, y_valid, x_test, y_test = gtsrb_list_path()

# TFDataSlices 传入路径列表返回tf.Data.Dataset的可迭代对象；而且train_ds经过map进行数据增强
train_ds = TFDataSlices(BATCH_SIZE).train_data(x_train, y_train)
valid_ds = TFDataSlices(BATCH_SIZE).test_data(x_valid, y_valid)
test_ds = TFDataSlices(BATCH_SIZE).test_data(x_test, y_test)

# 命令行参数传入
parse = argparse.ArgumentParser()
parse.add_argument('--model_path', type=str, default='model_h5_save/traffic_sign.h5', help='the train model path')
parse.add_argument('--tflite_path', type=str, default='model_h5_save/traffic_sign.tflite',
                   help='the tflite path to load')
parse.add_argument('--convert', action='store_true', help='Choose whether to convert h5 to tflite')
parse.add_argument('--optimize', action='store_true', help='Choose whether to optimize in TFLiteConverter')
parse.add_argument('--prune', action='store_true', help='Choose whether to pruning the model')
parse.add_argument('--zip', action='store_true', help='Choose whether to compress with zip algorithm')
parse.add_argument('--eval_tflite', action='store_true', help='Choose whether to evalute the tflite model')
cfg_parse = parse.parse_args()


def convert():
    model = tf.keras.models.load_model(cfg_parse.model_path)

    # 转换成tflite时选择是否需要剪枝
    if cfg_parse.prune:
        model = prune_model(model)
        model.summary()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        cfg_parse.tflite_path = cfg_parse.tflite_path.rstrip('.tflite') + '_pruned.tflite'  # 修改保存tflite的文件名
    else:
        model.summary()
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 选择是否需要优化，tf.lite.Optimize.DEFAULT 默认对计算图graph和大小size优化
    if cfg_parse.optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        cfg_parse.tflite_path = cfg_parse.tflite_path.rstrip('.tflite') + '_optimized.tflite'

    tflite_model = converter.convert()

    # 选择是否需要zip压缩，一般经过prune的模型都比较稀疏，压缩可以大大减小存储大小；需要用时可以在upzip加载就行
    if cfg_parse.zip:
        cfg_parse.tflite_path = cfg_parse.tflite_path.rstrip('.tflite') + 'zipped.zip'
        with zipfile.ZipFile(cfg_parse.tflite_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(tflite_model)

    with open(cfg_parse.tflite_path, 'wb') as file:
        file.write(tflite_model)


def prune_model(model):
    pruning_epochs = 10
    end_step = np.ceil(len(x_train) / BATCH_SIZE).astype(np.int32) * pruning_epochs
    prune_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                             final_sparsity=0.80,
                                                                             begin_step=0,
                                                                             end_step=end_step)}
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **prune_params)     # 注意**双星号
    model_for_pruning.compile(optimizer='adam',
                              loss=SparseCategoricalCrossentropy(from_logits=True),
                              metrics=['accuracy'])
    model_for_pruning.summary()
    pruning_callbacks = [tfmot.sparsity.keras.UpdatePruningStep(),
                         tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_summary')
                         ]
    model_for_pruning.fit(train_ds, validation_data=valid_ds,
                          epochs=pruning_epochs, callbacks=pruning_callbacks,
                          steps_per_epoch=len(y_train) // BATCH_SIZE)       # 需要添加steps_per_epoch
    pruning_loss, pruning_accuracy = model_for_pruning.evaluate(test_ds)
    model_strip_prune = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    # tf.keras.models.save_model(model_strip_prune, 'model_h5_save/traffic_sign_pruned.h5', include_optimizer=False)
    print('Pruning Accuracy is : ', pruning_accuracy)
    return model_strip_prune


def demo():
    interpreter = tf.lite.Interpreter(cfg_parse.tflite_path)
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
    interpreter = tf.lite.Interpreter(cfg_parse.tflite_path)
    interpreter.allocate_tensors()  # 在内存预分配张量tensor空间

    # 获取输入input,输出output的节点索引，可以使用demo()来看interpreter.get_input_details()返回的是什么
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on ever y image in the "test" dataset.
    prediction_result = []
    labels_list = []
    n = 0
    start_time = time.time()

    # 显示进度条
    widgets = ['Evaluate: ', progressbar.Percentage(), '  ', progressbar.Bar('='), '  ',
               # progressbar.Timer(), '  ',
               progressbar.ETA(), '  ',
               # progressbar.FileTransferSpeed()
               ]

    bar = progressbar.ProgressBar(widgets=widgets, maxval=99).start()

    # test_ds是可迭代对象，它的形状应为(分成多少个batch, 一个batch有多大, 图像高度, 图像宽度, 图像通道数)即(n, batch_size, h, w, c)
    # 第一个for是在test_ds的 多少个batch 维度循环，则predict_images 的形状为(batch_size, h, w, c),是一批图像
    # 第二个for是在 predict_images的 batch_size 维度循环，是一张张图像test_image
    for predict_images, true_labels in test_ds:
        n += 1
        for i, test_image in enumerate(predict_images):
            # Pre-processing: add batch dimension and convert to float32 to match with
            # the model's input data format.
            # 由于模型model的输入原本有batch_size维度，所以这里需要扩一维
            test_image = np.expand_dims(test_image, axis=0).astype(np.float32)

            # set_tensor相当于将图像的值设置allocate_tensors分配给input_tensor的内存上
            interpreter.set_tensor(input_index, test_image)

            # Run inference.
            # invoke 调用的意思
            interpreter.invoke()

            # Post-processing: remove batch dimension and find the digit with highest
            # probability.
            # 用索引获取interpreter输出output张量的值
            output = interpreter.tensor(output_index)
            digit = np.argmax(output()[0])
            prediction_result.append(digit)
            labels_list.append(np.asarray(true_labels)[i])
        bar.update(n)
    bar.finish()
    end_time = time.time()
    total_time = end_time - start_time

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_result)
    labels_array = np.array(labels_list)
    accuracy = np.mean(prediction_digits == labels_array)
    print('TFLite test_accuracy:', accuracy)
    print('The evaluate time is %.2f s' % total_time)


if __name__ == '__main__':
    if cfg_parse.convert:
        convert()
    demo()
    if cfg_parse.eval_tflite:
        evaluate_model()
