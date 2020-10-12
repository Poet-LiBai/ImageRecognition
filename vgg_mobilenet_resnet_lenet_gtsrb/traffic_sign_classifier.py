import os
import time

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from label_transform.gtsrb_csv_to_list_path import gtsrb_list_path
from net.lenet5_body import lenet_model
# from net.mobilenet_body import mobilenet_model
from utils.callbacks import train_callback_list
from utils.dataset_make import TFDataSlices
from utils.result_visual import signname_csv, predict_visual, plot_curve

# from tensorflow.keras.activations import

BATCH_SIZE = 128

x_train, y_train, x_valid, y_valid, x_test, y_test = gtsrb_list_path()
train_ds = TFDataSlices(BATCH_SIZE).train_data(x_train, y_train)
valid_ds = TFDataSlices(BATCH_SIZE).test_data(x_valid, y_valid)
test_ds = TFDataSlices(BATCH_SIZE).test_data(x_test, y_test)

# data_image_visual(valid_ds)
# timeit(test_ds)
# print(train_ds.as_numpy_iterator())
# METRICS = [
#     tf.keras.metrics.TruePositives(name='tp'),
#     tf.keras.metrics.FalsePositives(name='fp'),
#     tf.keras.metrics.TrueNegatives(name='tn'),
#     tf.keras.metrics.FalseNegatives(name='fn'),
#     tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#     tf.keras.metrics.Precision(name='precision'),
#     tf.keras.metrics.Recall(name='recall'),
#     tf.keras.metrics.AUC(name='auc')]


def train_state():
    model = lenet_model(43)
    model.summary()
    # model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    # callback_list = train_callback_list()
    # train_start = time.time()
    # train_history = model.fit(train_ds, epochs=50, verbose=1, steps_per_epoch=len(y_train) // BATCH_SIZE,
    #                           validation_data=valid_ds, callbacks=callback_list, initial_epoch=0)
    # train_end = time.time()
    # train_time = train_end - train_start
    # eval_start = time.time()
    # eval_history = model.evaluate(test_ds)
    # eval_end = time.time()
    # # eval_callback = eval_callbacks_list()
    # # plot_curve(eval_history)
    # eval_time = eval_end - eval_start
    # print('train total time: {:03f} s train per epoch time: {:03f} s'.format(train_time, train_time / 50))
    # print('evaluate time: {:03f} s'.format(eval_time),
    #       'eval_loss: {:03f} eval_accuracy: {:03f}'.format(eval_history[0], eval_history[1]))
    # plot_curve(train_history)


def predict_eval_state():
    model_path = os.path.abspath(
        'F:\\AI-modelsaver\\GTSRB\\LeNet5\\2020-08-14\\07-50-57\\ModelCheckPoint\\'
        'ep050-loss0.048-val_loss0.049-acc0.987-val_acc0.992.ckpt')
    # load_new_ckpt = input('是否重新替换新模型的路径：yes or no ?   ')
    # if str(load_new_ckpt) == 'yes' or 'y' or 'Y' or 'YES' or 'Yes':
    #     pass
    # else:
    #     print('请在源码中替换路径')

    model = tf.keras.models.load_model(model_path)
    model.summary()

    eval_start = time.time()
    eval_history = model.evaluate(test_ds)
    eval_end = time.time()
    # eval_callback = eval_callbacks_list()
    # plot_curve(eval_history)
    eval_time = eval_end - eval_start
    predict_images, true_labels = next(iter(test_ds))
    # predict_outputs = model(training=False).predict(predict_images, batch_size=BATCH_SIZE)
    predict_start = time.time()
    # predict_outputs = model(tf.expand_dims(predict_images[0], axis=0))
    predict_outputs = model.predict(predict_images)
    predict_end = time.time()
    predict_time = predict_end - predict_start
    class_name_list = signname_csv()
    eval_record = 'eval loss:{:.04f} eval accuracy:{:.04f}%'.format(eval_history[0], eval_history[1] * 100) + '\n' \
                  + 'evaluate time:{:.04f}s '.format(eval_time)
    predict_record = 'predict time per batch:{:.04f}s'.format(predict_time)
    # print(eval_record)
    print(predict_record)
    predict_visual(predict_images, predict_outputs, true_labels, class_name_list, eval_record, predict_record)


# train_or_predict = input('please select train or predict ?')
# if str(train_or_predict) == 'train' or 'T':
#     train_state()
# else:
predict_eval_state()
