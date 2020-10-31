import os
import time
import argparse
import tensorflow as tf
from adabelief_tf import AdaBelief
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from label_transform.gtsrb_csv_to_list_path import gtsrb_list_path
from net.lenet5_body import lenet_model
# from net.mobilenet_body import mobilenet_model
from utils.callbacks import train_callback_list
from utils.dataset_make import TFDataSlices
from utils.result_visual import signname_csv, predict_visual, plot_curve

BATCH_SIZE = 128
Optimizer = AdaBelief()
x_train, y_train, x_valid, y_valid, x_test, y_test = gtsrb_list_path()
train_ds = TFDataSlices(BATCH_SIZE).train_data(x_train, y_train)
valid_ds = TFDataSlices(BATCH_SIZE).test_data(x_valid, y_valid)
test_ds = TFDataSlices(BATCH_SIZE).test_data(x_test, y_test)

parse = argparse.ArgumentParser()
parse.add_argument('--train', action='store_true', help='Choose to train')
parse.add_argument('--predict', action='store_true', help='Choose to predict')
parse.add_argument('--model_save', type=str, default='model_h5_save/traffic_sign.h5', help='model save path')
parse.add_argument('--model_predict', type=str, default='', help='Choose which model to predict')
cfg_parse = parse.parse_args()


def train_state():
    model = lenet_model(43)
    model.summary()

    # from_logits=True时就不用在模型添加sigmoid将输出规范到0~1范围，延后到loss阶段规范0~1
    # 但是在tflite推理阶段其输出是没有规范到0~1的，若Android Demo时需要显示prob的百分比是不是很合适
    # 所以在
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    callback_list = train_callback_list()
    train_start = time.time()
    train_history = model.fit(train_ds, epochs=100, verbose=1, steps_per_epoch=len(y_train) // BATCH_SIZE,
                              validation_data=valid_ds, callbacks=callback_list, initial_epoch=0)
    tf.keras.models.save_model(model, cfg_parse.model_save, include_optimizer=False)
    train_end = time.time()
    train_time = train_end - train_start
    eval_start = time.time()
    # return_dict默认是False的，不然用不了history['loss']或history['accuracy']
    eval_history = model.evaluate(test_ds, return_dict=True)
    eval_end = time.time()
    # eval_callback = eval_callbacks_list()
    plot_curve(eval_history)
    eval_time = eval_end - eval_start
    print('train total time: {:03f} s train per epoch time: {:03f} s'.format(train_time, train_time / 50))
    print('evaluate time: {:03f} s'.format(eval_time),
          'eval_loss: {:03f} eval_accuracy: {:03f}'.format(eval_history[0], eval_history[1]))
    plot_curve(train_history)


def predict_eval_state():
    model_path = os.path.abspath(cfg_parse.model_save)
    if cfg_parse.model_predict is not None:
        model_path = os.path.abspath(cfg_parse.model_predict)
    model = tf.keras.models.load_model(model_path)

    # 由于tf.keras.models.save_model时设置include_optimizer=False,所以这里需要重新compile一下
    # metrics需要添加，不然后面model.fit 和model.evalute 返回的history会没有accuracy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    eval_start = time.time()
    eval_history = model.evaluate(test_ds, return_dict=True)
    eval_end = time.time()
    # eval_callback = eval_callbacks_list()
    plot_curve(eval_history)
    eval_time = eval_end - eval_start
    predict_images, true_labels = next(iter(test_ds))
    # predict_outputs = model(training=False).predict(predict_images, batch_size=BATCH_SIZE)
    predict_start = time.time()
    # predict_outputs = model(tf.expand_dims(predict_images[0], axis=0))
    predict_outputs = model.predict(predict_images)
    predict_end = time.time()
    predict_time = predict_end - predict_start
    class_name_list = signname_csv()
    eval_record = 'eval loss:{:.04f} eval accuracy:{:.04f}%'.format(eval_history['loss'], eval_history['accuracy'] * 100) + '\n' \
                  + 'evaluate time:{:.04f}s '.format(eval_time)
    predict_record = 'predict time per batch:{:.04f}s'.format(predict_time)
    # print(eval_record)
    print(predict_record)
    predict_visual(predict_images, predict_outputs, true_labels, class_name_list, eval_record, predict_record)


if __name__ == '__main__':
    if cfg_parse.predict:
        predict_eval_state()
    elif cfg_parse.train:
        train_state()
