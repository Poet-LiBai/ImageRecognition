import datetime
import os

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ProgbarLogger, History
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def train_callback_list(base_path='F:\\AI-modelsaver\\GTSRB\\LeNet5'):
    # ("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base_path, datetime.datetime.now().strftime("%Y-%m-%d"),
                           datetime.datetime.now().strftime("%H-%M-%S"))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练参数设置
    # 使用callback回调函数设置训练过程数据记录和终止条件
    # 设置tensorboard的保存目录,histogram权重直方图统计，write_graph模型计算流程节点图保存，write_image权重图可视化
    # 可以在命令行终端输入命令 tensorboard --logdir=日志保存目录即以下的log_dir
    tensorboard_log = TensorBoard(os.path.join(log_dir, 'TensorBoard'),
                                  histogram_freq=1, write_graph=True, write_images=True,
                                  update_freq='epoch')

    # csv_log = CSVLogger(os.path.join(log_dir, 'TensorBoard', 'train_log.csv'))
    # 设置checkpoint仅保存训练权重文件，以val_loss验证损失来监控指标,period=1表示每 1 epoch都保留一次权重文件

    # modelcheckpoint_path = os.path.join(log_dir, 'ModelCheckPoint')
    # if not os.path.exists(modelcheckpoint_path):
    #     os.makedirs(modelcheckpoint_path)
    # checkpoint = ModelCheckpoint(
    #     os.path.join(modelcheckpoint_path,
    #                  'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}'
    #                  '-acc{accuracy:.3f}-val_acc{val_accuracy:.3f}.ckpt'),
    #     monitor='val_loss', save_weights_only=False, save_best_only=True, period=10)

    # 当model不在改善时，在等待patience epoch轮后都没有提供，那就按照factor来衰减学习率
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

    # 当model的监控指标改变量，小于min_delta时，视为没有改善，所以在等待patience epoch 轮后就停止训练
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # probar_log = ProgbarLogger()
    callbacks_ = [tensorboard_log,
                  # csv_log,
                  # probar_log,
                  # checkpoint,
                  reduce_lr,
                  early_stopping
                  ]
    return callbacks_


def eval_callbacks_list(base_path='F:\\AI-modelsaver\\GTSRB\\LeNet5'):
    eval_log_dir = os.path.join(base_path, datetime.datetime.now().strftime("%Y-%m-%d"),
                                datetime.datetime.now().strftime("%H-%M-%S"), 'Tensorboard', 'evaluate')
    if not os.path.exists(eval_log_dir):
        os.makedirs(eval_log_dir)

    eval_callback = [TensorBoard(os.path.join(eval_log_dir), update_freq='batch'), History()]
    return eval_callback


def summary_custom():
    logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()


def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 0.2
    if epoch > 10:
        learning_rate = 0.02
    if epoch > 20:
        learning_rate = 0.01
    if epoch > 50:
        learning_rate = 0.005

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate
