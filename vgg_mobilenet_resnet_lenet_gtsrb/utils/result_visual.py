# import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 加载tf.data.Dataset后的数据集查看，特别是看batch_size的数据是否随机
signname_path = 'F:\\AI-dataset\\GTSRB\\signnames_ch.csv'
# plt显示中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']


def signname_csv():
    with open(signname_path, 'r', encoding='UTF-8') as f:
        signnames = pd.read_csv(f)
        signname_list = signnames.SignName
    return signname_list


def data_image_visual(dataset_iter):
    """
    传入的参数是tf.data.Dataset 或者 tf.keras.preprocessing.image.ImageDataGenerator
    等生成的数据迭代器iter;所谓的迭代器iter就是可以循环的量，可以通过iter(),next()来转换
    :param dataset_iter:
    :return:
    """
    n = 0
    signname = signname_csv()
    for images, labels in dataset_iter:
        # print(images, labels)
        m = 0
        plt.figure(figsize=(5, 5))
        for i in np.random.randint(0, 128, 20):
            plt.subplot(5, 4, m + 1)
            plt.title(str(int(labels[i])) + signname[int(labels[i])])
            plt.imshow(images[i])
            m += 1
        n += 1
        if n == 2:
            break
    plt.show()


# 可视化数据集分布的柱状图
def plot_bar(x_data):
    x, num_x = np.unique(x_data, return_counts=True)  # np.unique 返回去除传入参数的重复值后并按小到大排序后的序列，return_counts 返回重复值的次数
    total_num = np.sum(num_x)
    plt.bar(x, num_x, align='center')  # plt.bar 显示柱状图
    plt.xlabel('class')
    plt.ylabel('Frequency')
    plt.xlim([-1, 43])
    title_name = 'frequency of labels in ' + str(total_num) + ' dataset'
    plt.title(title_name)
    plt.show()


# 可视化模型训练曲线图#
def plot_curve(history):
    accuracy = history['accuracy']
    loss = history['loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(accuracy, label='Training accuracy')
    if 'val_accuracy' in history:
        val_accuracy = history['val_accuracy']
        plt.plot(val_accuracy, label='Validation accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    if 'val_loss' in history:
        val_loss = history['val_loss']
        plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([-0.01, 3.5])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# 可视化预测输出是否正确
def predict_visual(predict_images, predict_output, y_true_labels, class_name_list, eval_record, predict_record):
    predict_num = predict_output.shape[0]  # batch size 大小
    # 索引对应的标签名字
    # 例如cifar10数据集的类别名
    # class_name_list=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    label_index = [np.argmax(predict_output[i]) for i in range(predict_num)]
    label_name = [class_name_list[j] for j in label_index]
    print(label_index)
    print(label_name)
    # print([y_true_labels[i] for i in range(predict_num)])

    plt.figure(figsize=(20, 20))
    # top = 0.865, bottom = 0.0, left = 0.29, right = 0.695, hspace = 0.265, wspace = 0.0
    # hspace=0.265, wspace=0.470, top=0.865, bottom=0.0, left=0.02, right=0.98
    plt.subplots_adjust(top=0.865, bottom=0.0, left=0.29, right=0.695, hspace=0.265, wspace=0.0)
    # 随机显示50个预测图片
    n = 0
    false_count = 0
    for images_i in np.random.randint(0, predict_num, 20):
        n += 1
        plt.subplot(5, 4, n)
        plt.imshow(predict_images[images_i])
        color = 'green' if y_true_labels[images_i] == label_index[images_i] else 'blue'
        true_or_false = '-T' if y_true_labels[images_i] == label_index[images_i] \
            else '-F [' + label_name[y_true_labels[images_i]] + ']'
        false_count = false_count + 1 if y_true_labels[images_i] != label_index[images_i] else false_count
        plt.title(label_name[images_i] + true_or_false,
                  color=color, fontsize='x-large', fontweight='bold')
        plt.axis(False)
        _ = plt.suptitle('True=Green,False=Blue FalseCounts=' + str(false_count) +
                         '\n'  # \n前这么多空格，是为了垂直方向行与行之间左边对其时，标题整体仍然可以在中心
                         + eval_record + predict_record,
                         color='black', fontsize='xx-large', fontweight='heavy', )
        # horizontalalignment='left', verticalalignment='top')
        # plt.tight_layout()

    plt.show()

# def plot_cm(labels, predictions, p=0.5):
#     cm = confusion_matrix(labels, predictions > p)
#     plt.figure(figsize=(5, 5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title('Confusion matrix @{:.2f}'.format(p))
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')
#
#     print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#     print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#     print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#     print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#     print('Total Fraudulent Transactions: ', np.sum(cm[1]))
#
#
# def plot_roc(name, labels, predictions, **kwargs):
#     fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
#
#     plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
#     plt.xlabel('False positives [%]')
#     plt.ylabel('True positives [%]')
#     plt.xlim([-0.5, 20])
#     plt.ylim([80, 100.5])
#     plt.grid(True)
#     ax = plt.gca()
#     ax.set_aspect('equal')
#
#
# def plot_loss(history, label, n):
#     # Use a log scale to show the wide range of values.
#     plt.semilogy(history.epoch, history.history['loss'],
#                  color=colors[n], label='Train ' + label)
#     plt.semilogy(history.epoch, history.history['val_loss'],
#                  color=colors[n], label='Val ' + label,
#                  linestyle="--")
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#
#     plt.legend()
#
#
# def plot_metrics(history):
#     metrics = ['loss', 'auc', 'precision', 'recall']
#     for n, metric in enumerate(metrics):
#         name = metric.replace("_", " ").capitalize()
#         plt.subplot(2, 2, n + 1)
#         plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
#         plt.plot(history.epoch, history.history['val_' + metric],
#                  color=colors[0], linestyle="--", label='Val')
#         plt.xlabel('Epoch')
#         plt.ylabel(name)
#         if metric == 'loss':
#             plt.ylim([0, plt.ylim()[1]])
#         elif metric == 'auc':
#             plt.ylim([0.8, 1])
#         else:
#             plt.ylim([0, 1])
#
#         plt.legend()

# %matplotlib inline
# plot = plot_curve(history)
# import tensorboardcolab
# %load_ext tensorboard
# #tensorboardcolab --logdir=/CheckPoint/
# %tensorboard --logdir CheckPoint/vgg13_cifar10_ep50/
