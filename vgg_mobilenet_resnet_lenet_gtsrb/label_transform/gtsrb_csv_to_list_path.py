import os
import csv
import pickle
import random
import numpy as np
# import pandas as pd


# from utils.result_visual import plot_bar

# gtsrb_origin_path = 'F:\\AI-dataset\\GTSRB\GTSRB_Final_Training_Images\\GTSRB\\Final_Training\\Images'
# roi_path = 'F:\\AI-dataset\\GTSRB\\GTSRB_Training_Images_ROI_JPG'
# for c in range(0,43):
#
#     copy_from = os.path.join(gtsrb_origin_path,format(c,'05d'),'GT-' + format(c,'05d') + '.csv')
#     copy_to = os.path.join(roi_path,format(c,'05d'),'GT-' + format(c,'05d') + '.csv')
#     print(copy_from,'-->',copy_to)
#     shutil.copyfile(copy_from,copy_to)
#     print(os.listdir(os.path.join(roi_path,format(c,'05d'))))


# print('Read line:','Read lines:',test_annotation_without_sep,test_annotation_without_end)
# 可以使用np.random.choice(range,num,probability) 对数据集路径列表进行分层采样，来调节数据类别不平衡问题
# 使用scikit-learn 的k fold k折交叉验证

valid_rate = 0.2
train_base_path = 'F:\\AI-dataset\\GTSRB\\GTSRB_Training_Images_ROI_JPG'
test_base_path = 'F:\\AI-dataset\\GTSRB\\GTSRB_Test_Images_ROI_JPG'
pickel_base_path = 'F:\\AI-dataset\\traffic-signs-data'


def gtsrb_list_path():
    train_labels = []  # corresponding labels
    train_path_list = []
    test_labels = []
    test_path_list = []

    # get train_data
    for c in range(0, 43):
        prefix = train_base_path + '\\' + format(c, '05d') + '\\'  # subdirectory for class
        with open(prefix + 'GT-' + format(c, '05d') + '.csv') as train_csv:  # train annotations file
            train_csv_reader = csv.reader(train_csv, delimiter=';')  # csv parser for annotations file
            next(train_csv_reader)  # skip header
            # loop over all images in current annotations file
            for row in train_csv_reader:
                tr_image_name = row[0].replace('.ppm', '.jpg')
                tr_image_path = os.path.join(prefix, tr_image_name)
                train_path_list.append(tr_image_path)
                # train_images.append(plt.imread(tr_image_path) ) # 这个是直接读取图片文件，而不是路径the 1th column is the filename
                train_labels.append(int(row[7]))  # the 8th column is the label

    with open(os.path.join(test_base_path, 'GT-final_test.csv')) as test_csv:
        test_csv_reader = csv.reader(test_csv, delimiter=';')
        next(test_csv_reader)
        for row in test_csv_reader:
            te_image_name = row[0].replace('.ppm', '.jpg')
            te_image_path = os.path.join(test_base_path, te_image_name)
            test_path_list.append(te_image_path)
            # test_images.append(plt.imread(te_image_path))
            test_labels.append(int(row[7]))
    tem_list = list(np.transpose(np.vstack([train_path_list, train_labels])))
    # 将数据和标签合并配对，再打乱random.shuffle,打乱后切分train 和 valid,最后在提取整行，拆分数据 和 标签
    random.shuffle(tem_list)
    valid_len = int(len(tem_list) * valid_rate)
    valid_tem = tem_list[:valid_len]
    train_tem = tem_list[valid_len:]
    train_path_list, train_labels = np.transpose(train_tem)[0], list(map(int, np.transpose(train_tem)[1]))  # 要将标签转为int
    valid_path_list, valid_labels = np.transpose(valid_tem)[0], list(map(int, np.transpose(valid_tem)[1]))

    return train_path_list, train_labels, valid_path_list, valid_labels, test_path_list, test_labels


def gtsrb_class_num():
    train_class_num = {}
    train_list = os.listdir(train_base_path)
    for class_name in train_list:
        class_num = len(os.listdir(os.path.join(train_base_path, class_name))) - 1  # -1 是减去目录下csv文件数
        train_class_num.update({class_name: class_num})
    train_total_num = np.sum(list(train_class_num.values()))
    list_sort = np.mean(list(train_class_num.values()))
    test_num = len(os.listdir(test_base_path)) - 1
    return train_class_num, train_total_num, test_num, list_sort


# def pd_read_csv():

# def train_val_split():
#     order = np.random.choice(list, len(list), p=)


def use_pickle_load():
    training_file = os.path.join(pickel_base_path, 'train.p')
    validation_file = os.path.join(pickel_base_path, 'valid.p')
    testing_file = os.path.join(pickel_base_path, 'test.p')

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    x_train_pickle, y_train_pickle = train['features'], train['labels']
    x_valid_pickle, y_valid_pickle = valid['features'], valid['labels']
    x_test_pickle, y_test_pickle = test['features'], test['labels']
    print(len(x_train_pickle), len(x_valid_pickle), len(x_test_pickle))
    # x_train_pickle, y_train_pickle,, x_test_pickle, y_test_pickle
    return x_valid_pickle, y_valid_pickle


# def show_bar():
#     _, a, _, b, _, c = use_pickle_load()
#     for data in [a, b, c]:
#         plot_bar(data)


# def train_ds_shuffle(x_train_, y_train_):
#     order = np.arange(len(y_train_))
#     np.random.shuffle(order)
#     print(order)
#     x_train_shuffle = []
#     y_train_shuffle = []
#     for i in order:
#         x_train_shuffle.append(x_train_[i])
#         y_train_shuffle.append(y_train_[i])
#     print(len(x_train_shuffle), len(y_train_shuffle))
#     return x_train_shuffle, y_train_shuffle
