import os
import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import struct
import matplotlib.pyplot as plt

import tensorflow_core as tfc
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

data_len = 150
train_num = 168000
test_num = 72000
class_num = 2

batch_size = 128
epochs = 60
validation_split = 0.2

data_train_name = "data_train.dat"
label_train_name = "label_train.dat"
data_test_name = "data_test.dat"
label_test_name = "label_test.dat"
model_h5_name = "ecg_model.h5"
model_pb_name = "ecg_model.pb"
pre_result_name = "pre_result.txt"


# 保存预测结果文件，便于和后面C++的做对比
def save_prediction_to_text(prediction, save_num):
    if save_num >= test_num:
        save_num = test_num - 1
    pre_result_path = os.path.join(os.getcwd(), "pre_result")
    if not os.path.exists(pre_result_path):
        os.mkdir(pre_result_path)
    np.savetxt(os.path.join(pre_result_path, pre_result_name),
               prediction[0:save_num],
               delimiter=',  ')


# 数据归一化，采用最大最小方式
def normalization(data):
    data_range = np.max(data) - np.min(data)
    if 0 == data_range:
        return data
    else:
        return (data - np.min(data)) / data_range


# 获取数据
def load_data(filename, sample_num, sample_len):
    fp = open(filename, "rb")
    if sample_len == data_len:
        context = fp.read(2 * sample_num * sample_len)
        fmt_unpack = '%d' % (sample_num * sample_len) + 'h'
        dat_arr = np.array(struct.unpack(fmt_unpack, context), dtype=float)
        dat_arr = dat_arr.reshape((sample_num, sample_len))
        for i in range(0, sample_num):
            dat_arr[i] = normalization(dat_arr[i])
        dat_arr = dat_arr.reshape(-1, data_len, 1)
    else:
        context = fp.read(sample_num * sample_len)
        fmt_unpack = '%d' % (sample_num * sample_len) + 'B'
        dat_arr = np.array(struct.unpack(fmt_unpack, context))
    fp.close()
    return dat_arr


# 获取训练和测试的数据
def get_data():
    data_path = os.path.join(os.getcwd(), "data")

    # 训练数据，35个个体，168000个心拍，N与V比例1:1，前半部分为N，后半部分为V
    data_train = load_data(os.path.join(data_path, data_train_name), train_num, data_len)
    label_train = load_data(os.path.join(data_path, label_train_name), train_num, 1)

    # 测试数据，15个个体，72000个心拍，N与V比例1:1，前半部分为N，后半部分为V
    data_test = load_data(os.path.join(data_path, data_test_name), test_num, data_len)
    label_test = load_data(os.path.join(data_path, label_test_name), test_num, 1)

    return data_train, label_train, data_test, label_test


# 构建CNN模型
def build_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(data_len, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    return new_model


# 自定义学习率
def scheduler(epoch, current_learning_rate):
    if epoch == 19 or epoch == 39 or epoch == 59:
        return current_learning_rate / 5
    else:
        return min(current_learning_rate, 0.01)


# 绘制accuracy和loss曲线
def plot_history(history, metric):
    acc = history.history[metric]
    val_acc = history.history['val_' + metric]
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def save_model_to_pb():
    model_h5 = os.path.join(os.getcwd(), "model_h5")
    if not os.path.exists(model_h5):
        os.mkdir(model_h5)
    model_h5_path = os.path.join(model_h5, model_h5_name)

    model_pb = os.path.join(os.getcwd(), "model_pb")
    if not model_pb:
        os.mkdir(model_pb)

    model = tf.keras.models.load_model(model_h5_path,
                                       compile=False)
    model.summary()
    full_model = tf.function(lambda Input: model(Input))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape,
                                                                model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=model_pb,
                      name=model_pb_name,
                      as_text=False)


def main():
    data_train, label_train, data_test, label_test = get_data()

    project_path = os.getcwd()

    model_h5 = os.path.join(project_path, "model_h5")
    if not os.path.exists(model_h5):
        os.mkdir(model_h5)
    model_h5_path = os.path.join(model_h5, model_h5_name)

    load_model_flag = 0
    if os.path.exists(model_h5_path) and 1 == load_model_flag:
        # 导入训练好的模型
        model = tf.keras.models.load_model(filepath=model_h5_path)
    else:
        metric = 'accuracy'
        model = build_model()
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=[metric])
        model.summary()

        # 定义TensorBoard对象
        log_dir_path = os.path.join(project_path, "logs")
        if not os.path.exists(log_dir_path):
            os.mkdir(log_dir_path)
        log_dir = os.path.join(log_dir_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        ts_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # 学习率回调函数
        lr_callback = tfc.python.keras.callbacks.LearningRateScheduler(schedule=scheduler)

        # 训练与验证
        history = model.fit(data_train, label_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=validation_split,
                            callbacks=[ts_callback, lr_callback])

        # 绘制accuracy和loss曲线
        plot_history(history, metric)

        # 保存h5模型
        model.save(filepath=model_h5_path)

        # 保存pb模型
        save_model_to_pb()

    pred = model.predict(data_test)
    label_pred = np.argmax(pred, axis=1)

    # 保存前10个样本的预测结果
    save_prediction_to_text(pred, 10)

    acc = np.mean(label_pred == label_test)
    conf_mat = confusion_matrix(label_test, label_pred)  # 利用专用函数得到混淆矩阵
    acc_n = conf_mat[0][0] / np.sum(conf_mat[0])
    acc_v = conf_mat[1][1] / np.sum(conf_mat[1])

    print('\nAccuracy=%.3f%%' % (acc * 100))
    print('Accuracy_N=%.3f%%' % (acc_n * 100))
    print('Accuracy_V=%.3f%%' % (acc_v * 100))

    print('\nConfusion Matrix:\n')
    print(conf_mat)


if __name__ == '__main__':
    main()
