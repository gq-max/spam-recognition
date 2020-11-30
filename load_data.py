import scipy.io as scio
import numpy as np


def load_data():
    '''
    读取数据集， X共有1900个特征，训练集有4000个数据，测试集有1000个数据
    :return: train_X, train_y, test_X, test_y
    '''
    train_dataFile = '垃圾邮件训练和测试数据/spamTrain.mat'
    test_dataFile = '垃圾邮件训练和测试数据/spamTest.mat'
    # 1899个特征值，4000个数据
    train_data = scio.loadmat(train_dataFile)
    train_X = train_data.get("X")
    train_y = train_data.get("y")
    # 1899个特征值，1000个数据
    test_data = scio.loadmat(test_dataFile)
    test_X = test_data.get("Xtest")
    test_y = test_data.get("ytest")
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
