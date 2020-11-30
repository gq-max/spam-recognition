from load_data import load_data
import numpy as np
import time


def hinge(z):
    """hinge损失函数"""
    return np.where(z < 1, 1 - z, 0)


def hinge_gradient(z):
    """hinge损失函数的梯度"""
    return np.where(z < 1, -1, 0)


def predict(w, x, b):
    """计算值"""
    x = x.reshape(-1, 1)
    return np.dot(w.T, x) + b


def pegasos(X, y, T=1000, lam=1):
    m, n = X.shape  # m表示X的样本个数， n表示每个样本的特征个数
    # w = np.random.randn(n, 1)
    # b = np.random.randn(1)
    w = np.zeros((n, 1))
    b = 0
    for t in range(1, T + 1):
        eta = 1.0 / (lam * t)
        i = np.random.randint(m)
        p = predict(w, X[i], b)
        if y[i] * p < 1:
            w = (1.0 - 1 / t) * w + eta * y[i] * (X[i].reshape(-1, 1))
            b += eta * y[i]
        else:
            w = (1.0 - 1 / t) * w
            b = b
    return w, b


def batch_pegasos(X, y, T=1000, lam=0.1, size=10):
    m, n = X.shape  # m表示X的样本个数， n表示每个样本的特征个数
    # w = np.random.randn(n, 1)
    # b = np.random.randn(1)
    w = np.zeros((n, 1))
    b = 0
    dataIndex = np.arange(m)
    for t in range(1, T + 1):
        eta = 1.0 / (lam * t)
        np.random.shuffle(dataIndex)
        for j in range(size):
            i = dataIndex[j]
            p = predict(w, X[i], b)
            if y[i] * p < 1:
                w = (1.0 - 1 / t) * w + eta * y[i] * (X[i].reshape(-1, 1))
                b += eta * y[i]
            else:
                w = (1.0 - 1 / t) * w
                b = b
    return w, b


def accuracy(w, b, X, y):
    y_predict = np.array([predict(w, x, b) for x in X])
    y_predict = np.array([[1 if i > 0 else -1] for i in y_predict])
    num = 0
    for (i, j) in zip(y_predict, y):
        if i == j:
            num += 1
    acc = num / len(y) * 100
    return acc


def save(y_test, y_predict):
    fn = "预测值与测试集标签比较.txt"
    y_predict = [[True if i > 0 else False] for i in y_predict]
    y_test = [[True if i > 0 else False] for i in y_test]
    num = 0
    with open(fn, "w") as file_obj:
        string = "(标签，预测值)          "
        file_obj.write(string + string + string + string + "\n")
        for i, j in zip(y_test, y_predict):
            num += 1
            if num % 4 == 0:
                string = str((i, j)) + '    '
                file_obj.write(string + "\n")
            else:
                string = str((i, j)) + '    '
                file_obj.write(string)


def main():
    X_train, y_train, X_test, y_test = load_data()
    y_train = np.array([[1 if y == 1 else -1] for y in y_train])
    y_test = np.array([[1 if y == 1 else -1] for y in y_test])
    start = time.time()
    w, b = batch_pegasos(X_train, y_train, 2000, lam=0.1)
    acc_test = accuracy(w, b, X_test, y_test)
    end = time.time()
    print("在测试集上精度为：{}，时间为：{}".format(acc_test, end - start))
    y_predict = np.array([predict(w, x, b) for x in X_test])
    y_predict = np.array([[1 if i > 0 else -1] for i in y_predict])
    save(y_test, y_predict)


main()
