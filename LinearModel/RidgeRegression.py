import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, model_selection
def load_data():
    diabetes = datasets.load_diabetes()
    return model_selection.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)
def test_Ridge(*data):
    X_train, X_test, Y_train, Y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, Y_train)
    print('w: %s, d: %.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of square: %.2f' % np.mean((regr.predict(X_test) - Y_test) ** 2))
    print('Score: %.2f' % regr.score(X_test, Y_test))
    return None

X_train, X_test, Y_train, Y_test = load_data()
test_Ridge(X_train, X_test, Y_train, Y_test)

# 下面检验不同alpha值对于预测性能的影响
def test_Ridge_alpha(*data):
    X_train, X_test, Y_train, Y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    # enumerate()是python的内置函数
    # enumerate在字典上是枚举、列举的意思
    # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
    # enumerate多用于在for循环中得到计数
    for i, alpha in enumerate(alphas):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(X_train,Y_train)
        scores.append(regr.score(X_test,Y_test))
    # 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r'score')
    ax.set_xscale('log')
    ax.set_title('Ridge')
    plt.show()

    return None

(X_train, X_test, Y_train, Y_test) = load_data()
test_Ridge_alpha(X_train, X_test, Y_train, Y_test)