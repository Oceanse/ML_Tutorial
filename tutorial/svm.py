import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets



def svc():
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # 只取前两维特征，方便可视化
    y = iris.target
    # SVM 内部使用线性核函数，也就是线性SVM。线性SVM试图在特征空间中找到一个线性的决策边界，将数据分隔开
    # C=1: C 是 SVM 的正则化参数，控制了错误分类的惩罚程度。C 值越大，SVM 尝试尽可能正确分类所有训练样本，可能导致模型过拟合。较小的 C 值允许一些错误分类，以获得更广泛的决策边界，可能使模型泛化得更好。
    svc = svm.SVC(kernel='linear', C=1).fit(X, y)

    # 创建网格数据
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = (x_max / x_min) / 100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVC with linear kernel')
    plt.show()


if __name__ == "__main__":
    svc()