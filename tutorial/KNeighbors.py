import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def KNeighbors():
    # 生成虚拟数据集，生成 100 个样本点。
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    # 划分训练集和测试集，centers: 表示要生成的类别数，或者叫聚类中心的个数
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练KNN模型，也就是K最近邻分类器；  K表示选择最近的几个邻居进行投票
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = knn.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 可视化分类结果
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='x', s=200, cmap='viridis', linewidth=3)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Classification')
    plt.show()


if __name__ == "__main__":
    # 生成包含50条数据的啤酒数据集
    # beer_df = generate_beer_data(50)
    # 保存为CSV文件
    # beer_df.to_csv('resource/beer_data.csv', index=False)
    # print("啤酒数据集已生成并保存为 beer_data.csv")
    KNeighbors()