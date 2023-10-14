from sklearn.datasets import load_iris                  # 获取数据集
from sklearn.model_selection import train_test_split    # 划分数据集
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz



def decision_iris():
    """
    决策树对鸢尾花进行分类
    :return:
    """
    # 1）获取数据集
    iris = load_iris()

    # 2）划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)  # 随机数种子

    # 不用做特征工程：标准化
    # 3）决策树预估器
    #criterion='entropy'：这是决策树划分标准的选择，可以是'gini'表示基尼系数或'entropy'表示信息熵。在这里选择了信息熵作为划分标准。
    estimator = DecisionTreeClassifier(criterion='entropy')
    estimator.fit(x_train, y_train)

    # 4）模型评估
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print("读真实值和预测值：\n", y_test == y_predict)  # 直接比对

    # 方法2：计算准确率
    score = estimator.score(x_test, y_test)  # 测试集的特征值，测试集的目标值
    print("准确率：", score)

    # 生成决策树的可视化，并将输出保存到名为 'iris_tree.dot' 的文件中
    export_graphviz(estimator, out_file='iris_tree.dot', feature_names=iris.feature_names)

    return None


if __name__ == "__main__":
    decision_iris()