import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def test_logic_regression():
    # 加载乳腺癌数据集，这个数据集的结构是一个字典
    # data: 包含特征数据，每一行代表一个样本，每一列代表一个特征。
    # target: 包含目标变量，通常是我们要预测的结果。
    # feature_names: 包含特征的名字，通常对应data中的列名。
    # DESCR: 数据集的描述信息，包括数据来源、特征描述等
    data = load_breast_cancer()
    print("data:",data['data'])  # 输出特征数据
    print("target:",data['target'])  # 输出目标变量
    print("feature_names:", data['feature_names'])  # 输出特征名字
    print("DESCR:",data['DESCR'])  # 输出描述信息

    #利用 Pandas 库创建了一个 DataFrame（数据表格），其中将特征数据（data.data）作为数据，特征名称（data.feature_names）作为列名。
    X = pd.DataFrame(data.data, columns=data.feature_names)
    #使用 Pandas 库创建了一个 Series（序列），将 data.target 中的数据作为序列的数据，同时指定了序列的名称为 'target'；在Pandas库中，Series（序列）是一种一维标记数组的数据结构，
    y = pd.Series(data.target, name='target')

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建并训练逻辑回归模型
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算模型准确率accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', accuracy)

    # 计算模型召回率
    #混淆矩阵中，行代表实际的类别，列代表预测的类别。通常包括四个重要的指标
    # 真正例 (True Positives, TP): 模型将正类样本正确预测为正类的数量。
    # 真负例 (True Negatives, TN): 模型将负类样本正确预测为负类的数量。
    # 假正例 (False Positives, FP): 模型将负类样本错误预测为正类的数量。
    # 假负例 (False Negatives, FN): 模型将正类样本错误预测为负类的数量。
    conf_matrix = confusion_matrix(y_test, y_pred)    # 计算混淆矩阵
    # 从混淆矩阵中获取 真正例TP 和 假负例FN
    TP = conf_matrix[1, 1]  # 真正例
    FN = conf_matrix[1, 0]  # 假负例
    recall = TP / (TP + FN)
    # 打印召回率
    print("Recall:", recall)

    # 输出分类报告：包括每个类别的精确率、召回率、F1 分数和支持数量。
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

def test_logic_regression2():
    # 加载乳腺癌数据集，这个数据集的结构是一个字典
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建并训练逻辑回归模型
    model = LogisticRegression(
        penalty='l2',  # 正则化类型: 'l1', 'l2', 默认为 'l2'
        C=1.0,  # 正则化强度的倒数，C值越小，表示正则化强度越大；正则化强度越大意味着对模型参数的惩罚越强，模型会更倾向于选择较小的权重参数，从而降低模型的复杂度，防止过拟合。
        solver='lbfgs',  # 优化算法: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
        max_iter=100,  # 最大迭代次数
        random_state=None,  # 随机种子
        fit_intercept=True,  # 是否计算截距（偏置）
        verbose=0,  # 控制详细程度
        class_weight=None,  # 类权重，可以是字典或者 'balanced'
    )

    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算模型准确率accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Model Accuracy:', accuracy)

    # 计算模型召回率
    conf_matrix = confusion_matrix(y_test, y_pred)    # 计算混淆矩阵
    # 从混淆矩阵中获取 真正例TP 和 假负例FN
    TP = conf_matrix[1, 1]  # 真正例
    FN = conf_matrix[1, 0]  # 假负例
    recall = TP / (TP + FN)
    # 打印召回率
    print("Recall:", recall)

    # 输出分类报告：包括每个类别的精确率、召回率、F1 分数和支持数量。
    print('Classification Report:')
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    test_logic_regression2()