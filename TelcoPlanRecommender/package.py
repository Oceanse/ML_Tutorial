import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
from flask import Flask, request, jsonify
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif



def randomForestLowaccrate():
    # 加载数据并编码处理
    data = pd.read_csv('resource/consume2.csv')
    # 对标签列package编码：LabelEncoder将分类特征转换为整数值，这对于许多机器学习算法来说是必要的，因为它们通常要求输入是数值型的
    label_encoder = LabelEncoder()
    data['package'] = label_encoder.fit_transform(data['package'])
    # 对特征列gender编码：One-Hot encode；pd.get_dummies() 是一个用于执行独热编码（One-Hot Encoding）的函数，它通常用于将分类特征转换为数值特征
    data = pd.get_dummies(data, columns=['gender'])

    # 合并独热编码后的gender列，使得'gender'列以单个特征的形式出现，而不是因为one-hot编码后而被拆分的两列
    data['gender'] = data['gender_man'] + data['gender_woman']
    # 删除独热编码后的gender_F和gender_M列
    data = data.drop(['gender_man', 'gender_woman'], axis=1)

    # 提取特征向量：仅删除 User_ID and package columns
    features = data.drop(['User_ID', 'package'], axis=1)
    # 提取目标变量
    labels = data['package']

    # 选择10个最重要的特征
    k = 11
    # selector = SelectKBest(score_func=f_regression, k=k)
    # selector = SelectKBest(score_func=chi2, k=k)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit_transform(features, labels)
    # 可视化特征及其分数，了解每个特征与目标变量之间的关联强度
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features=features[selected_feature_names]

    # 计算average_traffic，合并三个特征为一个，并删除原始特征
    features['average_traffic'] = features[['0_traffic', '1_traffic', '2_traffic']].mean(axis=1)
    features.drop(['0_traffic', '1_traffic', '2_traffic'], axis=1, inplace=True)

    # 计算average_fee，合并三个特征为一个，并删除原始特征
    features['average_fee'] = features[['0_fee', '1_fee', '2_fee']].mean(axis=1)
    features.drop(['0_fee', '1_fee', '2_fee'], axis=1, inplace=True)
    print(features.head())
    # # 使用 pop 方法将 'package' 列弹出 DataFrame 并存储在变量中
    # package_column = features.pop('package')
    # # 使用 insert 方法将 'package' 列插入 DataFrame 的最后一列
    # features.insert(len(features.columns), 'package', package_column)
    # print(features.head())

    # 定位离群点索引,这里df删除了gender特征列，并进行了特征合并
    outliers_features_data = features['call_duration']
    # 计算选定特征的四分位数和IQR
    Q1 = outliers_features_data.quantile(0.25)
    Q3 = outliers_features_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((outliers_features_data < lower_bound) | (outliers_features_data > upper_bound))
    outliers_index_list = outliers[
        (outliers_features_data < lower_bound) | (outliers_features_data > upper_bound)].index.tolist()
    print("异常值索引:", outliers_index_list)
    num_rows = len(features)
    print("Number of rows in 'features before drop':", num_rows)
    # 删除离群点
    cleaned_features = features.drop(outliers_index_list)
    cleaned_labels = labels.drop(outliers_index_list)
    num_rows = len(cleaned_features)
    print("Number of rows in 'features after drop':", num_rows)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.2,
                                                        random_state=42)
    print(X_train.head()) # 输出训练集

    # 选择训练模型，这里选择随机森林分类器
    model = RandomForestClassifier(random_state=42)
    # model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)
    # report = classification_report(y_test, y_pred)
    # print('Classification Report:')
    # print(report)
    #
    # # 序列化保存模型到文件中
    # joblib.dump(model, 'resource/model.pkl')
def randomForestBack():
    # 加载数据：Load data
    data = pd.read_csv('resource/teleCom.csv')
    print(data.head())

    # 标签package列编码：LabelEncoder
    label_encoder = LabelEncoder()
    data['package'] = label_encoder.fit_transform(data['package'])
    # 特征gender列编码：One-Hot encode
    data = pd.get_dummies(data, columns=['gender'])

    # 提取特征向量：Drop User_ID and package columns
    features = data.drop(['User_ID', 'package'], axis=1)
    # 提取目标变量
    labels = data['package']

    # 提取最相关的4个特征：使用 SelectKBest 结合 f_regression 评分函数选择与目标变量 'package' 最相关的4个特征。
    k = 4
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit_transform(features, labels)
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features = features[selected_feature_names]
    print(features.head())

    # 通过箱线图找到离异点
    plt.figure(figsize=(10, 6))  # 绘制箱线图
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    df = pd.DataFrame(data)
    df.boxplot(column=selected_feature_names.tolist(), boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of Features')
    plt.ylabel('Values')
    plt.show()

    # 定位离群点索引
    outliers_features_data = features['call_time']
    # 计算选定特征的四分位数和IQR
    Q1 = outliers_features_data.quantile(0.25)
    Q3 = outliers_features_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((outliers_features_data < lower_bound) | (outliers_features_data > upper_bound))
    outliers_index_list = outliers[
        (outliers_features_data < lower_bound) | (outliers_features_data > upper_bound)].index.tolist()
    print("异常值索引:", outliers_index_list)
    # 删除离群点
    cleaned_features = features.drop(outliers_index_list)
    cleaned_labels = labels.drop(outliers_index_list)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.5,
                                                        random_state=42)
    print(X_train.head()) # 输出训练集

    # 选择训练模型，这里选择随机森林分类器
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(report)

    # 序列化保存模型到文件中
    joblib.dump(model, 'resource/model.pkl')
def randomForestBack2():
    # 加载数据：Load data
    data = pd.read_csv('resource/teleCom.csv')
    # 创建DataFrame
    df = pd.DataFrame(data)
    # 标签编码：Encode the package column
    label_encoder = LabelEncoder()
    data['package'] = label_encoder.fit_transform(data['package'])
    print(data.head())



    # 提取目标变量
    labels = data['package']
    # 对性别列One-Hot encode
    data = pd.get_dummies(data, columns=['gender'])
    # 提取特征向量：Drop User_ID and package columns
    features = data.drop(['User_ID', 'package'], axis=1)



    # 提取最相关的4个特征：使用 SelectKBest 结合 f_regression 评分函数选择与目标变量 'package' 最相关的4个特征。
    k = 4
    selector = SelectKBest(score_func=f_regression, k=k)
    # Fit the selector to the features and transform the features
    selector.fit_transform(features, labels)
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features = features[selected_feature_names]
    print(features.head())

    # 通过箱线图找到离异点
    plt.figure(figsize=(10, 6))  # 绘制箱线图
    # 设置箱线图颜色
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    df.boxplot(column=selected_feature_names.tolist(), boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of Features')
    plt.ylabel('Values')
    plt.show()

    # Assuming data is a NumPy array or a Pandas Series
    outliers_features_data = features['call_time']
    # 计算选定特征的四分位数和IQR
    Q1 = outliers_features_data.quantile(0.25)
    Q3 = outliers_features_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((outliers_features_data < lower_bound) | (outliers_features_data > upper_bound))
    outliers_index_list = outliers[
        (outliers_features_data < lower_bound) | (outliers_features_data > upper_bound)].index.tolist()
    print("异常值索引:", outliers_index_list)

    # 删除异常点
    cleaned_features = features.drop(outliers_index_list)
    cleaned_labels = labels.drop(outliers_index_list)

    # 将布尔类型的表示转换为数值型的 0 和 1
    # cleaned_features['gender_male'] = cleaned_features['gender_male'].astype(int)
    # cleaned_features['gender_female'] = cleaned_features['gender_female'].astype(int)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.5,
                                                        random_state=42)
    # 输出训练集
    print(X_train.head())

    # 选择模型，这里选择随机森林分类器
    model = RandomForestClassifier(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(report)

    # 序列化保存模型到文件中
    joblib.dump(model, 'resource/model.pkl')

def logisticRegression():
    # 读取数据
    data = pd.read_csv('resource/teleCom.csv')

    # 将套餐选择列作为标签，并将其编码为数值
    label_encoder = LabelEncoder()
    data['套餐选择(标签)'] = label_encoder.fit_transform(data['套餐选择(标签)'])
    labels = data['套餐选择(标签)']
    features = data.drop(['User_ID', '套餐选择(标签)'], axis=1)

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 选择模型，这里选择逻辑回归
    model = LogisticRegression(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(report)


def accurateTest():
    # 加载数据：Load data
    data = pd.read_csv('resource/teleCom2.csv')
    print(data.head())

    # 标签package列编码：LabelEncoder
    label_encoder = LabelEncoder()
    data['package'] = label_encoder.fit_transform(data['package'])
    # 特征gender列编码：One-Hot encode
    data = pd.get_dummies(data, columns=['gender'])

    # 提取特征向量：Drop User_ID and package columns
    features = data.drop(['User_ID', 'package'], axis=1)
    # 提取目标变量
    labels = data['package']

    # 提取最相关的4个特征：使用 SelectKBest 结合 f_regression 评分函数选择与目标变量 'package' 最相关的4个特征。
    k = 4
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit_transform(features, labels)
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features = features[selected_feature_names]
    print(features.head())

    # 设置每个箱子的填充颜色
    box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    # 通过箱线图找到离异点
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.figure(figsize=(10, 6))  # 绘制箱线图
    plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    df = pd.DataFrame(data)
    bp = df.boxplot(column=selected_feature_names.tolist(), boxprops=boxprops, medianprops=medianprops,
                    patch_artist=True)
    # 遍历子元素找到箱子并设置填充颜色
    for box, c in zip(bp.findobj(match=plt.matplotlib.patches.PathPatch), box_colors):
        box.set_facecolor(c)  # 设置箱子的颜色

    plt.title('Boxplot of Features')
    plt.ylabel('Values')
    plt.show()

    # Assuming data is a NumPy array or a Pandas Series
    outliers_features_data = features['call_time']
    # 计算选定特征的四分位数和IQR
    Q1 = outliers_features_data.quantile(0.25)
    Q3 = outliers_features_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((outliers_features_data < lower_bound) | (outliers_features_data > upper_bound))
    outliers_index_list = outliers[
        (outliers_features_data < lower_bound) | (outliers_features_data > upper_bound)].index.tolist()
    print("异常值索引:", outliers_index_list)

    # 删除异常点
    cleaned_features = features.drop(outliers_index_list)
    cleaned_labels = labels.drop(outliers_index_list)

    # 将布尔类型的表示转换为数值型的 0 和 1
    # cleaned_features['gender_male'] = cleaned_features['gender_male'].astype(int)
    # cleaned_features['gender_female'] = cleaned_features['gender_female'].astype(int)

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(cleaned_features, cleaned_labels, test_size=0.5,
                                                        random_state=42)
    # 输出训练集
    print(X_train.head())

    # 选择模型，这里选择随机森林分类器
    model = RandomForestClassifier(random_state=42)

    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:')
    print(report)

    # 序列化保存模型到文件中
    joblib.dump(model, 'resource/model.pkl')



if __name__ == "__main__":
    randomForestLowaccrate()
# logisticRegression()
# show()
