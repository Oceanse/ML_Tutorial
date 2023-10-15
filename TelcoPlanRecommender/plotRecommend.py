import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

# 套餐类型分布图
def plot():
    data = pd.read_csv('resource/consume.csv')
    df = pd.DataFrame(data)
    print(df.columns)
    print(df.shape)
    # 使用matplotlib来绘制柱状图
    plt.figure(figsize=(10, 5))
    # 计算了"package"列中所有类型的数量，并绘制柱状图
    df['package'].value_counts().plot(kind='bar')
    plt.title('Distribution of package types', fontsize=16, weight='bold')
    plt.xlabel('Package Type')
    plt.ylabel('Count')
    plt.show()



#  选择特征; 柱状图，展示每个特征的得分。特征的得分越高，其与目标变量的关系越强。
def selectFeature():
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

    # 选择11个最重要的特征
    k = 10
    # selector = SelectKBest(score_func=f_regression, k=k)
    # selector = SelectKBest(score_func=chi2, k=k)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit_transform(features, labels)
    # 可视化特征及其分数，了解每个特征与目标变量之间的关联强度
    scores = selector.scores_
    plt.figure(figsize=(12, 8))
    sns.barplot(x=features.columns, y=scores)
    plt.title("Feature importance based on Univariate Selection")
    plt.ylabel("F Score")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.show()
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    # features = features[selected_feature_names]
    # print(features.head())


# 选择特征--->绘制箱线图查看离群点
def boxplot():
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
    k = 10
    # selector = SelectKBest(score_func=f_regression, k=k)
    # selector = SelectKBest(score_func=chi2, k=k)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit_transform(features, labels)
    selected_feature_names = features.columns[selector.get_support()]
    features = features[selected_feature_names]
    print(features.head())

    # 设置每个箱子的填充颜色
    box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightsalmon', 'lightpink', 'lightseagreen',
                  'lightsteelblue', 'lightcyan', 'lightgoldenrodyellow', 'lightgray']

    # 通过箱线图找到离异点
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.figure(figsize=(15, 6))  # 绘制箱线图
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



# 选择特征--->绘制箱线图剔除离群点--->合并特征，简化模型特征的数量
def mergeFeature():
    # 加载数据并编码处理
    data = pd.read_csv('resource/consume.csv')
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

    # 选择11个最重要的特征
    k = 10
    # selector = SelectKBest(score_func=f_regression, k=k)
    # selector = SelectKBest(score_func=chi2, k=k)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit_transform(features, labels)
    # 可视化特征及其分数，了解每个特征与目标变量之间的关联强度
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    new_data = pd.concat([features[selected_feature_names], labels], axis=1)

    # 合并特征
    df = pd.DataFrame(new_data)
    # # 计算average_traffic，合并三个特征为一个，并删除原始特征
    df['average_traffic'] = df[['0_traffic', '1_traffic', '2_traffic']].mean(axis=1)
    df.drop(['0_traffic', '1_traffic', '2_traffic'], axis=1, inplace=True)
    # # 计算average_fee，合并三个特征为一个，并删除原始特征
    df['average_fee'] = df[['0_fee', '1_fee', '2_fee']].mean(axis=1)
    df.drop(['0_fee', '1_fee', '2_fee'], axis=1, inplace=True)

    # 使用 pop 方法将 'package' 列弹出 DataFrame 并存储在变量中
    package_column = df.pop('package')
    # 使用 insert 方法将 'package' 列插入 DataFrame 的最后一列
    df.insert(len(df.columns), 'package', package_column)

    # # 打印结果
    print(df)




# 它是基于模型训练的结果来评估每个特征对模型预测的贡献
def featureImportance():
    data = pd.read_csv('resource/consume.csv')

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
    k = 11
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit_transform(features, labels)
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features = features[selected_feature_names]

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
    print(X_train.head())  # 输出训练集

    # 选择训练模型，这里选择随机森林分类器
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 获取特征重要性
    feature_importances = model.feature_importances_

    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importances)
    plt.barh(X_train.columns[sorted_idx], feature_importances[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.show()




if __name__ == "__main__":
    boxplot()

