import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

def test_linear_regression():
    # boston = datasets.load_boston()
    # X = boston.data[:, np.newaxis, 5]  # 只取'B'列作为特征
    # y = boston.target

    # 假设数据集文件名为 'boston_housing.csv'，调整为你实际的文件名
    data = pd.read_csv('resource/boston.csv')

    # 提取特征集：从数据集 data 中删除名为 'MEDV' 的列(房价)，axis=1 表示沿着列的方向进行操作，也就是删除列，这样，X 就包含了除目标列之外的所有特征。
    # X 是一个包含了各种特征的矩阵，其中每一行是一个数据样本，每一列是一个特征
    X = data.drop('MEDV', axis=1)

    # 提取标签
    y = data['MEDV']  # 房价标签

    # 划分数据集为训练集和测试集，test_size=0.2 表示将数据划分为80%的训练集和20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建线性回归模型
    model = LinearRegression()
    # 使用训练集（X_train, y_train）来进行训练模型，在训练过程中，模型会调整自身的参数以最小化损失函数
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)
    # print('真实值:', y_test)
    # print('预测值:', y_pred)
    # print("x=", X)
    # print("y=", y)
    print("特征名：", X.columns)
    # 获取权重（系数）
    weights = model.coef_
    print('Weights (Coefficients):', weights)

    # 获取偏置（截距）
    intercept = model.intercept_
    print('Intercept:', intercept)

    # 绘制真实值和预测值折线图
    plt.figure(figsize=(10, 5), dpi=200)#置图形窗口的大小为宽度 10 英寸、高度 5 英寸，图形的分辨率为 200 DPI（每英寸点数）。

    # range(len(y_test)): x轴数据，表示折线图上点的位置，从0到len(y_test) - 1。
    # y_test: y轴数据，表示房价的真实值。
    # linestyle = '-': 线的样式为实线。
    # linewidth = 3: 线的宽度为3。
    # color = 'r': 线的颜色为红色(red)。
    # label = 'True Values': 图例的标签，用于标识这条线表示真实值。
    plt.plot(range(len(y_test)), y_test, linestyle='-', linewidth=3, color='r', label='True Values')
    plt.plot(range(len( y_pred)),  y_pred, linestyle='-', linewidth=3, color='b', label='Predicted Values')

    plt.grid(alpha=0.4, linestyle=':') # 绘制网格
    plt.legend()  #显示图例，标明红色线条对应真实值，蓝色线条对应预测值
    plt.xlabel('number')  # 设置x轴的标签文本
    plt.ylabel('prices')  # 设置y轴的标签文本
    plt.show()  # 显示房价分布和机器学习到的函数模型

    #均方误差（Mean Squared Error，MSE） 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)



def test_linear_regression2():

    # 假设数据集文件名为 'boston_housing.csv'，调整为你实际的文件名
    data = pd.read_csv('resource/boston.csv')
    # Extract features (X) and target (y)
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型前，选择与房价最相关的前5个特征
    k = 5  # Choose the top 5 features
    selector = SelectKBest(score_func=f_regression, k=k)

    # 对训练集和测试集的特征进行选择，
    X_train_selected = selector.fit_transform(X_train, y_train)

    # 获取打印选择的特征
    selected_feature_names = X_train.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)

    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    # Transform the test set using the same selected features
    X_test_selected = selector.transform(X_test)

    # Predict using the selected features
    y_pred = model.predict(X_test_selected)

    # Evaluate the model, for example using Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)

    # 绘制模型的预测结果与真实结果的散点图和预测结果的回归线
    plt.scatter(y_test,  y_pred, label="test")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'k--',
             lw=3,
             label="predict"
             )
    plt.show()



def test_linear_regression3():

    # 假设数据集文件名为 'boston_housing.csv'，调整为你实际的文件名
    data = pd.read_csv('resource/boston.csv')

    # 提取特征集
    X = data.drop('MEDV', axis=1)

    # 提取标签
    y = data['MEDV']  # 房价标签

    # 划分数据集为训练集和测试集，test_size=0.2 表示将数据划分为80%的训练集和20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化:将特征的值缩放到均值为0、方差为1的范围内
    # 一些机器学习算法，如线性回归、支持向量机、神经网络等，对特征的尺度（单位）比较敏感，因此在这些算法中，通常会对特征进行标准化，
    # 以确保各个特征对模型的影响权重大致相等。
    # 如果你使用的是决策树、随机森林等不敏感于特征尺度的算法，可能就不需要进行标准化。
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 创建线性回归模型
    model = LinearRegression()
    # 使用训练集（X_train, y_train）来进行训练模型，在训练过程中，模型会调整自身的参数以最小化损失函数
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 绘制真实值和预测值折线图
    plt.figure(figsize=(10, 5), dpi=200)#置图形窗口的大小为宽度 10 英寸、高度 5 英寸，图形的分辨率为 200 DPI（每英寸点数）。
    plt.plot(range(len(y_test)), y_test, linestyle='-', linewidth=3, color='r', label='True Values')
    plt.plot(range(len( y_pred)),  y_pred, linestyle='-', linewidth=3, color='b', label='Predicted Values')
    plt.grid(alpha=0.4, linestyle=':')
    plt.legend()
    plt.xlabel('number')
    plt.ylabel('prices')
    plt.show()

    #均方误差（Mean Squared Error，MSE） 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)




def test_linear_regression4():

    # 假设数据集文件名为 'boston_housing.csv'，调整为你实际的文件名
    data = pd.read_csv('resource/boston.csv')

    # 提取特征集
    X = data.drop('MEDV', axis=1)

    # 提取标签
    y = data['MEDV']  # 房价标签

    # 划分数据集为训练集和测试集，test_size=0.2 表示将数据划分为80%的训练集和20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     #测试发现，数据如果不标准化，均方误差大的离谱
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 创建线性回归模型
    #learning_rate： 用于控制模型参数更新的步长，学习率过大可能导致错过损失函数的最小值，学习率过小可能导致训练过程缓慢或者陷入局部最小值;constant指定学习率采用常数方式
    #eta0=0.001：初始学习率的值。这是学习率的初始设定值
    #max_iter = 10000：指定最大迭代次数,确保模型不会无限迭代。
    model = SGDRegressor(learning_rate="constant", eta0=0.001, max_iter=1000)
    # 使用训练集（X_train, y_train）来进行训练模型，在训练过程中，模型会调整自身的参数以最小化损失函数
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    #均方误差（Mean Squared Error，MSE） 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error4:', mse)




def test_linear_regression5():

    # 假设数据集文件名为 'boston_housing.csv'，调整为你实际的文件名
    data = pd.read_csv('resource/boston.csv')

    # 提取特征集
    X = data.drop('MEDV', axis=1)

    # 提取标签
    y = data['MEDV']  # 房价标签

    # 划分数据集为训练集和测试集，test_size=0.2 表示将数据划分为80%的训练集和20%的测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     #测试发现，数据如果不标准化，均方误差大的离谱
    transfer = StandardScaler()
    X_train = transfer.fit_transform(X_train)
    X_test = transfer.transform(X_test)

    # 创建岭回归模型,只不过在算法建立回归方程时候，加上正则化的限制，防止过拟合
    # alpha: 这是岭回归的正则化参数，控制了正则化的强度。较大的 alpha 会更大程度削弱模型的权重趋，从而限制模型的复杂度。
    # max_iter: 这是迭代的最大次数，用于控制模型的求解过程。
    model = Ridge(alpha=0.5, max_iter=10000)
    # 使用训练集（X_train, y_train）来进行训练模型，在训练过程中，模型会调整自身的参数以最小化损失函数
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    #均方误差（Mean Squared Error，MSE） 评估模型
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error5:', mse)

if __name__ == "__main__":
    test_linear_regression5()