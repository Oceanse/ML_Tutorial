from flask import Flask, jsonify, request, render_template, send_file
import numpy as np
import joblib
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif
import seaborn as sns

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes


@app.route('/')
def index():
    # 模板文件,存放在 项目根目录/templates/index.html
    return render_template('index.html')


# 预测套餐类型
@app.route('/api/predict', methods=['POST'])
def predict():
    loaded_model = joblib.load('resource/model.pkl')
    data = request.get_json(force=True)
    features = [
        data['age'],
        data['call_duration'],
        data['call_count'],
        data['Online_Time'],
        data['average_traffic'],
        data['average_fee']
    ]

    # 一维数组转换为一个矩阵(二维数组)
    features = np.array(features).reshape(1, -1)
    prediction = loaded_model.predict(features)
    print(prediction)
    class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
    predicted_label = class_mapping[prediction[0]]

    return jsonify({'predicted_label': predicted_label})



# 获取数据集
@app.route('/api/get_data', methods=['GET'])
def get_data():
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
    k = 10
    # selector = SelectKBest(score_func=f_regression, k=k)
    # selector = SelectKBest(score_func=chi2, k=k)
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit_transform(features, labels)
    # 可视化特征及其分数，了解每个特征与目标变量之间的关联强度
    # Get the selected feature names
    selected_feature_names = features.columns[selector.get_support()]
    print("Selected features:", selected_feature_names)
    features = features[selected_feature_names]

    # 计算average_traffic，合并三个特征为一个，并删除原始特征
    features['average_traffic'] = features[['0_traffic', '1_traffic', '2_traffic']].mean(axis=1)
    features.drop(['0_traffic', '1_traffic', '2_traffic'], axis=1, inplace=True)

    # 计算average_fee，合并三个特征为一个，并删除原始特征
    features['average_fee'] = features[['0_fee', '1_fee', '2_fee']].mean(axis=1)
    features.drop(['0_fee', '1_fee', '2_fee'], axis=1, inplace=True)
    print(features.columns)
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

    # 把测试集的特征列和标签列进行合并，并把标签从 0、1、2、3 映射到 A、B、C、D
    data = pd.concat([X_test, y_test], axis=1)


    page = request.args.get('page', type=int)#页码
    page_size = request.args.get('pageSize', type=int)#页面大小

    # Calculate start and end indices for pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # Get the specified page of data
    paginated_data = data.iloc[start_idx:end_idx].to_dict(orient='records')
    print('data', paginated_data)

    # 计算总页数，并将其包含在返回的 JSON 数据
    total_rows = len(data)
    total_pages = math.ceil(total_rows / page_size)

    # 确保 start_idx 和 end_idx 不超出数据范围。如果超出范围，返回空数据或合适的错误信息。
    if start_idx >= total_rows:
        return jsonify({'data': []})

    end_idx = min(end_idx, total_rows)  # Ensure end index is within data range
    paginated_data = data.iloc[start_idx:end_idx].to_dict(orient='records')

    return jsonify({'data': paginated_data,'totalPages': total_pages})



# 返回所有特征列的箱线图
@app.route('/api/boxplot', methods=['GET'])
def boxplotoverview():

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
    selected_feature_names = features.columns[selector.get_support()]
    print('selected_feature_names',selected_feature_names)
    features = features[selected_feature_names]
    print(features.head())

    # 设置每个箱子的填充颜色
    box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightsalmon', 'lightpink', 'lightseagreen',
                  'lightsteelblue', 'lightcyan', 'lightgoldenrodyellow']

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
    plt.savefig('static/boxplot.png')
    return jsonify({'image_path': 'static/boxplot.png'})


# 返回指定特征列的箱线图
@app.route('/api/boxplot_image/<feature>', methods=['GET'])
def get_boxplot_image(feature):
    img_base64 = generate_boxplot(feature)
    return send_file(BytesIO(base64.b64decode(img_base64)),
                     mimetype='image/png')

# Function to generate boxplot image for a specific feature
def generate_boxplot(feature):
    data = pd.read_csv('resource/consume.csv')
    plt.figure(figsize=(8, 6))

    # Create a boxplot and get the box artist
    bp = plt.boxplot(data[feature], patch_artist=True)

    # Define the colors for the boxes
    box_colors = ['lightblue']

    # Set box colors
    for box, color in zip(bp['boxes'], box_colors):
        box.set(color='black', linewidth=1)  # Set box border color
        box.set(facecolor=color)  # Set box fill color

    plt.xlabel(feature)
    plt.ylabel('Value')
    plt.title(f'Boxplot for {feature}')

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Encode the image to base64
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    return img_base64


# 返回指定特征列的score
@app.route('/api/featureScore', methods=['GET'])
def featureScore():
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
    plt.savefig('static/featureScore.png')
    return jsonify({'image_path': 'static/featureScore.png'})



if __name__ == '__main__':
    app.run(debug=True)
