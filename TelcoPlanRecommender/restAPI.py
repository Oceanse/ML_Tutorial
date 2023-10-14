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
    features = [data['age'], data['call_time'], data['traffic'], data['fee']]
    # 一维数组转换为一个矩阵(二维数组)
    features = np.array(features).reshape(1, -1)
    prediction = loaded_model.predict(features)
    print("Prediction:", prediction)
    class_mapping = {0: 'A', 1: 'B', 2: 'C'}
    predicted_label = class_mapping[prediction[0]]
    print(predicted_label)
    # return predicted_label
    return jsonify({'predicted_label': predicted_label})

# 返回所有特征列的箱线图
@app.route('/api/boxplot', methods=['GET'])
def boxplotoverview():
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

    # 设置每个箱子的填充颜色
    box_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    # 通过箱线图找到离异点
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.figure(figsize=(10, 6))  # 绘制箱线图
    plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    df = pd.DataFrame(data)
    bp = df.boxplot(column=selected_feature_names.tolist(), boxprops=boxprops, medianprops=medianprops,patch_artist=True)
    # 遍历子元素找到箱子并设置填充颜色
    for box, c in zip(bp.findobj(match=plt.matplotlib.patches.PathPatch), box_colors):
        box.set_facecolor(c)  # 设置箱子的颜色

    plt.title('Boxplot of Features')
    plt.ylabel('Values')
    # Save the plot as an image
    plt.savefig('static/boxplot.png')
    return jsonify({'image_path': 'static/boxplot.png'})


# Endpoint to serve the images
@app.route('/api/boxplot_image/<feature>', methods=['GET'])
def get_boxplot_image(feature):
    img_base64 = generate_boxplot(feature)
    return send_file(BytesIO(base64.b64decode(img_base64)),
                     mimetype='image/png')


# 获取数据集
@app.route('/api/get_data', methods=['GET'])
def get_data():
    data = pd.read_csv('resource/teleCom.csv')
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




# Function to generate boxplot image for a specific feature
def generate_boxplot(feature):
    data = pd.read_csv('resource/teleCom.csv')
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



# Function to generate boxplot image for a specific feature
def generate_boxplotback(feature):
    data = pd.read_csv('resource/teleCom.csv')
    plt.figure(figsize=(8, 6))
    plt.boxplot(data[feature])
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



# 返回所有特征列的箱线图
@app.route('/api/boxplotback', methods=['GET'])
def boxplotoverviewback():
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
    boxprops = dict(linestyle='--', linewidth=2, color='blue')  # 箱线的样式
    medianprops = dict(linestyle='-', linewidth=2, color='red')  # 中位线的样式
    plt.figure(figsize=(10, 6))  # 绘制箱线图
    plt.gca().set_facecolor('lightgray')  # 设置背景颜色
    df = pd.DataFrame(data)
    df.boxplot(column=selected_feature_names.tolist(), boxprops=boxprops, medianprops=medianprops)
    plt.title('Boxplot of Features')
    plt.ylabel('Values')
    # Save the plot as an image
    plt.savefig('static/boxplot.png')
    return jsonify({'image_path': 'static/boxplot.png'})

if __name__ == '__main__':
    app.run(debug=True)
