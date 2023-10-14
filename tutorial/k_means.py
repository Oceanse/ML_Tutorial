import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 模拟啤酒品牌数据
def generate_beer_data(num_samples):
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH', 'BrandI', 'BrandJ']

    beer_data = {
        'Brand': np.random.choice(brands, num_samples),#品牌
        'Alcohol Content': np.random.randint(4, 10, num_samples),#酒精度
        'Bitterness': np.random.randint(1, 10, num_samples), #苦味
        'Maltiness': np.random.randint(2, 8, num_samples), #麦芽糖
        'Mouthfeel': np.random.randint(1, 7, num_samples) # 口感
    }

    return pd.DataFrame(beer_data)


def kmeans():
    # 生成包含50条数据的啤酒数据集
    beer_data = generate_beer_data(50)
    # 提取需要聚类的特征
    features = beer_data[['Alcohol Content', 'Bitterness', 'Maltiness', 'Mouthfeel']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # 使用K-means聚类，假设要分成5类
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_scaled)
    beer_data['Cluster'] = kmeans.labels_  # 将聚类结果添加到数据中

    # 可视化聚类结果
    # 创建一个宽度为10英寸、高度为6英寸的画布
    plt.figure(figsize=(10, 6))

    # 假设 x 和 y 是你要展示的两个特征
    x_feature = 'Alcohol Content'
    y_feature = 'Bitterness'

    # 根据不同聚类显示不同颜色的点
    for i in range(num_clusters):
        cluster_data = beer_data[beer_data['Cluster'] == i]
        plt.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {i}')

    plt.scatter(kmeans.cluster_centers_[:, features.columns.get_loc(x_feature)],
                kmeans.cluster_centers_[:, features.columns.get_loc(y_feature)],
                s=300, c='red', marker='x', label='Centroids')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title('K-means Clustering of Beers')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 生成包含50条数据的啤酒数据集
    # beer_df = generate_beer_data(50)
    # 保存为CSV文件
    # beer_df.to_csv('resource/beer_data.csv', index=False)
    # print("啤酒数据集已生成并保存为 beer_data.csv")
    kmeans()