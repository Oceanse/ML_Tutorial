from sklearn.datasets import load_iris

def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述：\n", iris["DESCR"])           # 数据集的描述信息
    print("查看特征值的名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)  # shape:(150,4)
    return None

if __name__ == "__main__":
    datasets_demo()