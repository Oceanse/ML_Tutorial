import pandas as pd
def randomSeq():
    # 读取原始CSV文件
    input_file = '/Users/ocean/study/repo/ML_Tutorial/TelcoPlanRecommender/resource/consume.csv'  # 请将文件名替换为你的原始CSV文件名
    output_file = '/Users/ocean/study/repo/ML_Tutorial/TelcoPlanRecommender/resource/consume2.csv'  # 请将文件名替换为你想要输出的文件名

    # 读取CSV文件
    df = pd.read_csv(input_file)
    # 随机打乱样本顺序
    df_shuffled = df.sample(frac=1, random_state=42)  # 使用random_state以确保结果可重复
    # 将随机打乱的结果写入新的CSV文件
    df_shuffled.to_csv(output_file, index=False)



# 随机打乱样本顺序
def resetID():
    data = pd.read_csv('resource/consume.csv')
    df = pd.DataFrame(data)
    # 重置User_ID列为从1开始的递增值
    df['User_ID'] = range(1, len(df) + 1)
    # 将DataFrame写入新文件
    df.to_csv('/Users/ocean/study/repo/ML_Tutorial/TelcoPlanRecommender/resource/consume.csv', index=False)


def deleteColumn():
    data = pd.read_csv('resource/consume.csv')
    df = pd.DataFrame(data)
    # 删除exceeded_plan_fee列
    df.drop(columns=['exceeded_plan_fee'], inplace=True)
    # 将结果保存到新文件
    df.to_csv('/Users/ocean/study/repo/ML_Tutorial/TelcoPlanRecommender/resource/consume2.csv', index=False)


if __name__ == "__main__":
    deleteColumn()
