from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def numeric_eda(df_list:List[pd.DataFrame],label_list: List[str],target:str)->None:
    """
    可视化数值型特征的分布情况

    参数：
        df_list: 数据集列表
        label_list: 标签列表
        target: 目标变量名称
    """
    assert len(df_list) == len(label_list)
    if len(df_list)==2:
        custom_palette = ["#5A8100", "#FFB400"]
        # 其他配色方案
        """
        ['#B74803','#022E51'] 深棕 深蓝 海底的小丑鱼
        ["#5A8100", "#FFB400"] 鲜绿 鲜黄 森林里的巨嘴鸟
        ["#C7A003", "#3D4E17"] 暗黄 暗青 蓝色火山下的鸢尾花
        """
    elif len(df_list)==3:
        custom_palette = []
        """
        ["#5A8100", "#FFB400", "FF6C02"] 鲜绿 鲜黄 鲜橙 森林里的巨嘴鸟
        ["#C7A003", "#3D4E17", "#151F1E"] 暗黄 暗青 暗蓝 蓝色火山下的鸢尾花
        """
    else:
        raise ValueError("Only support up to 3 datasets.")
    numeric_cols = [col for col in df_list[0].columns if df_list[0][col].dtype in ['int64', 'float64']]
    numeric_cols.remove(target)
    # 添加新一列，用于区分数据集
    for df,label in zip(df_list,label_list):
        df['Dataset'] = label
    for col in numeric_cols:
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 箱型图
        plt.subplot(1, 2, 1)
        sns.boxplot(data=pd.concat(df_list, axis=0, ignore_index=True), 
                   x=col, 
                   hue="Dataset",
                   legend=False,
                   palette=custom_palette)
        plt.xlabel(col)
        plt.title(f"Box Plot for {col}")

        # 直方图
        plt.subplot(1, 2, 2)
        for i,(df,label) in enumerate(zip(df_list,label_list)):
            sns.histplot(data=df, x=col, color=custom_palette[i], kde=True, bins=30, label=label)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram for {col} {label_list}")
        plt.legend()    
    plt.tight_layout()
    plt.show()