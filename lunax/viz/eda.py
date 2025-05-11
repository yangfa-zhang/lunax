from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def numeric_eda(df_list:List[pd.DataFrame],label_list: List[str],target:str, custom_palette=None)->None:
    """
    可视化数值型特征的分布情况

    参数：
        df_list: 数据集列表
        label_list: 标签列表
        target: 目标变量名称
        custom_palette: 自定义颜色列表

    返回：
        None
    """
    assert len(df_list) == len(label_list)
    if custom_palette is None:
        if len(df_list)==2:
            custom_palette = ["#5A8100", "#FFB400"]
            # 其他配色方案
            """
            ['#B74803','#022E51'] 深棕 深蓝 海底的小丑鱼
            ["#5A8100", "#FFB400"] 鲜绿 鲜黄 森林里的巨嘴鸟
            ["#C7A003", "#3D4E17"] 暗黄 暗青 蓝色火山下的鸢尾花
            ["#FCA3B9","#FCD752"] 浅粉 浅黄 衣服搭配
            ["#285185","#D67940"] 深蓝 橙色 衣服搭配
            """
        elif len(df_list)==3:
            custom_palette = ["#5A8100", "#FFB400", "#FF6C02"]
            """
            ["#5A8100", "#FFB400", "#FF6C02"] 鲜绿 鲜黄 鲜橙 森林里的巨嘴鸟
            ["#C7A003", "#3D4E17", "#151F1E"] 暗黄 暗青 暗蓝 蓝色火山下的鸢尾花
            """
        else:
            raise ValueError("Only support up to 3 datasets.")
    numeric_cols = [col for col in df_list[0].columns if df_list[0][col].dtype in ['int64', 'float64']]
    if target in numeric_cols:
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


def categoric_eda(df_list:List[pd.DataFrame],label_list: List[str],target:str, custom_palette=None)->None:
    """
    可视化数值型特征的分布情况

    参数：
        df_list: 数据集列表
        label_list: 标签列表
        target: 目标变量名称
        custom_palette: 自定义颜色列表

    返回：
        None
    """
    assert len(df_list) == len(label_list)
    if custom_palette is None:
        if len(df_list)==2:
            custom_palette = ["#5A8100", "#FFB400"]
            # 其他配色方案
            """
            ['#B74803','#022E51'] 深棕 深蓝 海底的小丑鱼
            ["#5A8100", "#FFB400"] 鲜绿 鲜黄 森林里的巨嘴鸟
            ["#C7A003", "#3D4E17"] 暗黄 暗青 蓝色火山下的鸢尾花
            ["#FCA3B9","#FCD752"] 浅粉 浅黄 衣服搭配
            ["#285185","#D67940"] 深蓝 橙色 衣服搭配
            """
        elif len(df_list)==3:
            custom_palette = ["#5A8100", "#FFB400", "#FF6C02"]
            """
            ["#5A8100", "#FFB400", "#FF6C02"] 鲜绿 鲜黄 鲜橙 森林里的巨嘴鸟
            ["#C7A003", "#3D4E17", "#151F1E"] 暗黄 暗青 暗蓝 蓝色火山下的鸢尾花
            """
        else:
            raise ValueError("Only support up to 3 datasets.")
    categoric_cols = [col for col in df_list[0].columns if df_list[0][col].dtype in ['object']]
    if target in categoric_cols:
        categoric_cols.remove(target)            
    for col in categoric_cols:
        sns.set_style('whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 饼图
        plt.subplot(1, 2, 1)
        df_list[0][col].value_counts().plot.pie(autopct='%1.1f%%', colors=custom_palette,wedgeprops=dict(width=0.3), startangle=140)
        plt.title(f"Pie Chart for {col}")

        # 柱状图
        plt.subplot(1, 2, 2)
        sns.countplot(data=pd.concat(df_list, axis=0, ignore_index=True),
                   x=col,
                   color="#FF6C02",
                   alpha=0.8
                   )
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Bar Chart for {col}")
    plt.tight_layout()
    plt.show()