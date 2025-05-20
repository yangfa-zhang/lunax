import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Literal, Optional

def load_data(file_path: str, file_type: Literal['csv','parquet']) -> pd.DataFrame:
    """
    从表格数据文件中加载数据为 DataFrame。
    
    参数：
        file_path: str - 数据文件路径
        
    返回：
        pd.DataFrame - 加载后的数据
    """
    if file_type not in ['csv','parquet']:
        raise ValueError("file_type must be 'csv' or 'parquet'")
    if file_type == 'parquet':
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)

def split_data(
    df: pd.DataFrame, 
    target: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    将数据集按照给定比例划分为 train/val。
    
    参数：
        df: 数据集
        target: 目标列名
        test_size: 验证集比例
        random_state: 随机种子
    
    返回：
        X_train, X_val, y_train, y_val
    """
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_data(
    df: pd.DataFrame,
    target: str = None,
    numeric_strategy: str = "mean",
    category_strategy: str = "most_frequent",
    scale_numeric: bool = True,
    encode_categorical: bool = True
) -> Tuple[pd.DataFrame]:
    """
    对数据进行缺失值处理、编码、标准化等预处理操作。
    
    参数：
        df: 原始数据
        target: 目标列
        numeric_strategy: 数值缺失填充方法 ['mean', 'median']
        category_strategy: 类别缺失填充方法 ['most_frequent']
        scale_numeric: 是否标准化数值特征
        encode_categorical: 是否对类别特征做编码
    
    返回：
        df: 预处理后的数据
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # 复制数据，避免修改原始数据
    df = df.copy()
    
    # 分离特征类型（排除目标列）
    if target is not None:
        feature_cols = df.columns.drop(target)
    numeric_features = df[feature_cols].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns
    
    # 处理数值特征
    if len(numeric_features) > 0:
        # 数值型缺失值填充
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df[numeric_features] = num_imputer.fit_transform(df[numeric_features])
        
        # 数值特征标准化
        if scale_numeric:
            scaler = StandardScaler()
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    # 处理类别特征
    if len(categorical_features) > 0:
        # 类别型缺失值填充
        cat_imputer = SimpleImputer(strategy=category_strategy)
        df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])
        
        # 类别特征编码
        if encode_categorical:
            for col in categorical_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
    
    return df
