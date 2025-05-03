import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Literal

def load_data(file_path: str, file_type: Literal['str','parquet']) -> pd.DataFrame:
    """
    从表格数据文件中加载数据为 DataFrame。
    
    参数：
        file_path: str - 数据文件路径
        
    返回：
        pd.DataFrame - 加载后的数据
    """
    if file_type not in ['str','parquet']:
        raise ValueError("file_type must be 'str' or 'parquet'")
    if file_type == 'parquet':
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)

def split_data(
    df: pd.DataFrame, 
    target: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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