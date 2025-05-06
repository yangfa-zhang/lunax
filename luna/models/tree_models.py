from .base_model import BaseModel
from xgboost import XGBRegressor, XGBClassifier
from typing import Union, Optional, List, Dict
import pandas as pd
import numpy as np

class xgb_reg(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化XGBoost回归模型。
        
        参数：
            params: 可选，传入XGBoost模型的超参数字典
        """
        self.model = XGBRegressor(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 打印目标值的范围信息
        print(f"目标值范围: [{y.min():.3f}, {y.max():.3f}]")
        print(f"目标值统计:\n均值: {y.mean():.3f}\n标准差: {y.std():.3f}\n中位数: {y.median():.3f}\n")
        
        preds = self.predict(X)
        return {
            "rmse": mean_squared_error(y, preds, squared=False),
            "r2": r2_score(y, preds)
        }


class xgb_clf(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化XGBoost分类模型。
        
        参数：
            params: 可选，传入XGBoost模型的超参数字典
        """
        self.model = XGBClassifier(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 直接打印标签信息
        unique_labels = np.unique(y)
        print(f"标签取值范围: {unique_labels}, 标签数量: {len(unique_labels)}")
        print(f"各类别样本数量:\n{y.value_counts()}\n")
        
        preds = self.predict(X)
        return {
            "accuracy": accuracy_score(y, preds),
            "precision": precision_score(y, preds, average='weighted'),
            "recall": recall_score(y, preds, average='weighted'),
            "f1": f1_score(y, preds, average='weighted')
        }

