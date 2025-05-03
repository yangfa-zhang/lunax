from .base_model import BaseModel
from xgboost import XGBRegressor
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
        preds = self.predict(X)
        return {
            "rmse": mean_squared_error(y, preds, squared=False),
            "r2": r2_score(y, preds)
        }