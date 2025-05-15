from .base_model import BaseModel
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from typing import Union, Optional, List, Dict
import pandas as pd
import numpy as np
from tabulate import tabulate

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
        print("[lunax]> model training finished.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
        
        # 打印目标值的范围信息
        print("[lunax]> target value description:")
        stats_table = [["min", "max", "mean", "std", "median"],
                      [f"{y.min():.2f}", f"{y.max():.2f}", f"{y.mean():.2f}", 
                       f"{y.std():.2f}", f"{y.median():.2f}"]]
        print(tabulate(stats_table, headers="firstrow", tablefmt="grid"))
       
        preds = self.predict(X)
        rmse = root_mean_squared_error(y, preds)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        
        # 打印评估结果
        print("[lunax]> model evaluation results:")
        metrics_table = [["metrics", "rmse", "mse", "mae", "r2"],
                        ["values", f"{rmse:.2f}", f"{mse:.2f}", f"{mae:.2f}", f"{r2:.2f}"]]
        print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
        
        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2
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
        
        # 打印标签信息
        print("[lunax]> label information:")
        class_dist_table = [["label", "count"]]
        for label, count in y.value_counts().items():
            class_dist_table.append([label, count])
        print(tabulate(class_dist_table, headers="firstrow", tablefmt="grid"))
        
        # 计算评估指标
        preds = self.predict(X)
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, average='weighted')
        recall = recall_score(y, preds, average='weighted')
        f1 = f1_score(y, preds, average='weighted')
        
        # 打印评估结果
        print("[lunax]> model evaluation results:")
        metrics_table = [["metrics", "accuracy", "precision", "recall", "f1"],
                        ["values", f"{accuracy:.2f}", f"{precision:.2f}", 
                         f"{recall:.2f}", f"{f1:.2f}"]]
        print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


class lgbm_reg(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化LightGBM回归模型。
        
        参数：
            params: 可选，传入LightGBM模型的超参数字典
        """
        self.model = LGBMRegressor(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        print("[lunax]> model training finished.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error
        
        # 打印目标值的范围信息
        print("[lunax]> target value description:")
        stats_table = [["min", "max", "mean", "std", "median"],
                      [f"{y.min():.2f}", f"{y.max():.2f}", f"{y.mean():.2f}", 
                       f"{y.std():.2f}", f"{y.median():.2f}"]]
        print(tabulate(stats_table, headers="firstrow", tablefmt="grid"))
       
        preds = self.predict(X)
        rmse = root_mean_squared_error(y, preds)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        
        # 打印评估结果
        print("[lunax]> model evaluation results:")
        metrics_table = [["metrics", "rmse", "mse", "mae", "r2"],
                        ["values", f"{rmse:.2f}", f"{mse:.2f}", f"{mae:.2f}", f"{r2:.2f}"]]
        print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
        
        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }


class lgbm_clf(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化LightGBM分类模型。
        
        参数：
            params: 可选，传入LightGBM模型的超参数字典
        """
        self.model = LGBMClassifier(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
        print("[lunax]> model training finished.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型性能"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # 打印标签信息
        print("[lunax]> label information:")
        class_dist_table = [["label", "count"]]
        for label, count in y.value_counts().items():
            class_dist_table.append([label, count])
        print(tabulate(class_dist_table, headers="firstrow", tablefmt="grid"))
        
        # 计算评估指标
        preds = self.predict(X)
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, average='weighted')
        recall = recall_score(y, preds, average='weighted')
        f1 = f1_score(y, preds, average='weighted')
        
        # 打印评估结果
        print("[lunax]> model evaluation results:")
        metrics_table = [["metrics", "accuracy", "precision", "recall", "f1"],
                        ["values", f"{accuracy:.2f}", f"{precision:.2f}", 
                         f"{recall:.2f}", f"{f1:.2f}"]]
        print(tabulate(metrics_table, headers="firstrow", tablefmt="grid"))
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

