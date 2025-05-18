from .base_model import BaseModel
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from typing import Union, Optional, List, Dict
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

# 交叉验证训练
def _cross_validate(model, X: pd.DataFrame, y: pd.Series, k_fold: int, 
                   is_classifier: bool = False) -> None:
    """
    执行k折交叉验证
    
    参数：
        model: 模型实例
        X: 特征数据
        y: 目标变量
        k_fold: 交叉验证折数
        is_classifier: 是否为分类模型
    """
    # 选择合适的交叉验证方式
    if is_classifier:
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
        split_data = kf.split(X, y)
    else:
        kf = KFold(n_splits=k_fold, shuffle=True, random_state=42)
        split_data = kf.split(X)
        
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(split_data, 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        if is_classifier:
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            fold_scores.append({'accuracy': acc, 'f1': f1})
            print(f"[lunax]> Fold {fold}/{k_fold} - Accuracy: {acc:.4f}, F1: {f1:.4f}")
        else:
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            fold_scores.append({'mse': mse, 'r2': r2})
            print(f"[lunax]> Fold {fold}/{k_fold} - MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # 计算平均分数
    if is_classifier:
        avg_acc = np.mean([score['accuracy'] for score in fold_scores])
        avg_f1 = np.mean([score['f1'] for score in fold_scores])
        print(f"[lunax]> Average scores - Accuracy: {avg_acc:.4f}, F1: {avg_f1:.4f}")
    else:
        avg_mse = np.mean([score['mse'] for score in fold_scores])
        avg_r2 = np.mean([score['r2'] for score in fold_scores])
        print(f"[lunax]> Average scores - MSE: {avg_mse:.4f}, R2: {avg_r2:.4f}")

def reg_evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    回归模型评估
    
    参数：
        y_true: 真实值
        y_pred: 预测值
    返回：
        包含评估指标的字典
    """
    # 打印目标值的范围信息
    print("[lunax]> target value description:")
    stats_table = [["min", "max", "mean", "std", "median"],
                  [f"{y_true.min():.2f}", f"{y_true.max():.2f}", f"{y_true.mean():.2f}", 
                   f"{y_true.std():.2f}", f"{y_true.median():.2f}"]]
    print(tabulate(stats_table, headers="firstrow", tablefmt="grid"))
   
    # 计算评估指标
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
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

def clf_evaluate(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    分类模型评估
    
    参数：
        y_true: 真实标签
        y_pred: 预测标签
    返回：
        包含评估指标的字典
    """
    # 打印标签信息
    print("[lunax]> label information:")
    class_dist_table = [["label", "count"]]
    for label, count in y_true.value_counts().items():
        class_dist_table.append([label, count])
    print(tabulate(class_dist_table, headers="firstrow", tablefmt="grid"))
    
    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
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


class xgb_reg(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化XGBoost回归模型。
        
        参数：
            params: 可选，传入XGBoost模型的超参数字典
        """
        self.model = XGBRegressor(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series, k_fold: Optional[int] = None) -> None:
        """
        训练模型，支持 k 折交叉验证

        参数：
            X: 特征数据
            y: 目标变量
            k_fold: 交叉验证折数，默认为 None（不使用交叉验证）
        """
        if k_fold is None:
            print("[lunax]> training model without k-fold cross validation...")
            self.model.fit(X, y)
            print("[lunax]> model training finished.")
            return
        
        _cross_validate(self.model, X, y, k_fold, is_classifier=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        return reg_evaluate(y, preds)


class xgb_clf(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化XGBoost分类模型。
        
        参数：
            params: 可选，传入XGBoost模型的超参数字典
        """
        self.model = XGBClassifier(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series, k_fold: Optional[int] = None) -> None:
        """
        训练模型，支持 k 折交叉验证

        参数：
            X: 特征数据
            y: 目标变量
            k_fold: 交叉验证折数，默认为 None（不使用交叉验证）
        """
        if k_fold is None:
            print("[lunax]> training model without k-fold cross validation...")
            self.model.fit(X, y)
            print("[lunax]> model training finished.")
            return
        
        _cross_validate(self.model, X, y, k_fold, is_classifier=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型性能"""
        preds = self.predict(X)
        return clf_evaluate(y, preds)


class lgbm_reg(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化LightGBM回归模型。
        
        参数：
            params: 可选，传入LightGBM模型的超参数字典
        """
        self.model = LGBMRegressor(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series, k_fold: Optional[int] = None) -> None:
        """
        训练模型，支持 k 折交叉验证

        参数：
            X: 特征数据
            y: 目标变量
            k_fold: 交叉验证折数，默认为 None（不使用交叉验证）
        """
        if k_fold is None:
            print("[lunax]> training model without k-fold cross validation...")
            self.model.fit(X, y)
            print("[lunax]> model training finished.")
            return
        
        _cross_validate(self.model, X, y, k_fold, is_classifier=False)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        preds = self.predict(X)
        return reg_evaluate(y, preds)

class lgbm_clf(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化LightGBM分类模型。
        
        参数：
            params: 可选，传入LightGBM模型的超参数字典
        """
        self.model = LGBMClassifier(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series, k_fold: Optional[int] = None) -> None:
        """
        训练模型，支持 k 折交叉验证

        参数：
            X: 特征数据
            y: 目标变量
            k_fold: 交叉验证折数，默认为 None（不使用交叉验证）
        """
        if k_fold is None:
            print("[lunax]> training model without k-fold cross validation...")
            self.model.fit(X, y)
            print("[lunax]> model training finished.")
            return
        
        _cross_validate(self.model, X, y, k_fold, is_classifier=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """返回预测概率"""
        return self.model.predict_proba(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """评估模型性能"""
        preds = self.predict(X)
        return clf_evaluate(y, preds)

