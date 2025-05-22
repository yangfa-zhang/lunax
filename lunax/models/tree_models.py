from .base_model import BaseModel
from .utils import _cross_validate, reg_evaluate, clf_evaluate
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        preds = self.predict(X)
        return reg_evaluate(y, preds, log_info=log_info)


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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        """评估模型性能"""
        preds = self.predict(X)
        probs = self.predict_proba(X)
        return clf_evaluate(y, preds, probs,log_info=log_info)


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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        preds = self.predict(X)
        return reg_evaluate(y, preds, log_info=log_info)


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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        """评估模型性能"""
        preds = self.predict(X)
        probs = self.predict_proba(X)
        return clf_evaluate(y, preds, probs,log_info=log_info)


class cat_reg(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化CatBoost回归模型。
        
        参数：
            params: 可选，传入CatBoost模型的超参数字典
        """
        self.model = CatBoostRegressor(
            verbose=False,
            **(params or {})
        )

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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        preds = self.predict(X)
        return reg_evaluate(y, preds, log_info=log_info)


class cat_clf(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化CatBoost分类模型。
        
        参数：
            params: 可选，传入CatBoost模型的超参数字典
        """
        self.model = CatBoostClassifier(
            verbose=False,
            **(params or {})
        )

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

    def evaluate(self, X: pd.DataFrame, y: pd.Series, log_info: bool = True) -> Dict[str, float]:
        """评估模型性能"""
        preds = self.predict(X)
        probs = self.predict_proba(X)
        return clf_evaluate(y, preds, probs,log_info=log_info)

