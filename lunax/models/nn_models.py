from .base_model import BaseModel
from .utils import _cross_validate, reg_evaluate, clf_evaluate
from tabpfn import TabPFNClassifier, TabPFNRegressor
from typing import Union, Optional, List, Dict
import pandas as pd
import numpy as np


class pfn_reg(BaseModel):
    def __init__(self):
        self.model = TabPFNRegressor(ignore_pretraining_limits=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return reg_evaluate(y_true=y, y_pred=y_pred)


class pfn_clf(BaseModel):
    def __init__(self):
        self.model = TabPFNClassifier(ignore_pretraining_limits=True)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        y_pred = self.predict(X)
        return clf_evaluate(y_true=y, y_pred=y_pred)

