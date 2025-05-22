from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error, r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from ..models.base_model import BaseModel

class HillClimbingEnsemble:
    def __init__(
        self,
        models: List[BaseModel],
        metric: Union[str, List[str]] = "auc",
        maximize: bool = True,
        max_iter: int = 100,
        step_size: float = 0.1,
        tolerance: float = 1e-5,
        n_random_starts: int = 5,
        random_state: Optional[int] = None
    ):
        """
        初始化模型集成优化器
        
        Args:
            models: BaseModel子类实例列表，这些模型需要已经训练好
            metric: 评估指标
            maximize: 是否最大化评估指标
            max_iter: 最大迭代次数
            step_size: 权重调整步长
            tolerance: 收敛阈值
            n_random_starts: 随机初始化次数
            random_state: 随机种子
        """
        self.models = models
        self.n_models = len(models)
        self.metric = metric if isinstance(metric, list) else [metric]
        self.maximize = maximize
        self.max_iter = max_iter
        self.step_size = step_size
        self.tolerance = tolerance
        self.n_random_starts = n_random_starts
        
        if random_state is not None:
            np.random.seed(random_state)
            
        self.best_weights = None
        self.best_score = float('-inf') if maximize else float('inf')
        self.history = []
        
    def _get_ensemble_predictions(
        self, 
        X: pd.DataFrame, 
        weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """获取加权集成预测
        
        Returns:
            tuple: (预测概率, 预测标签)
        """
        predictions = []
        for model, weight in zip(self.models, weights):
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            predictions.append(pred * weight)
        
        proba = np.sum(predictions, axis=0)
        labels = np.argmax(proba, axis=1)
        return proba, labels
    
    def _evaluate_weights(
        self,
        weights: np.ndarray,
        X: pd.DataFrame,
        y: pd.Series
    ) -> float:
        """评估当前权重组合的性能"""
        ensemble_probs,ensemble_preds = self._get_ensemble_predictions(X, weights)
        
        accuracy = accuracy_score(y, ensemble_preds)
        precision = precision_score(y, ensemble_preds, average='weighted', zero_division=0)
        recall = recall_score(y, ensemble_preds, average='weighted')
        f1 = f1_score(y, ensemble_preds, average='weighted')
        # 对于二分类问题
        if ensemble_probs.shape[1] == 2:
            auc = roc_auc_score(y, ensemble_probs[:, 1])
        # 对于多分类问题
        else:
            auc = roc_auc_score(y, ensemble_probs, multi_class='ovr', average='weighted')
        scores = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        final_score = np.mean([scores[m] for m in self.metric])
        return final_score if self.maximize else -final_score
    
    def _get_neighbors(self, weights: np.ndarray) -> List[np.ndarray]:
        """获取当前权重的邻域解"""
        neighbors = []
        for i in range(self.n_models):
            for step in [-self.step_size, self.step_size]:
                new_weights = weights.copy()
                new_weights[i] += step
                # 确保权重非负且和为1
                if new_weights[i] >= 0:
                    new_weights = new_weights / new_weights.sum()
                    neighbors.append(new_weights)
        return neighbors
    
    def fit(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> np.ndarray:
        """
        优化集成权重
        
        Args:
            X_val: 验证集特征
            y_val: 验证集标签
            
        Returns:
            最优权重数组
        """
        for _ in range(self.n_random_starts):
            # 随机初始化权重
            current_weights = np.random.random(self.n_models)
            current_weights = current_weights / current_weights.sum()
            
            for _ in range(self.max_iter):
                current_score = self._evaluate_weights(
                    current_weights, X_val, y_val
                )
                self.history.append((current_weights.copy(), current_score))
                
                # 更新最优解
                if (self.maximize and current_score > self.best_score) or \
                   (not self.maximize and current_score < self.best_score):
                    self.best_score = current_score
                    self.best_weights = current_weights.copy()
                
                # 搜索邻域
                neighbors = self._get_neighbors(current_weights)
                best_neighbor = None
                best_neighbor_score = float('-inf') if self.maximize else float('inf')
                
                for neighbor in neighbors:
                    score = self._evaluate_weights(neighbor, X_val, y_val)
                    if (self.maximize and score > best_neighbor_score) or \
                       (not self.maximize and score < best_neighbor_score):
                        best_neighbor = neighbor
                        best_neighbor_score = score
                
                # 判断是否收敛
                if abs(best_neighbor_score - current_score) < self.tolerance:
                    break
                    
                current_weights = best_neighbor
        
        return self.best_weights
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用最优权重进行预测标签"""
        if self.best_weights is None:
            raise ValueError("模型还未训练，请先调用fit方法")
        _, labels = self._get_ensemble_predictions(X, self.best_weights)
        return labels
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """使用最优权重进行预测概率"""
        if self.best_weights is None:
            raise ValueError("模型还未训练，请先调用fit方法")
        proba, _ = self._get_ensemble_predictions(X, self.best_weights)
        return proba