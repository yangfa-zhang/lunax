from .base_tuner import BaseTuner
from ..models.base_model import BaseModel
from typing import Tuple, Dict, Literal, Optional, Type, Callable
import pandas as pd
import numpy as np
import optuna

class OptunaTuner(BaseTuner):
    def __init__(self, 
                 param_space: Dict[str, Tuple],
                 n_trials: int = 50, 
                 direction: str = "minimize", 
                 metric_name: str = "rmse", 
                 timeout: Optional[int] = None):
        """
        初始化 Optuna 调参器。

        参数：
            param_space: 参数搜索空间，格式为 {'param_name': (param_type, low, high)}
                        param_type 可以是 'int', 'float', 'categorical'
                        例如：{
                            'max_depth': ('int', 3, 10),
                            'learning_rate': ('float', 0.01, 0.3),
                            'booster': ('categorical', ['gbtree', 'gblinear'])
                        }
            n_trials: 试验次数
            direction: 优化方向（"minimize" 或 "maximize"）
            metric_name: 评估指标名称
            timeout: 可选，超时时间（秒）
        """
        self.param_space = param_space
        self.n_trials = n_trials
        self.direction = direction
        self.metric_name = metric_name
        self.timeout = timeout
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def _objective(self, trial, model_class, X_train, y_train, X_val, y_val):
        """优化目标函数"""
        # 根据参数空间定义生成试验参数
        params = {}
        for param_name, (param_type, *args) in self.param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])
        
        # 修改这里：使用 **params 解包参数字典
        
        # 训练模型
        model = model_class(**params)
        model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = model.predict(X_val)
        
        # 根据指标名称计算相应的评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': mean_squared_error(y_val, y_pred, squared=False),
            'mse': mean_squared_error(y_val, y_pred),
            'mae': mean_absolute_error(y_val, y_pred),
            'r2': r2_score(y_val, y_pred)
        }
        
        if self.metric_name not in metrics:
            raise ValueError(f"Unsupported metric: {self.metric_name}")
            
        return metrics[self.metric_name]

    def optimize(self, 
                model_class: Type[BaseModel], 
                X_train: pd.DataFrame, 
                y_train: pd.Series,
                X_val: pd.DataFrame, 
                y_val: pd.Series) -> Dict:
        """
        实现 Optuna 的超参数搜索逻辑。

        参数：
            model_class: 模型类
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        返回：
            Dict: 包含最优参数和对应的评估指标值
        """
        # 创建study对象
        self.study = optuna.create_study(direction=self.direction)
        
        # 定义优化函数
        objective = lambda trial: self._objective(
            trial, model_class, X_train, y_train, X_val, y_val
        )
        
        # 运行优化
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # 保存最优结果
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study  # 返回完整study对象以便进行更详细的分析
        }