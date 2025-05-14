from .base_tuner import BaseTuner
from ..models.base_model import BaseModel
from typing import Tuple, Dict, Literal, Optional, Type, Callable
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor,XGBClassifier
from lightgbm import LGBMRegressor,LGBMClassifier
from catboost import CatBoostRegressor,CatBoostClassifier

class OptunaTuner(BaseTuner):
    def __init__(self, 
                 param_space: Dict[str, Tuple]=None,
                 n_trials: int = 50, 
                 metric_name: str = None, 
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
            metric_name: 评估指标名称
            timeout: 可选，超时时间（秒）
        """
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric_name = metric_name
        self.timeout = timeout
        self.study = None
        self.best_params = None
        self.best_value = None
        
    def _objective(self, trial, model_class:str, X_train, y_train, X_val, y_val)->float:
        """
        优化目标函数
        """
        # 根据参数空间定义生成试验参数
        params = {}
        if self.param_space is None:
            # 根据模型类名定义参数搜索空间
            if model_class in ['XGBRegressor', 'XGBClassifier']:
                # XGBoost 参数说明
                print(f"[lunax]> XGBoost Parameter Explanations:")
                print("[Model complexity parameters]>")
                print("- lambda: \tL2 regularization. Smoother than L1. Better for sparse data. Prevents overfitting.")
                print("- reg_lambda/alpha: \tRegularization. Control model complexity. Prevents overfitting.")
                print("- gamma: \tTREE ONLY. Minimum loss reduction for split. Prevents overfitting.")
                print("- max_depth: \tHigher = more complex model. Prevents overfitting.")
                print("- subsample: \tNumber of samples per tree. Prevents overfitting.")
                print("- colsample_bytree: \tFraction of features used per tree. Prevents overfitting.")
                print("- min_child_weight: \tMinimum sum of instance weight in a child. Prevents overfitting.")
                print("\n")
                print("[Training and Optimization Parameters]>")
                print("- eta: \tLearning rate.")
                print("- booster: \t\"gbtree\" for nonlinear features. \"gblinear\" for linear features")
                print("- grow_policy: \tControls how new nodes are added to the tree. \"lossguide\" for best split. \"depthwise\" for best depth.")
                print("\n")
                
                # 首先选择booster类型
                booster = trial.suggest_categorical('booster', ['gbtree', 'gblinear'])
                
                # 基础参数
                params = {
                    'booster': booster,
                    'seed': trial.suggest_int('seed', 0, 0), # 随机种子
                    'eta': trial.suggest_float('eta', 0.01, 0.2), # 学习率
                    'lambda': trial.suggest_float('lambda', 0.0, 100), # L2正则化参数，权重永远不会等于零，比 Alpha 更平滑
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1), 
                    'reg_alpha': trial.suggest_int('reg_alpha', 40, 180),
                }
                
                # 如果是gbtree，添加树相关的参数
                if booster == 'gbtree':
                    params.update({
                        'gamma': trial.suggest_float('gamma', 1, 9),
                        'subsample': trial.suggest_float('subsample', 0.6, 1),
                        'max_depth': trial.suggest_int('max_depth', 3, 18),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                        'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
                        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                    })
            elif model_class in ['LGBMRegressor', 'LGBMClassifier']:
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                    'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                    'subsample': trial.suggest_float('subsample', 0.5, 1),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'objective': trial.suggest_categorical('objective', ['regression', 'binary', 'multiclass'])
                }
        else:
            # 如果提供了自定义参数空间
            for param_name, (param_type, *args) in self.param_space.items():
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(param_name, args[0], args[1])
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(param_name, args[0], args[1])
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, args[0])
        
        # 区分分类模型和回归模型
        if model_class in ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']:
            # 训练回归模型
            if model_class == 'XGBRegressor':
                model = XGBRegressor(**params)
            elif model_class == 'LGBMRegressor':
                model = LGBMRegressor(**params)
            elif model_class == 'CatBoostRegressor':
                model = CatBoostRegressor(**params)

            model.fit(X_train, y_train)
            
            # 评估回归模型
            y_pred = model.predict(X_val)
            
            # 根据指标名称计算相应的评估指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
            
            metrics = {
                'mse': mean_squared_error(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'r2': r2_score(y_val, y_pred),
                'rmse': root_mean_squared_error(y_val, y_pred)
            }
            if self.metric_name is None:
                self.metric_name = 'mse'
            if self.metric_name not in metrics.keys():
                raise ValueError(f"Unsupported metric: {self.metric_name}, supported metrics: {metrics.keys()}")
            
            return metrics[self.metric_name]
        elif model_class in ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
            # 训练分类模型
            if model_class == 'XGBClassifier':
                model = XGBClassifier(**params)
            elif model_class == 'LGBMClassifier':
                model = LGBMClassifier(**params)
            elif model_class == 'CatBoostClassifier':
                model = CatBoostClassifier(**params)

            model.fit(X_train, y_train)
            
            # 评估分类模型
            y_pred = model.predict(X_val)
            
            # 根据指标名称计算相应的评估指标
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred)
            }
            if self.metric_name is None:
                self.metric_name = 'f1'
            if self.metric_name not in metrics.keys():
                raise ValueError(f"Unsupported metric: {self.metric_name}, supported metrics: {metrics.keys()}")
            
            return metrics[self.metric_name]
        else:
            raise ValueError(f"Unsupported model type: {model_class.__name__}")

    def optimize(self, 
                model_class: str, 
                X_train: pd.DataFrame, 
                y_train: pd.Series,
                X_val: pd.DataFrame, 
                y_val: pd.Series) -> Dict:
        """
        实现 Optuna 的超参数搜索逻辑。

        参数：
            model_class: 模型类。可以包括 XGBRegressor,XGBClassifier,LGBMRegressor,LGBMClassifier,CatBoostRegressor,CatBoostClassifier
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签

        返回：
            Dict: 包含最优参数和对应的评估指标值
        """
        # 根据模型类型确定优化方向
        if model_class in ['XGBRegressor', 'LGBMRegressor', 'CatBoostRegressor']:
            direction = "minimize"
        elif model_class in ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']:
            direction = "maximize"
        else:
            raise ValueError(f"Unsupported model type: {model_class}")
        # 创建study对象
        self.study = optuna.create_study(direction=direction)
        
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