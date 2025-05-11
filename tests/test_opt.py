"""
测试luna中的hyper_opt模块
python -m pytest tests/test_opt.py -v
"""
import pytest
import pandas as pd
import numpy as np
from luna.models import xgb_reg
from luna.hyper_opt import OptunaTuner

@pytest.fixture
def sample_tuning_data():
    """生成超参数调优测试数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'feature3': np.random.normal(0, 1, 200)
    })
    # 确保生成一维的目标变量
    y = pd.Series(2 * X['feature1'] + X['feature2'] - 0.5 * X['feature3'] + np.random.normal(0, 0.1, 200))
    
    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 确保 y 是一维的
    y_train = pd.Series(y_train)
    y_val = pd.Series(y_val)
    
    return X_train, X_val, y_train, y_val

class TestOptunaTuner:
    def test_init(self):
        """测试调优器初始化"""
        tuner = OptunaTuner()
        assert tuner is not None
        
    def test_optimize(self, sample_tuning_data):
        """测试超参数优化"""
        X_train, X_val, y_train, y_val = sample_tuning_data
        tuner = OptunaTuner()
        
        best_params = tuner.optimize(
            'XGBRegressor',
            X_train.values,
            X_val.values,
            y_train.values,
            y_val.values,
        )
        
        assert isinstance(best_params, dict)
        assert len(best_params) > 0
        
    def test_get_best_model(self, sample_tuning_data):
        """测试获取最优模型"""
        X_train, X_val, y_train, y_val = sample_tuning_data
        tuner = OptunaTuner()
        
        # 先进行优化
        tuner.optimize(
            'XGBRegressor', 
            X_train.values,  # 转换为numpy数组
            X_val.values,    # 转换为numpy数组
            y_train.values,  # 转换为numpy数组
            y_val.values,    # 转换为numpy数组
        )
        
        # 获取最优模型
        best_model=xgb_reg(tuner.best_params)
        assert isinstance(best_model, xgb_reg)  # 修正类型检查
        # 测试模型可以正常预测
        preds = best_model.predict(X_val)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y_val)