"""
测试lunax的hyper_opt模块
python -m pytest tests/test_opt.py -v
"""
import pytest
import pandas as pd
import numpy as np
from lunax.hyper_opt import OptunaTuner
from xgboost import XGBRegressor, XGBClassifier

@pytest.fixture
def sample_regression_data():
    """生成回归测试数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'feature3': np.random.normal(0, 1, 200)
    })
    y = pd.Series(2 * X['feature1'] + X['feature2'] - 0.5 * X['feature3'] + np.random.normal(0, 0.1, 200))
    
    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, X_val, y_train, y_val

@pytest.fixture
def sample_classification_data():
    """生成分类测试数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'feature3': np.random.normal(0, 1, 200)
    })
    # 生成二分类标签
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    
    # 划分训练集和验证集
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    return X_train, X_val, y_train, y_val

class TestOptunaTuner:
    def test_init_default(self):
        """测试默认初始化"""
        tuner = OptunaTuner()
        assert tuner is not None
        assert tuner.n_trials == 50
        assert tuner.metric_name is None
        assert tuner.timeout is None
        
    def test_init_custom(self):
        """测试自定义参数初始化"""
        param_space = {
            'max_depth': ('int', 3, 10),
            'learning_rate': ('float', 0.01, 0.3)
        }
        tuner = OptunaTuner(param_space=param_space, n_trials=10, metric_name='rmse', timeout=60)
        assert tuner.param_space == param_space
        assert tuner.n_trials == 10
        assert tuner.metric_name == 'rmse'
        assert tuner.timeout == 60
        
    def test_regression_optimize(self, sample_regression_data):
        """测试回归模型优化"""
        X_train, X_val, y_train, y_val = sample_regression_data
        model_classes = ['XGBRegressor', 'LGBMRegressor']
        
        for model_class in model_classes:
            tuner = OptunaTuner(n_trials=2, metric_name='rmse', model_class=model_class)
            
            result = tuner.optimize(
                X_train, y_train,
                X_val, y_val
            )
            
            assert isinstance(result, dict)
            assert 'best_params' in result
            assert 'best_value' in result
            assert 'n_trials' in result
            assert 'study' in result
            assert result['n_trials'] == 2
        
    def test_classification_optimize(self, sample_classification_data):
        """测试分类模型优化"""
        X_train, X_val, y_train, y_val = sample_classification_data
        model_classes = ['XGBClassifier', 'LGBMClassifier']
        
        for model_class in model_classes:
            tuner = OptunaTuner(n_trials=2, metric_name='accuracy', model_class=model_class)
            
            result = tuner.optimize(
                X_train, y_train,
                X_val, y_val
            )
            
            assert isinstance(result, dict)
            assert 'best_params' in result
            assert 'best_value' in result
            assert 'n_trials' in result
            assert 'study' in result
            assert result['n_trials'] == 2
        
    def test_custom_param_space(self, sample_regression_data):
        """测试自定义参数空间"""
        X_train, X_val, y_train, y_val = sample_regression_data
        param_space = {
            'max_depth': ('int', 3, 5),
            'learning_rate': ('float', 0.1, 0.2),
            'booster': ('categorical', ['gbtree'])
        }
        tuner = OptunaTuner(param_space=param_space, n_trials=2,model_class='XGBRegressor')
        
        result = tuner.optimize(
            X_train, y_train,
            X_val, y_val
        )
        
        assert isinstance(result['best_params'], dict)
        assert all(k in result['best_params'] for k in param_space.keys())
        
    def test_invalid_model_type(self, sample_regression_data):
        """测试无效的模型类型"""
        X_train, X_val, y_train, y_val = sample_regression_data
        tuner = OptunaTuner(n_trials=2,model_class='InvalidModel')
        
        with pytest.raises(ValueError):
            tuner.optimize(
                X_train, y_train,
                X_val, y_val
            )