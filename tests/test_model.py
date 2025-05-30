"""
测试lunax中的models模块
python -m pytest tests/test_model.py -v
"""
import pytest
import pandas as pd
import numpy as np
from lunax.models import xgb_reg, xgb_clf, lgbm_reg, lgbm_clf, cat_reg, cat_clf, pfn_reg, pfn_clf

@pytest.fixture
def sample_regression_data():
    """生成回归测试数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    y = pd.Series(X['feature1'] * 2 + X['feature2'] + np.random.normal(0, 0.1, 100))
    return X, y

@pytest.fixture
def sample_classification_data():
    """生成分类测试数据"""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })
    y = pd.Series((X['feature1'] + X['feature2'] > 0).astype(int))
    return X, y

class TestXGBRegressor:
    def test_init(self):
        """测试回归模型初始化"""
        model = xgb_reg()
        assert model is not None
        
    def test_fit_predict(self, sample_regression_data):
        """测试回归模型训练和预测"""
        X, y = sample_regression_data
        model = xgb_reg()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_evaluate(self, sample_regression_data):
        """测试回归模型评估"""
        X, y = sample_regression_data
        model = xgb_reg()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['rmse', 'mse', 'mae', 'r2'])
        assert all(isinstance(v, float) for v in metrics.values())

class TestXGBClassifier:
    def test_init(self):
        """测试分类模型初始化"""
        model = xgb_clf()
        assert model is not None
        
    def test_fit_predict(self, sample_classification_data):
        """测试分类模型训练和预测"""
        X, y = sample_classification_data
        model = xgb_clf()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_predict_proba(self, sample_classification_data):
        """测试分类模型概率预测"""
        X, y = sample_classification_data
        model = xgb_clf()
        model.fit(X, y)
        
        probs = model.predict_proba(X)
        assert isinstance(probs, np.ndarray)
        assert probs.shape[0] == len(y)
        assert probs.shape[1] == len(np.unique(y))
        
    def test_evaluate(self, sample_classification_data):
        """测试分类模型评估"""
        X, y = sample_classification_data
        model = xgb_clf()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
        assert all(isinstance(v, float) for v in metrics.values())

class TestLGBMRegressor:
    def test_init(self):
        """测试回归模型初始化"""
        model = lgbm_reg()
        assert model is not None
        
    def test_fit_predict(self, sample_regression_data):
        """测试回归模型训练和预测"""
        X, y = sample_regression_data
        model = lgbm_reg()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_evaluate(self, sample_regression_data):
        """测试回归模型评估"""
        X, y = sample_regression_data
        model = lgbm_reg()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['rmse', 'mse', 'mae', 'r2'])
        assert all(isinstance(v, float) for v in metrics.values())

    def test_kfold_fit(self, sample_regression_data):
        """测试k折交叉验证训练"""
        X, y = sample_regression_data
        model = lgbm_reg()
        model.fit(X, y, k_fold=3)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)

class TestLGBMClassifier:
    def test_init(self):
        """测试分类模型初始化"""
        model = lgbm_clf()
        assert model is not None
        
    def test_fit_predict(self, sample_classification_data):
        """测试分类模型训练和预测"""
        X, y = sample_classification_data
        model = lgbm_clf()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_predict_proba(self, sample_classification_data):
        """测试分类模型概率预测"""
        X, y = sample_classification_data
        model = lgbm_clf()
        model.fit(X, y)
        
        probs = model.predict_proba(X)
        assert isinstance(probs, np.ndarray)
        assert probs.shape[0] == len(y)
        assert probs.shape[1] == len(np.unique(y))
        
    def test_evaluate(self, sample_classification_data):
        """测试分类模型评估"""
        X, y = sample_classification_data
        model = lgbm_clf()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
        assert all(isinstance(v, float) for v in metrics.values())

    def test_kfold_fit(self, sample_classification_data):
        """测试k折交叉验证训练"""
        X, y = sample_classification_data
        model = lgbm_clf()
        model.fit(X, y, k_fold=3)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)


# 在文件末尾添加 CatBoost 测试类
class TestCatBoostRegressor:
    def test_init(self):
        """测试回归模型初始化"""
        model = cat_reg()
        assert model is not None
        
    def test_fit_predict(self, sample_regression_data):
        """测试回归模型训练和预测"""
        X, y = sample_regression_data
        model = cat_reg()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_evaluate(self, sample_regression_data):
        """测试回归模型评估"""
        X, y = sample_regression_data
        model = cat_reg()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['rmse', 'mse', 'mae', 'r2'])
        assert all(isinstance(v, float) for v in metrics.values())

    def test_kfold_fit(self, sample_regression_data):
        """测试k折交叉验证训练"""
        X, y = sample_regression_data
        model = cat_reg()
        model.fit(X, y, k_fold=3)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)

class TestCatBoostClassifier:
    def test_init(self):
        """测试分类模型初始化"""
        model = cat_clf()
        assert model is not None
        
    def test_fit_predict(self, sample_classification_data):
        """测试分类模型训练和预测"""
        X, y = sample_classification_data
        model = cat_clf()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_predict_proba(self, sample_classification_data):
        """测试分类模型概率预测"""
        X, y = sample_classification_data
        model = cat_clf()
        model.fit(X, y)
        
        probs = model.predict_proba(X)
        assert isinstance(probs, np.ndarray)
        assert probs.shape[0] == len(y)
        assert probs.shape[1] == len(np.unique(y))
        
    def test_evaluate(self, sample_classification_data):
        """测试分类模型评估"""
        X, y = sample_classification_data
        model = cat_clf()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
        assert all(isinstance(v, float) for v in metrics.values())

    def test_kfold_fit(self, sample_classification_data):
        """测试k折交叉验证训练"""
        X, y = sample_classification_data
        model = cat_clf()
        model.fit(X, y, k_fold=3)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)


# 在文件末尾添加 TabPFN 测试类
class TestPFNRegressor:
    def test_init(self):
        """测试回归模型初始化"""
        model = pfn_reg()
        assert model is not None
        
    def test_fit_predict(self, sample_regression_data):
        """测试回归模型训练和预测"""
        X, y = sample_regression_data
        model = pfn_reg()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_evaluate(self, sample_regression_data):
        """测试回归模型评估"""
        X, y = sample_regression_data
        model = pfn_reg()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['rmse', 'mse', 'mae', 'r2'])
        assert all(isinstance(v, float) for v in metrics.values())

class TestPFNClassifier:
    def test_init(self):
        """测试分类模型初始化"""
        model = pfn_clf()
        assert model is not None
        
    def test_fit_predict(self, sample_classification_data):
        """测试分类模型训练和预测"""
        X, y = sample_classification_data
        model = pfn_clf()
        model.fit(X, y)
        
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(y)
        
    def test_evaluate(self, sample_classification_data):
        """测试分类模型评估"""
        X, y = sample_classification_data
        model = pfn_clf()
        model.fit(X, y)
        
        metrics = model.evaluate(X, y)
        assert isinstance(metrics, dict)
        assert all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1'])
        assert all(isinstance(v, float) for v in metrics.values())