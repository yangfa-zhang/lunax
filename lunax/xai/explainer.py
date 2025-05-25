import shap
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict
import matplotlib.pyplot as plt
from tabulate import tabulate
from ..models.tree_models import xgb_reg, xgb_clf, lgbm_reg, lgbm_clf, cat_reg, cat_clf


class TreeExplainer:
    def __init__(self, model: Union[xgb_reg, xgb_clf, lgbm_reg, lgbm_clf, cat_reg, cat_clf]):
        """
        初始化树模型解释器

        参数:
            model: 已训练的树模型实例
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model.model)
        
    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        计算SHAP值

        参数:
            X: 需要解释的特征数据

        返回:
            SHAP值数组
        """
        return self.explainer.shap_values(X)

    def plot_summary(self, X: pd.DataFrame, max_display: int = 20) -> None:
        """
        绘制SHAP值汇总图

        参数:
            X: 需要解释的特征数据
            max_display: 显示的最大特征数量
        """
        print('[lunax]> Clear blue/red separation indicates a highly influential feature.')
        shap_values = self.get_shap_values(X)
        if isinstance(self.model, (xgb_clf, lgbm_clf, cat_clf)):
            # 对于分类模型，我们使用第一个类的SHAP值
            shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                            X, max_display=max_display)
        else:
            shap.summary_plot(shap_values, X, max_display=max_display)

    def plot_dependence(self, X: pd.DataFrame, feature: str, interaction_index: Optional[str] = None) -> None:
        """
        绘制SHAP依赖图

        参数:
            X: 需要解释的特征数据
            feature: 要分析的特征名
            interaction_index: 交互特征名（可选）
        """
        shap_values = self.get_shap_values(X)
        if isinstance(self.model, (xgb_clf, lgbm_clf, cat_clf)):
            shap.dependence_plot(feature, shap_values[1] if isinstance(shap_values, list) else shap_values, 
                               X, interaction_index=interaction_index)
        else:
            shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_index)

    def plot_force(self, X: pd.DataFrame, index: int = 0) -> None:
        """
        绘制单个预测的力图

        参数:
            X: 需要解释的特征数据
            index: 要解释的样本索引
        """
        shap_values = self.get_shap_values(X)
        if isinstance(self.model, (xgb_clf, lgbm_clf, cat_clf)):
            shap.force_plot(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) 
                          else self.explainer.expected_value,
                          shap_values[1][index] if isinstance(shap_values, list) else shap_values[index],
                          X.iloc[index], matplotlib=True)
        else:
            shap.force_plot(self.explainer.expected_value,
                          shap_values[index],
                          X.iloc[index], matplotlib=True)

    def get_feature_importance(self, X: pd.DataFrame, print_table: bool = True) -> pd.Series:
        """
        获取基于SHAP值的特征重要性

        参数:
            X: 需要解释的特征数据
            print_table: 是否打印格式化表格，默认为True

        返回:
            按重要性排序的特征Series
        """
        shap_values = self.get_shap_values(X)
        if isinstance(self.model, (xgb_clf, lgbm_clf, cat_clf)):
            shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        feature_importance = pd.Series(np.abs(shap_values).mean(axis=0), 
                                     index=X.columns).sort_values(ascending=False)
        
        if print_table:
            # 创建一个包含排名的DataFrame
            df_importance = pd.DataFrame({
                'Feature': feature_importance.index,
                'Importance': feature_importance.values
            }).reset_index(drop=True)
            df_importance.index = df_importance.index + 1  # 从1开始的排名
            
            # 使用tabulate打印格式化表格
            print("[lunax]> Feature Importance Ranking:")
            print(tabulate(df_importance, 
                         headers='keys',
                         tablefmt='pretty',
                         floatfmt='.4f'))
            
        return feature_importance