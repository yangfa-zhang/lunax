from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Literal, Optional

class BaseTuner(ABC):
    """
    所有调参器的基础接口，定义搜索方法。
    """

    @abstractmethod
    def optimize(self, 
                 model_class: Type[BaseModel], 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 X_val: pd.DataFrame, 
                 y_val: pd.Series) -> Dict:
        """
        执行超参数搜索。

        参数：
            model_class: 需要调参的模型类，应继承自 BaseModel。
            X_train: 训练特征。
            y_train: 训练标签。
            X_val: 验证特征。
            y_val: 验证标签。

        返回：
            最优超参数字典。
        """
        pass