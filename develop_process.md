# luna: 表格数据机器学习库

## 代码结构

```
luna/
├── data_processing/
│   ├── __init__.py
│   ├── loader.py          # 数据加载
│   ├── preprocessor.py    # 数据预处理
│   ├── feature.py        # 特征工程
│   ├── splitter.py       # 数据集分割
│   └── validator.py      # 数据验证与清洗
│
├── models/
│   ├── __init__.py
│   ├── tree/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py
│   │   ├── lightgbm_model.py
│   │   ├── catboost_model.py
│   │   ├── random_forest.py
│   │   └── decision_tree.py
│   │
│   └── neural/
│       ├── __init__.py
│       ├── tabnet.py
│       ├── tab_transformer.py
│       ├── tab_fpn.py
│       ├── deepfm.py
│       └── autoint.py
│
├── hyperopt/
│   ├── __init__.py
│   ├── optuna_optimizer.py
│   ├── param_space.py
│   └── parallel.py
│
├── visualization/
│   ├── __init__.py
│   ├── training.py       # 训练过程可视化
│   └── prediction.py     # 预测结果可视化
│
├── evaluation/
│   ├── __init__.py
│   ├── classification.py # 分类指标
│   └── regression.py     # 回归指标
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── model_io.py      # 模型保存与加载
│   ├── config.py        # 配置管理
│   └── experiment.py    # 实验追踪
│
├── api/
│   ├── __init__.py
│   ├── high_level.py    # 高层封装API
│   ├── deployment.py    # 模型部署接口
│   └── batch.py        # 批处理接口
│
├── tests/               # 单元测试目录
│   └── __init__.py
│
├── examples/            # 示例代码目录
│   └── __init__.py
│
├── setup.py            # 包安装配置
├── requirements.txt    # 依赖包列表
└── README.md          # 项目说明文档
```

## 开发流程

### 阶段一

#### 环境配置

requirements.txt

#### 数据模块

- 数据加载

```
def load_data(file_path: str) -> pd.DataFrame:
    """
    从CSV文件中加载数据为 DataFrame。
  
    参数：
        file_path: str - 数据文件路径
  
    返回：
        pd.DataFrame - 加载后的数据
    """

```

- 数据集分割

```
def split_data(
    df: pd.DataFrame, 
    target: str, 
    test_size: float = 0.2, 
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    将数据集按照给定比例划分为 train/val/test。
  
    参数：
        df: 数据集
        target: 目标列名
        test_size: 测试集比例
        val_size: 验证集比例（从 train 中再切）
        random_state: 随机种子
  
    返回：
        train_df, val_df, test_df
    """

```

- 数据预处理

```
def preprocess_data(
    df: pd.DataFrame,
    target: str,
    numeric_strategy: str = "mean",
    category_strategy: str = "most_frequent",
    scale_numeric: bool = True,
    encode_categorical: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """
    对数据进行缺失值处理、编码、标准化等预处理操作。
  
    参数：
        df: 原始数据
        target: 目标列
        numeric_strategy: 数值缺失填充方法 ['mean', 'median']
        category_strategy: 类别缺失填充方法 ['most_frequent']
        scale_numeric: 是否标准化数值特征
        encode_categorical: 是否对类别特征做编码
  
    返回：
        X: 特征矩阵
        y: 标签向量
        preprocess_info: 字典，包含编码器/缩放器等信息
    """
```

#### 模型模块

- base模型

```
class BaseModel(ABC):
    """
    所有模型的基础接口，定义 fit/predict/evaluate 方法。
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        pass

```

- xgboost做回归任务

```
from xgboost import XGBRegressor

class XGBoostRegressor(BaseModel):
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化XGBoost回归模型。
  
        参数：
            params: 可选，传入XGBoost模型的超参数字典
        """
        self.model = XGBRegressor(**(params or {}))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, r2_score
        preds = self.predict(X)
        return {
            "rmse": mean_squared_error(y, preds, squared=False),
            "r2": r2_score(y, preds)
        }

```

### 阶段二

#### 超参数搜索模块

- BaseTuner

```
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

```

- OptunaTuner

```
class OptunaTuner(BaseTuner):
    def __init__(self, 
                 n_trials: int = 50, 
                 direction: str = "minimize", 
                 metric_name: str = "rmse", 
                 timeout: Optional[int] = None):
        """
        初始化 Optuna 调参器。

        参数：
            n_trials: 试验次数。
            direction: 优化方向（"minimize" 或 "maximize"）。
            metric_name: 评估指标名称。
            timeout: 可选，超时时间（秒）。
        """
        pass

    def optimize(self, 
                 model_class: Type[BaseModel], 
                 X_train: pd.DataFrame, 
                 y_train: pd.Series,
                 X_val: pd.DataFrame, 
                 y_val: pd.Series) -> Dict:
        """
        实现 Optuna 的超参数搜索逻辑。

        参考 BaseTuner 接口。
        """
        pass

```

### 阶段三

#### 可视化

添加可视化内容 （学习曲线、解释性分析）

```
viz/
├── __init__.py
├── eda.py              # 数据探索功能
├── feature_analysis.py # 特征重要性、分布等
├── model_eval.py       # 训练结果评估相关
├── target_analysis.py  # 目标变量分析
├── utils.py            # 公共可视化工具（如颜色设置、主题）
```

eda.py

https://www.kaggle.com/code/tarundirector/rev-rain-pred-eda-time-series-ai-news#%5B3%5D-%F0%9F%92%A1-Exploratory-Data-Analysis-(EDA)

https://www.kaggle.com/code/tarundirector/backpack-pred-baseline-ensemble-eda#%5B3%5D-%F0%9F%92%A1-Exploratory-Data-Analysis-(EDA)

配色方案： https://zhuanlan.zhihu.com/p/183710989

- plot_missing_values(df)
- plot_numeric_distributions(df)
- plot_categorical_counts(df)
- correlation_matrix(df)
- pairplot(df, target=None)

feature_analysis.py

- plot_feature_importance(importances, feature_names)
- plot_feature_distribution(df, feature, target)
- plot_feature_vs_target(df, feature, target)

model_eval.py

- plot_learning_curve(train_scores, val_scores)
- plot_confusion_matrix(y_true, y_pred)
- plot_roc_curve(y_true, y_proba)
- plot_precision_recall_curve(y_true, y_proba)
- plot_residuals(y_true, y_pred)

target_analysis.py

- plot_target_distribution(y)
- plot_target_vs_feature(df, target, feature)

### 阶段四

#### 增加其他模型

### 阶段五

#### 增加其他调参方式
