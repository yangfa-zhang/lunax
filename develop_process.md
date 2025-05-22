

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

调参可视化：参考https://bluecast.readthedocs.io/en/latest/Model%20explainability%20%28XAI%29.html

看看因果推断在这里有什么用：https://erdogant.github.io/bnlearn/pages/html/Examples.html
hyper tuning过程可视化


https://www.kaggle.com/code/thedevastator/the-fine-art-of-hyperparameter-tuning

发布：

```
python setup.py sdist
twine upload dist\lunax-0.0.7.tar.gz
```

### 参考kaggle笔记本：
https://www.kaggle.com/competitions/playground-series-s3e4/discussion/381305#2116795
https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm

### 阶段六：数据处理增强


#### 异常值处理

```
def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 1.5
) -> Dict[str, List[int]]:
    """
    检测数值型特征中的异常值。
  
    参数：
        df: 数据
        method: 检测方法 ["iqr", "zscore", "isolation_forest"]
        threshold: 阈值
  
    返回：
        各特征异常值的索引字典
    """
```

#### 特征选择

```
def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "variance",
    threshold: float = 0.01
) -> List[str]:
    """
    选择重要特征。
  
    参数：
        X: 特征矩阵
        y: 目标变量
        method: 选择方法 ["variance", "mutual_info", "chi2", "lasso"]
        threshold: 选择阈值
  
    返回：
        选中的特征列表
    """
```



#### AutoML

```
class AutoML:
    def __init__(
        self,
        task_type: str,
        time_limit: int = 3600,
        metric: str = "auto"
    ):
        """
        初始化AutoML系统。
  
        参数：
            task_type: 任务类型 ["classification", "regression"]
            time_limit: 时间限制（秒）
            metric: 评估指标
        """
```

#### 模型解释

```
class ModelExplainer:
    def __init__(self, model: BaseModel):
        """
        初始化模型解释器。
  
        参数：
            model: 需要解释的模型
        """
  
    def explain_prediction(self, X: pd.DataFrame) -> Dict:
        """
        解释模型预测。
  
        参数：
            X: 特征矩阵
  
        返回：
            预测解释结果
        """
```

### 阶段八：实验管理

#### 实验追踪

```
class ExperimentTracker:
    def __init__(self, experiment_name: str):
        """
        初始化实验追踪器。
  
        参数：
            experiment_name: 实验名称
        """
  
    def log_params(self, params: Dict) -> None:
        """记录实验参数"""
  
    def log_metrics(self, metrics: Dict) -> None:
        """记录实验指标"""
  
    def log_artifact(self, path: str) -> None:
        """记录实验产物"""
```

#### 模型版本控制

```
class ModelVersionControl:
    def __init__(self, model_dir: str):
        """
        初始化模型版本控制系统。
  
        参数：
            model_dir: 模型存储目录
        """
  
    def save_version(self, model: BaseModel, version: str) -> None:
        """保存模型版本"""
  
    def load_version(self, version: str) -> BaseModel:
        """加载模型版本"""
```

#### 批处理管道

```
class BatchProcessor:
    def __init__(
        self,
        model: BaseModel,
        batch_size: int = 1000,
        max_workers: int = 4
    ):
        """
        初始化批处理器。
  
        参数：
            model: 预测模型
            batch_size: 批处理大小
            max_workers: 最大工作进程数
        """
  
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """执行批处理预测"""
```

### 阶段十：性能优化

#### 并行处理

```
class ParallelProcessor:
    def __init__(self, n_jobs: int = -1):
        """
        初始化并行处理器。
  
        参数：
            n_jobs: 并行任务数，-1表示使用所有CPU
        """
  
    def parallel_apply(self, func: Callable, data: List) -> List:
        """并行执行函数"""
```

#### 实时监控

```
class TrainingMonitor:
    def __init__(self, update_interval: int = 1):
        """
        初始化训练监控器。
  
        参数：
            update_interval: 更新间隔（秒）
        """
  
    def start_monitoring(self) -> None:
        """开始监控"""
  
    def stop_monitoring(self) -> None:
        """停止监控"""
```
