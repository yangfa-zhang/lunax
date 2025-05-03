luna是一个专门用于表格类型数据的机器学习框架

```
luna/
│
├── data/
│   ├── loader.py            # 读取数据
│   ├── preprocessing.py     # 缺失值处理、类别编码等
│
├── models/
│   ├── base.py              # BaseModel定义
│   ├── tree.py              # LightGBM/XGBoost/CatBoost
│   ├── nn.py                # MLP, TabNet, TabFPN等
│
├── tuner/
│   └── optuna_tuner.py      # Optuna的统一接口封装
│
├── trainer/
│   └── trainer.py           # 模型训练、交叉验证
│
├── visualize/
│   ├── training_plot.py     # loss/metric可视化
│   └── interpret.py         # 特征重要性、SHAP等
│
├── utils/
│   ├── metrics.py           # 评估指标
│   └── logger.py            # 日志封装
│
└── main.py                  # 运行主入口（或命令行接口）

```
### 迭代流程
1. 只支持回归 + Xgboost + Optuna + 可视化
2. 