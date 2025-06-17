[![Python version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/lunax/)
### 
[CN](README.CN.md) | [EN](README.md)
### 

<div>

<a href="./imgs/luna3.jpg"><img src="./imgs/luna3.jpg" width="90" align="left" /></a>``Lunax`` is a machine learning framework specifically designed for the processing and analysis of tabular data. The name **Lunax** is derived from the name of a beloved feline mascot luna🐱 at South China University of Technology. Navigate to [API documentations](https://lunax-doc.readthedocs.io/en/latest/) for more detailed information. **⭐️ Star it if you like it ⭐️**
</div>

---

### Installation
```bash
conda create -n your_env_name python=3.11
conda activate your_env_name
pip install lunax
```
### Features
- Data loading and Data pre-processing
- EDA analysis
- Supports multi-model training and Hyperparameter tuning
- Comprehensive model evaluation
- Ensemble learning
- Explainable AI (XAI)

### Quick Start
#### Data Loading and Pre-processing
```Python
from lunax.data_processing import *
df_train = load_data('train.csv') # or df = load_data('train.parquet')
target = 'label_column_name'
df_train = preprocess_data(df_train,target) # data pre-processing, including missing value handling, feature encoding, feature scaling
X_train, X_val, y_train, y_val = split_data(df_train, target)
```
#### Exploratory Data Analysis
```Python
from lunax.viz import numeric_eda, categoric_eda
numeric_eda([df_train,df_test],['train','test'],target=target) # numeric feature analysis
categoric_eda([df_train,df_test],['train','test'],target=target) # categorical feature analysis
```

<table>
  <tr>
    <td><img src="./imgs/eda.png" width="600"/></td>
    <td><img src="./imgs/eda2.png" width="600"/></td>
  </tr>
</table>

#### Automation Machine Learning Modeling
```Python
from lunax.models import xgb_clf # or xgb_reg, lgbm_reg, lgbm_clf, cat_clf, cat_reg
from lunax.hyper_opt import OptunaTuner
tuner = OptunaTuner(n_trials=10,model_class="XGBClassifier") # Hyperparameter optimizer, n_trials is the number of optimization times
# or "XGBRegressor", "LGBMRegressor", "LGBMClassifier" , "CatClassifier", "CatRegressor"
results = tuner.optimize(X_train, y_train, X_val, y_val)
best_params = results['best_params']
model = xgb_clf(best_params)
model.fit(X_train, y_train)
```
#### Model Evaluation
```Python
model.evaluate(X_val, y_val)
```
```text
[lunax]> label information:
+---------+---------+
|   label |   count |
+=========+=========+
|       1 |     319 |
+---------+---------+
|       0 |     119 |
+---------+---------+
[lunax]> model evaluation results:
+-----------+------------+-------------+----------+------+
| metrics   |   accuracy |   precision |   recall |   f1 |
+===========+============+=============+==========+======+
| values    |       0.73 |        0.53 |     0.73 | 0.61 |
+-----------+------------+-------------+----------+------+
```
#### Ensemble Learning
```Python
from lunax.ensembles import HillClimbingEnsemble
model1 = xgb_clf()
model2 = lgbm_clf()
model3 = cat_clf()
for model in [model1, model2, model3]:
    model.fit(X_train, y_train)
ensemble = HillClimbingEnsemble(
    models=[model1, model2, model3],
    metric=['auc'],
    maximize=True
)
best_weights = ensemble.fit(X_val, y_val)
predictions = ensemble.predict(df_test)
```
#### Explainable AI (XAI)
```Python
from lunax.xai import TreeExplainer
explainer = TreeExplainer(model)
explainer.plot_summary(X_val)
importance = explainer.get_feature_importance(X_val)
```
```text
[lunax]> Clear blue/red separation indicates a highly influential feature.
```
<img src="./imgs/shap.png" width="600" />

```text
[lunax]> Feature Importance Ranking:
+----+---------------+---------------------+
|    |    Feature    |     Importance      |
+----+---------------+---------------------+
| 1  |     cloud     | 2.3085615634918213  |
| 2  |   sunshine    | 0.6377484202384949  |
| 3  |   dewpoint    | 0.5257667899131775  |
| 4  |   humidity    | 0.4827548861503601  |
| 5  |   windspeed   | 0.40086665749549866 |
| 6  |      id       | 0.38620123267173767 |
| 7  |   pressure    | 0.3780971169471741  |
| 8  |    mintemp    | 0.32988569140434265 |
| 9  |      day      | 0.30587586760520935 |
| 10 |    maxtemp    | 0.26082852482795715 |
| 11 | winddirection | 0.23236176371574402 |
| 12 |  temparature  | 0.17218443751335144 |
+----+---------------+---------------------+
```

