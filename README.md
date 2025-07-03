# sk-neuro

Scikit-learn–style machine learning wrappers for neuroimaging prediction.

## Overview

`sk-neuro` provides lightweight estimators for predictive modelling with brain features. Core components include:

- **Connectome-Based Predictive Modeling (CPM)**
- **ElasticNet regression with glmnet**
- **Utilities for cross-validation and visualisation**

All estimators follow the scikit-learn API so they can be easily combined with pipelines and hyperparameter search tools.

## Installation

```bash
pip install git+https://github.com/zh1peng/sk-neuro.git
```

Or from a local clone:

```bash
git clone https://github.com/zh1peng/sk-neuro.git
cd sk-neuro
pip install -e .
```

## Quick start

### CPM example

```python
import numpy as np
from skneuro.cpm import CPM

# toy data: 100 subjects × 200 connectivity edges
X = np.random.randn(100, 200)
y = np.random.randn(100)

model = CPM(threshold=0.01)
model.fit(X, y)
print(model.predict(X[:5]))  # predictions for first 5 subjects

# SHAP explanation
shap_vals, feats, names = model.explain(X[:5])
```

### ElasticNet example

```python
from skneuro.elasticnet.estimator import GlmnetElasticNetCV

X = np.random.randn(50, 20)
y = np.random.randn(50, 1)

clf = GlmnetElasticNetCV(nfold=5)
clf.fit(X, y)
print(clf.predict(X[:3]))
```

See the documentation strings in each module for further details.

## License

This project is licensed under the MIT License.
