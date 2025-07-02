# sk-neuro

Scikit-learn–style machine learning wrappers for neuroimaging prediction.

## Overview

`sk-neuro` is a lightweight Python package for predictive modeling using neuroimaging features such as brain connectomes. It provides clean and modular wrappers around:

- **Connectome-Based Predictive Modeling (CPM)**
- **ElasticNet regression (GLMNet-style)**
- **SHAP explainability for interpretable models**
- **Bootstrap-based confidence intervals**
- **Nested CV and hyperparameter search utilities**

All components follow the `scikit-learn` API, allowing easy integration into ML pipelines, cross-validation workflows, and hyperparameter tuning schemes.

## Installation

```bash
pip install git+https://github.com/zh1peng/sk-neuro.git
```
Or with local development:

```
git clone https://github.com/yourusername/sk-neuro.git
cd sk-neuro
pip install -e .
```

This package uses a standard `pyproject.toml` and the PEP 517 build system. To create a source distribution or wheel run:

```bash
python -m build
```

