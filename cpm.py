import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
import shap

# ─── Core utilities ───────────────────────────────────────────────────────────

def apply_mask(row_fc, mask):
    """Apply a boolean mask to a 1D FC vector."""
    return row_fc[mask]

def sum_edges(vec, method='pos'):
    """Sum positive or negative entries of a 1D vector."""
    if method == 'pos':
        return vec[vec > 0].sum()
    elif method == 'neg':
        return vec[vec < 0].sum()
    else:
        raise ValueError("method must be 'pos' or 'neg'")

def fc_behav_corr(fc, y):
    """
    Edge-wise Pearson correlation.
    Returns arrays (r_vals, p_vals) of length = n_edges.
    """
    y = y.ravel()
    E = fc.shape[1]
    r_vals = np.empty(E)
    p_vals = np.empty(E)
    for i in range(E):
        r_vals[i], p_vals[i] = pearsonr(fc[:, i], y)
    return r_vals, p_vals

# ─── Edge selector ────────────────────────────────────────────────────────────

class SelectEdges(BaseEstimator, TransformerMixin):
    """Feature‐selector: keep edges with p < threshold, then sum pos/neg edges."""
    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y):
        _, p_vals = fc_behav_corr(X, y)
        self.mask_ = p_vals < self.threshold
        return self

    def transform(self, X):
        # apply mask, then sum up pos/neg edges per subject
        sig = np.array([apply_mask(x, self.mask_) for x in X])
        pos = np.apply_along_axis(sum_edges, 1, sig, method='pos')
        neg = np.apply_along_axis(sum_edges, 1, sig, method='neg')
        return np.vstack((pos, neg)).T

    def get_support_mask(self):
        return self.mask_

# ─── CPM wrapper with SHAP support ───────────────────────────────────────────

class CPM(BaseEstimator):
    """
    Sklearn‐style CPM with .fit/.predict and .explain for SHAP.
    Internally uses SelectEdges + a linear model (default Ridge).
    """
    def __init__(self, threshold=0.01, estimator=None):
        self.threshold = threshold
        self.estimator = estimator if estimator is not None else Ridge()
        self.selector = SelectEdges(threshold=self.threshold)

    def fit(self, X, y):
        self.selector.set_params(threshold=self.threshold).fit(X, y)
        feats = self.selector.transform(X)
        self.estimator.fit(feats, y)
        return self

    def predict(self, X):
        feats = self.selector.transform(X)
        return self.estimator.predict(feats)

    def explain(self, X):
        """
        Returns (shap_values, feature_matrix, feature_names).
        Feature names are ['pos_sum','neg_sum'] by construction.
        """
        feats = self.selector.transform(X)
        explainer = shap.LinearExplainer(self.estimator, feats, feature_dependence="independent")
        shap_vals = explainer.shap_values(feats)
        return shap_vals, feats, ['pos_sum', 'neg_sum']

# ─── Nested CV + Bootstrap CI example ────────────────────────────────────────

# assume X (n×E) and y (n,) are your data arrays

# 1. Build pipeline & param grid
pipe = Pipeline([
    ('sel', SelectEdges()),        # will get its threshold from GridSearchCV
    ('reg', Ridge())
])
param_grid = {
    'sel__threshold': [1e-4, 1e-3, 1e-2, 5e-2],
    'reg__alpha':   np.logspace(-3, 1, 10)
}

inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=10, shuffle=True, random_state=1)

# 2. Wrap in a GridSearchCV for nested tuning
grid = GridSearchCV(pipe,
                    param_grid,
                    cv=inner_cv,
                    scoring='r2',
                    n_jobs=-1)

# 3. Outer‐loop CV predictions
y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)
r_outer = pearsonr(y, y_pred)[0]
print(f"Nested‐CV Pearson r = {r_outer:.3f}")

# 4. Bootstrap CI on that r
def bootstrap_ci(model, X, y, cv, n_boot=500, ci=95):
    rng = np.random.RandomState(2)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]; Xb = X[idx]
        ypb = cross_val_predict(model, Xb, yb, cv=cv, n_jobs=-1)
        stats.append(pearsonr(yb, ypb)[0])
    lo = np.percentile(stats, (100 - ci) / 2)
    hi = np.percentile(stats, 100 - (100 - ci) / 2)
    return lo, hi

lo, hi = bootstrap_ci(grid, X, y, outer_cv)
print(f"95% CI for r: [{lo:.3f}, {hi:.3f}]")

# 5. Refit on full data & SHAP explain
grid.fit(X, y)
best_pipe = grid.best_estimator_
shap_vals, feats, names = best_pipe.named_steps['sel'].transform(X), None, None
# actually, to get SHAP:
selector = best_pipe.named_steps['sel']
model    = best_pipe.named_steps['reg']
feats    = selector.transform(X)
explainer= shap.LinearExplainer(model, feats, feature_dependence="independent")
shap_vals= explainer.shap_values(feats)

# now you can:
# shap.summary_plot(shap_vals, feats, feature_names=['pos_sum','neg_sum'])
# shap.force_plot(explainer.expected_value, shap_vals[0], feats[0], feature_names=['pos_sum','neg_sum'])
