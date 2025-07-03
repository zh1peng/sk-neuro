import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import f_classif
import shap

def apply_mask(row_fc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Keep only the elements of a 1D connectivity vector indicated by a boolean mask.

    Parameters
    ----------
    row_fc : np.ndarray, shape (n_edges,)
        Flattened connectivity values (upper triangle of an N×N FC matrix).
    mask : np.ndarray of bool, shape (n_edges,)
        Indicates which edges to retain.

    Returns
    -------
    np.ndarray, shape (n_selected_edges,)
        The subset of row_fc where mask is True.
    """
    return row_fc[mask]


def sum_edges(vec: np.ndarray, method: str = 'pos') -> float:
    """
    Aggregate a vector of edge weights by summing only positive or only negative entries.

    Parameters
    ----------
    vec : np.ndarray, shape (n,)
        Vector of edge weights.
    method : {'pos', 'neg'}
        'pos' sums elements > 0; 'neg' sums elements < 0.

    Returns
    -------
    float
        Sum of the selected entries.

    Raises
    ------
    ValueError
        If method is not 'pos' or 'neg'.
    """
    if method == 'pos':
        return vec[vec > 0].sum()
    elif method == 'neg':
        return vec[vec < 0].sum()
    else:
        raise ValueError("method must be 'pos' or 'neg'")


def regress_out_covariates(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Residualize outcome(s) Y with respect to covariates C via linear regression.

    Adds an intercept to C and removes its linear contribution from Y.

    Parameters
    ----------
    Y : np.ndarray, shape (n_samples, n_targets)
        Outcome matrix to residualize.
    C : np.ndarray, shape (n_samples, n_covariates)
        Covariate matrix (columns are covariates).

    Returns
    -------
    np.ndarray, shape same as Y
        Residuals of Y after regressing out C.
    """
    n = C.shape[0]
    Xc = np.hstack((np.ones((n, 1)), C))
    pinv = np.linalg.pinv(Xc)
    fitted = Xc @ (pinv @ Y)
    return Y - fitted


def fc_behav_test(X: np.ndarray,
                  y: np.ndarray,
                  task: str = 'regression',
                  covariates: np.ndarray = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform edge-wise univariate tests between connectivity features and target.

    For regression: Pearson correlation (r, p-value).  
    For classification: ANOVA F-test (F, p-value).

    Optionally residualizes both X and y against covariates first.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_edges)
        Connectivity features.
    y : np.ndarray, shape (n_samples,) or (n_samples, 1)
        Continuous target (regression) or class labels (classification).
    task : {'regression', 'classification'}
        Type of test.
    covariates : np.ndarray, shape (n_samples, k), optional
        If provided, regress out from X and y before testing.

    Returns
    -------
    values : np.ndarray, shape (n_edges,)
        Pearson r values or ANOVA F values per edge.
    p_vals : np.ndarray, shape (n_edges,)
        Corresponding p-values.
    """
    if covariates is not None:
        X = regress_out_covariates(X, covariates)
        y = regress_out_covariates(y.reshape(-1, 1), covariates).ravel()

    if task == 'regression':
        E = X.shape[1]
        r_vals = np.empty(E)
        p_vals = np.empty(E)
        for i in range(E):
            r_vals[i], p_vals[i] = pearsonr(X[:, i], y)
        return r_vals, p_vals
    else:
        F_vals, p_vals = f_classif(X, y)
        return F_vals, p_vals


# ─── Edge selector ────────────────────────────────────────────────────────────

class EdgeSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that retains edges with univariate p-value below a threshold,
    then summarizes retained edges into two features per subject (pos_sum, neg_sum).

    Parameters
    ----------
    threshold : float
        p-value cutoff for edge selection.
    task : {'regression', 'classification'}
        Determines the univariate test to use.
    covariates : np.ndarray, optional
        If provided, residualize X and y against these covariates before testing.
    """
    def __init__(self, threshold: float = 0.01,
                 task: str = 'regression',
                 covariates: np.ndarray = None):
        self.threshold = threshold
        self.task = task
        self.covariates = covariates

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Compute the mask of edges to retain.

        Parameters
        ----------
        X : np.ndarray, shape (n, E)
        y : np.ndarray, shape (n,) or (n, 1)

        Returns
        -------
        self
        """
        _, p_vals = fc_behav_test(X, y,
                                  task=self.task,
                                  covariates=self.covariates)
        self.mask_ = p_vals < self.threshold
        if not np.any(self.mask_):
            raise ValueError(f"No edges survive p<{self.threshold} selection!")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the mask and compute summary features.

        Parameters
        ----------
        X : np.ndarray, shape (n, E)

        Returns
        -------
        np.ndarray, shape (n, 2)
            Columns are [sum of positive edges, sum of negative edges].
        """
        sig = np.array([apply_mask(row, self.mask_) for row in X])
        pos = np.apply_along_axis(sum_edges, 1, sig, method='pos')
        neg = np.apply_along_axis(sum_edges, 1, sig, method='neg')
        return np.vstack((pos, neg)).T

    def get_support_mask(self) -> np.ndarray:
        """
        Retrieve the boolean mask of retained edges.

        Returns
        -------
        np.ndarray, shape (E,)
        """
        return self.mask_


# ─── CPM wrapper with covariate support ────────────────────────────────────────

class CPM(BaseEstimator):
    """
    Connectome-based Predictive Model (CPM) with optional covariate adjustment.

    Steps:
      1. Residualize (optional) and select edges via univariate tests.
      2. Summarize selected edges into two features per subject.
      3. (Optional) Append covariates to summary features.
      4. Fit a final linear or logistic model.

    Parameters
    ----------
    threshold : float, default=0.01
        p-value threshold for edge selection.
    task : {'regression', 'classification'}, default='regression'
        Type of prediction task.
    covariates : array-like, shape (n_samples, n_covariates), optional
        Covariates to adjust for and/or include.
    include_covariates : bool, default=False
        If True, concatenates covariates to summary features before final modeling.
    estimator : sklearn estimator, optional
        Overrides default Ridge (regression) or LogisticRegression (classification).
    """
    def __init__(self,
                 threshold: float = 0.01,
                 task: str = 'regression',
                 covariates: np.ndarray = None,
                 include_covariates: bool = False,
                 estimator=None):
        self.threshold = threshold
        self.task = task
        self.covariates = None if covariates is None else np.asarray(covariates)
        self.include_covariates = include_covariates

        if estimator is None:
            self.estimator = (LogisticRegression(max_iter=1000)
                              if task == 'classification'
                              else Ridge())
        else:
            self.estimator = estimator

        self.selector = EdgeSelector(threshold=self.threshold,
                                     task=self.task,
                                     covariates=self.covariates)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the CPM pipeline on data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)
        y : np.ndarray, shape (n_samples,) or (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n = X.shape[0]

        if self.include_covariates:
            if self.covariates is None:
                raise ValueError("include_covariates=True but no covariates provided")
            if self.covariates.shape[0] != n:
                raise ValueError("covariates must have same n_samples as X")

        # 1) Edge selection (with optional covariate residualization)
        self.selector.set_params(threshold=self.threshold,
                                  task=self.task,
                                  covariates=self.covariates).fit(X, y)

        # 2) Summary features
        feats = self.selector.transform(X)

        # 3) Append covariates if requested
        if self.include_covariates:
            feats = np.hstack((feats, self.covariates))

        # 4) Final model fit
        self.estimator.fit(feats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the fitted CPM model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        feats = self.selector.transform(X)
        if self.include_covariates:
            feats = np.hstack((feats, self.covariates))
        return self.estimator.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
        """
        feats = self.selector.transform(X)
        if self.include_covariates:
            feats = np.hstack((feats, self.covariates))
        return self.estimator.predict_proba(feats)

    def explain(self, X: np.ndarray):
        """
        Compute SHAP values for the summary features (and covariates).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)

        Returns
        -------
        shap_vals : array-like
            SHAP values for each feature.
        feats : np.ndarray
            Feature matrix used for explanation.
        names : List[str]
            Names of features (['pos_sum','neg_sum', ...covars]).
        """
        feats = self.selector.transform(X)
        if self.include_covariates:
            feats = np.hstack((feats, self.covariates))
        explainer = shap.LinearExplainer(self.estimator,
                                         feats,
                                         feature_dependence="independent")
        shap_vals = explainer.shap_values(feats)

        names = ['pos_sum', 'neg_sum']
        if self.include_covariates:
            names += [f"cov{i}" for i in range(self.covariates.shape[1])]

        return shap_vals, feats, names
