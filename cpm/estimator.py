#!/usr/bin/env python3
"""cpm_code/CPM_pipe.py — Connectome-based Predictive Modeling (clean version).

This file defines four public callables that are fully clone-able and
cross-validation friendly:

``regress_out_covariates``
    Linear residualisation via ordinary least squares.
``fc_behav_test``
    Edge-wise Pearson/ANOVA with optional covariate residualisation.
``EdgeSelector``
    Transformer that converts full FC edges → CPM summary features.
``CPM_pipe``
    End-to-end CPM pipeline that routes columns by **name** and fits a final
    linear/logistic model. Compatible with :func:`sklearn.base.clone`.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np
import shap
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

__all__ = [
    "regress_out_covariates",
    "fc_behav_test",
    "EdgeSelector",
    "CPM_pipe",
]

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _to_numpy(arr):
    """Convert input to NumPy array without copying if already an ndarray.

    Parameters
    ----------
    arr : array-like
        Any object convertible to a NumPy array.

    Returns
    -------
    np.ndarray
        The input as a NumPy array.
    """
    return np.asarray(arr)


def regress_out_covariates(Y: np.ndarray, C: np.ndarray) -> np.ndarray:  # noqa: N802
    """Residualise target(s) Y with respect to covariates C via OLS.

    Solves for β in [1, C]β ≈ Y and returns the residuals Y − Ŷ.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_targets)
        Matrix of outcomes to residualise (each column is one target variable).
    C : ndarray, shape (n_samples, n_covariates)
        Covariate matrix (does *not* include intercept column).

    Returns
    -------
    residuals : ndarray, shape (n_samples, n_targets)
        The matrix of residuals after regressing out C (with intercept).

    Raises
    ------
    ValueError
        If Y and C have mismatched numbers of rows (samples).

    Examples
    --------
    >>> Y = np.random.randn(100, 2)
    >>> C = np.random.randn(100, 3)
    >>> resid = regress_out_covariates(Y, C)
    >>> resid.shape
    (100, 2)
    """
    Y = _to_numpy(Y)
    C = _to_numpy(C)
    if Y.shape[0] != C.shape[0]:
        raise ValueError("Y and C must have the same number of rows.")
    # add intercept column of ones
    Xc = np.hstack((np.ones((Y.shape[0], 1)), C))
    # compute fitted values and subtract
    beta = np.linalg.pinv(Xc) @ Y
    return Y - (Xc @ beta)


def fc_behav_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task: str = "regression",
    covariates: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform edge-wise univariate tests between FC edges and behavior.

    For regression, uses two-tailed Pearson correlation per edge.
    For classification, uses ANOVA F-test per edge.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_edges)
        Matrix of flattened functional connectivity edge weights.
    y : ndarray, shape (n_samples,)
        Vector of behavioral or diagnostic values.
    task : {'regression', 'classification'}, default='regression'
        Type of statistical test to perform.
        - 'regression': two-tailed Pearson correlation.
        - 'classification': F-test (ANOVA) via sklearn.feature_selection.f_classif.
    covariates : ndarray, shape (n_samples, n_covariates) or None, default=None
        Optional covariate matrix. If provided, both X and y are
        residualised with respect to these via :func:`regress_out_covariates`.

    Returns
    -------
    stats : ndarray, shape (n_edges,)
        Test statistic per edge (Pearson r or F-value).
    pvals : ndarray, shape (n_edges,)
        Two-tailed p-value per edge (or ANOVA p-value).

    Raises
    ------
    ValueError
        If `task` is not one of the allowed options.

    See Also
    --------
    scipy.stats.pearsonr
    sklearn.feature_selection.f_classif

    Examples
    --------
    >>> X = np.random.randn(50, 1000)
    >>> y = np.random.randn(50)
    >>> stats, pvals = fc_behav_test(X, y, task='regression')
    >>> stats.shape, pvals.shape
    ((1000,), (1000,))
    """
    X = _to_numpy(X)
    y = _to_numpy(y).ravel()

    # residualise if covariates provided
    if covariates is not None:
        X = regress_out_covariates(X, covariates)
        y = regress_out_covariates(y.reshape(-1, 1), covariates).ravel()

    if task == "regression":
        n_edges = X.shape[1]
        stats = np.empty(n_edges)
        pvals = np.empty(n_edges)
        for i in range(n_edges):
            stats[i], pvals[i] = pearsonr(X[:, i], y)
        return stats, pvals

    if task == "classification":
        return f_classif(X, y)

    raise ValueError("task must be 'regression' or 'classification'.")


# -----------------------------------------------------------------------------
# Transformer: EdgeSelector
# -----------------------------------------------------------------------------

class EdgeSelector(BaseEstimator, TransformerMixin):
    """Transformer that selects behavior-associated FC edges and summarizes them.

    During `fit`, performs univariate tests on each edge and builds boolean
    masks for positive and/or negative edge sets below threshold. During
    `transform`, sums edge weights across each selected set to yield one or
    two summary features per sample: 'pos_sum' and/or 'neg_sum'.

    Parameters
    ----------
    threshold : float, default=0.01
        P-value threshold for selecting edges.
    task : {'regression', 'classification'}, default='regression'
        Type of univariate test (see `fc_behav_test`).
    network_type : {'pos', 'neg', 'both'}, default='both'
        Which edges to include in summary:
        - 'pos': only edges with positive association.
        - 'neg': only edges with negative association.
        - 'both': include both and return two features.
    covariates : ndarray, shape (n_samples, n_covariates) or None
        Optional covariate matrix for residualisation before edge testing.

    Attributes
    ----------
    mask_pos_ : ndarray of bool, shape (n_edges,)
        Boolean mask of positively associated edges surviving threshold.
    mask_neg_ : ndarray of bool, shape (n_edges,)
        Boolean mask of negatively associated edges surviving threshold.

    Examples
    --------
    >>> sel = EdgeSelector(threshold=0.05, task='regression', network_type='both')
    >>> sel.fit(X, y)
    >>> feats = sel.transform(X)
    >>> feats.shape  # two summary features per sample
    (n_samples, 2)
    """

    def __init__(
        self,
        threshold: float = 0.01,
        task: str = "regression",
        network_type: str = "both",
        covariates: np.ndarray | None = None,
    ) -> None:
        self.threshold = threshold
        self.task = task
        self.network_type = network_type
        self.covariates = covariates

    def fit(self, X, y):  # noqa: D401
        """Learn which edges survive behavioral association threshold.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_edges)
            FC edge matrix.
        y : array-like, shape (n_samples,)
            Behavioral or diagnostic vector.

        Returns
        -------
        self : EdgeSelector
            Fitted selector with `mask_pos_` and `mask_neg_` attributes.

        Raises
        ------
        ValueError
            If no edges survive the given threshold for the specified network_type.
        """
        X = _to_numpy(X)
        y = _to_numpy(y).ravel()
        vals, p = fc_behav_test(X, y, task=self.task, covariates=self.covariates)
        self.mask_pos_ = (vals > 0) & (p < self.threshold)
        self.mask_neg_ = (vals < 0) & (p < self.threshold)

        if self.network_type == "pos" and not self.mask_pos_.any():
            raise ValueError("No positive edges survive threshold.")
        if self.network_type == "neg" and not self.mask_neg_.any():
            raise ValueError("No negative edges survive threshold.")
        if self.network_type == "both" and not (
            self.mask_pos_.any() or self.mask_neg_.any()
        ):
            raise ValueError("No edges survive threshold.")
        return self

    def transform(self, X):  # noqa: D401
        """Transform input to CPM summary features based on fitted masks.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_edges)
            FC edge matrix to summarize.

        Returns
        -------
        features : ndarray, shape (n_samples, n_features)
            Summary features: one column per selected network_type ('pos_sum',
            'neg_sum', or both).
        """
        X = _to_numpy(X)
        feats: List[np.ndarray] = []
        if self.network_type in ("both", "pos"):
            feats.append(X[:, self.mask_pos_].sum(axis=1))
        if self.network_type in ("both", "neg"):
            feats.append(X[:, self.mask_neg_].sum(axis=1))
        return np.vstack(feats).T

    def get_feature_names(self):  # noqa: D401
        """Get names of the summary features produced by `transform`.

        Returns
        -------
        names : list of str
            List containing 'pos_sum' and/or 'neg_sum', depending on network_type.
        """
        names: List[str] = []
        if self.network_type in ("both", "pos"):
            names.append("pos_sum")
        if self.network_type in ("both", "neg"):
            names.append("neg_sum")
        return names


# -----------------------------------------------------------------------------
# End-to-end CPM pipeline
# -----------------------------------------------------------------------------

class CPM_pipe(Pipeline, RegressorMixin):
    """Complete CPM workflow: edge selection → covariate passthrough → model.

    Wraps EdgeSelector and a linear/logistic estimator into a scikit-learn
    Pipeline that selects FC edges, optionally passes through covariates,
    and fits a regression or classification model.

    Parameters
    ----------
    fc_features : iterable of str
        Column names in the input DataFrame (or keys in dict) corresponding
        to flattened FC edges.
    covariate_features : iterable of str or None, default=None
        Column names for covariates to pass through into the final estimator.
    threshold : float, default=0.01
        P-value threshold for edge selection.
    task : {'regression','classification'}, default='regression'
        Task type; selects default estimator (Ridge for regression,
        LogisticRegression for classification) if `estimator` is None.
    network_type : {'pos','neg','both'}, default='both'
        Which edges to include ('pos','neg','both').
    estimator : estimator instance or None, default=None
        Custom estimator to use after edge selection and covariate passthrough.
        Must implement `fit` and `predict` (and `predict_proba` if
        classification).

    Attributes
    ----------
    named_steps : dict
        Dictionary mapping step names to transformers/estimators:
        - 'pre': ColumnTransformer with 'cpm' (EdgeSelector) and 'cov'.
        - 'model': final estimator instance.
    """

    def __init__(
        self,
        fc_features: Iterable[str],
        covariate_features: Iterable[str] | None = None,
        *,
        threshold: float = 0.01,
        task: str = "regression",
        network_type: str = "both",
        estimator=None,
    ) -> None:
        # store params for clone() compatibility
        self.fc_features = fc_features
        self.covariate_features = covariate_features
        self.threshold = threshold
        self.task = task
        self.network_type = network_type
        self.estimator = estimator

        # prepare column lists
        _fc_cols = list(fc_features)
        _cov_cols = [] if covariate_features is None else list(covariate_features)

        # default estimator if none provided
        _est = (
            estimator
            if estimator is not None
            else (Ridge() if task == "regression" else LogisticRegression(max_iter=1000))
        )

        # build preprocessing + modeling pipeline
        pre = ColumnTransformer(
            [
                ("cpm", EdgeSelector(threshold, task, network_type), _fc_cols),
                ("cov", "passthrough", _cov_cols),
            ]
        )
        super().__init__([("pre", pre), ("model", _est)])

    # def explain(self, X):
    #     """Compute SHAP values for CPM features + covariates.

    #     Parameters
    #     ----------
    #     X : DataFrame-like or dict
    #         Input data containing FC edge columns and (optional) covariates.

    #     Returns
    #     -------
    #     shap_vals : array
    #         SHAP values for each feature (CPM summary + covariates).
    #     feats : ndarray, shape (n_samples, n_features)
    #         Numeric feature matrix used for SHAP (output of 'pre'.transform).
    #     names : list of str
    #         Feature names corresponding to columns in `feats`.

    #     Examples
    #     --------
    #     >>> pipe = CPM_pipe(fc_cols, cov_cols, task='regression')
    #     >>> pipe.fit(df)
    #     >>> shap_vals, feats, names = pipe.explain(df)
    #     """
    #     feats = self.named_steps["pre"].transform(X)
    #     expl = shap.LinearExplainer(
    #         self.named_steps["model"], feats, feature_perturbation="correlation_dependent" 
    #     )
    #     shap_vals = expl.shap_values(feats)
    #     names = self.named_steps["pre"].named_transformers_["cpm"].get_feature_names()
    #     if self.covariate_features is not None:
    #         names.extend(self.covariate_features)
    #     return shap_vals, feats, names
