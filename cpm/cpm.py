import numpy as np
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_selection import f_classif
import shap

def regress_out_covariates(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Residualize one or more outcome variables with respect to covariates via ordinary least squares regression.

    This function adds an intercept column to the covariate matrix, computes the least-squares fit
    of Y on C, and returns the residuals (original Y minus fitted values).

    Parameters
    ----------
    Y : np.ndarray, shape (n_samples, n_targets)
        Outcome matrix where each column is a target variable to residualize.
    C : np.ndarray, shape (n_samples, n_covariates)
        Covariate matrix without intercept (columns correspond to covariates).

    Returns
    -------
    np.ndarray, shape (n_samples, n_targets)
        Residuals of Y after regressing out the linear effects of C.

    Raises
    ------
    ValueError
        If the number of rows in Y and C do not match.

    Notes
    -----
    - Uses Moore–Penrose pseudoinverse for numerical stability.
    - Residuals have zero correlation with each covariate (by construction).

    Examples
    --------
    >>> Y = np.random.randn(100, 2)
    >>> C = np.random.randn(100, 3)
    >>> Y_resid = regress_out_covariates(Y, C)
    """
    if Y.shape[0] != C.shape[0]:
        raise ValueError("Number of samples in Y and C must match.")
    n = C.shape[0]
    # Add intercept
    Xc = np.hstack((np.ones((n, 1)), C))
    # Compute fitted values
    pinv = np.linalg.pinv(Xc)
    fitted = Xc @ (pinv @ Y)
    return Y - fitted


def fc_behav_test(
    X: np.ndarray,
    y: np.ndarray,
    task: str = 'regression',
    covariates: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform univariate edge-wise statistical tests between connectivity features and behavior.

    For continuous targets (regression), computes Pearson correlation (r and p-value) per edge.
    For categorical targets (classification), performs one-way ANOVA F-test (F statistic and p-value) per edge.
    Optionally residualizes features and target against covariates prior to testing.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_edges)
        Connectivity feature matrix where each column is a flattened edge weight.
    y : np.ndarray, shape (n_samples,) or (n_samples, 1)
        Target vector: continuous for regression, integer labels for classification.
    task : {'regression', 'classification'}, default 'regression'
        Type of statistical test to perform.
    covariates : np.ndarray, shape (n_samples, n_covariates), optional
        If provided, both X and y will be residualized via regress_out_covariates before testing.

    Returns
    -------
    values : np.ndarray, shape (n_edges,)
        Test statistics per edge (Pearson r or F statistic).
    p_vals : np.ndarray, shape (n_edges,)
        Corresponding p-values per edge.

    Raises
    ------
    ValueError
        If `task` is not one of 'regression' or 'classification'.

    Examples
    --------
    >>> X = np.random.randn(50, 1000)
    >>> y = np.random.randn(50)
    >>> r_vals, p_vals = fc_behav_test(X, y, task='regression')
    """
    # Optional covariate residualization
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

    elif task == 'classification':
        F_vals, p_vals = f_classif(X, y)
        return F_vals, p_vals

    else:
        raise ValueError("Invalid task. Must be 'regression' or 'classification'.")


class EdgeSelector(BaseEstimator, TransformerMixin):
    """
    Selects predictive edges based on univariate tests and summarizes them.

    This transformer retains only those edges whose p-value from a univariate test
    (regression or classification) falls below a given threshold. It then reduces the
    feature space by summarizing the selected edges into two features per sample:
    the sum of all positive-edge weights and the sum of all negative-edge weights.

    Parameters
    ----------
    threshold : float, default=0.01
        P-value cutoff for selecting edges.
    task : {'regression', 'classification'}, default='regression'
        Determines whether to use Pearson correlation or ANOVA for univariate testing.
    covariates : np.ndarray, shape (n_samples, n_covariates), optional
        If provided, residualize features and target against these covariates before selection.

    Attributes
    ----------
    mask_ : np.ndarray, shape (n_edges,)
        Boolean mask indicating which edges were selected (True) or discarded (False).

    Examples
    --------
    >>> selector = EdgeSelector(threshold=0.05, task='regression')
    >>> selector.fit(X, y)
    >>> X_reduced = selector.transform(X)
    """
    def __init__(
        self,
        threshold: float = 0.01,
        task: str = 'regression',
        covariates: np.ndarray = None
    ):
        self.threshold = threshold
        self.task = task
        self.covariates = covariates

    def _apply_mask(self, row_fc: np.ndarray) -> np.ndarray:
        """
        Internal: Keep only the edge weights where mask_ is True.

        Parameters
        ----------
        row_fc : np.ndarray, shape (n_edges,)
            Flattened connectivity vector for one sample.

        Returns
        -------
        np.ndarray, shape (n_selected_edges,)
            Edge weights passing the mask.
        """
        return row_fc[self.mask_]

    def _sum_edges(self, vec: np.ndarray, method: str = 'pos') -> float:
        """
        Internal: Sum edge weights by sign.

        Parameters
        ----------
        vec : np.ndarray, shape (n_selected_edges,)
            Connectivity weights after masking.
        method : {'pos', 'neg'}, default='pos'
            'pos' sums only positive weights;
            'neg' sums only negative weights.

        Returns
        -------
        float
            Sum of selected entries.

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

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Determine which edges survive univariate testing.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)
            Connectivity feature matrix.
        y : np.ndarray, shape (n_samples,) or (n_samples, 1)
            Target variable(s).

        Returns
        -------
        self
            Fitted selector with mask_.

        Raises
        ------
        ValueError
            If no edges remain under the threshold.
        """
        _, p_vals = fc_behav_test(
            X,
            y,
            task=self.task,
            covariates=self.covariates
        )
        self.mask_ = p_vals < self.threshold
        if not np.any(self.mask_):
            raise ValueError(f"No edges survive p<{self.threshold} selection!")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Summarize selected edges into two features per sample.

        Steps:
        1. Apply mask to each sample's connectivity vector.
        2. Compute sum of positive weights and sum of negative weights.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)
            Connectivity features.

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Array where columns are [pos_sum, neg_sum].
        """
        sig = np.array([self._apply_mask(row) for row in X])
        pos = np.apply_along_axis(self._sum_edges, 1, sig, method='pos')
        neg = np.apply_along_axis(self._sum_edges, 1, sig, method='neg')
        return np.vstack((pos, neg)).T

    def get_support_mask(self) -> np.ndarray:
        """
        Return the boolean mask of selected edges.

        Returns
        -------
        np.ndarray, shape (n_edges,)
            Mask where True indicates edges kept during fit().
        """
        return self.mask_


# ─── CPM wrapper with covariate support ────────────────────────────────────────
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression




class CPM_pipe(Pipeline):
    """
    Connectome-based Predictive Modeling (CPM) pipeline combining edge selection and prediction,
    with built-in SHAP explanation support.

    This pipeline consists of two main steps:
      1. **EdgeSelector**: performs univariate testing on each connectivity edge, retains
         those below the specified p-value threshold, and summarizes retained edges into
         two features per subject (sum of positive edges, sum of negative edges), optionally
         appending covariates.
      2. **Estimator**: fits a linear (Ridge) or logistic regression model on the summary
         features (plus covariates, if requested).

    Parameters
    ----------
    threshold : float, default=0.01
        P-value cutoff for selecting edges in the univariate test.
    task : {'regression', 'classification'}, default='regression'
        Determines whether to use Pearson correlation (regression) or ANOVA F-test
        (classification) in EdgeSelector, and which default estimator to use.
    covariates : array-like of shape (n_samples, n_covariates), optional
        Covariate matrix for residualization and optional inclusion in the final model.
    include_covariates : bool, default=False
        If True, horizontally stacks `covariates` onto the two summary features before
        fitting the estimator.
    estimator : estimator instance or None, default=None
        A scikit-learn estimator. If None, defaults to:
          - `Ridge()` for regression tasks
          - `LogisticRegression(max_iter=1000)` for classification tasks

    Attributes
    ----------
    named_steps : dict
        Mapping of pipeline step names to their fitted transformer/estimator objects.
    include_covariates : bool
        Indicates whether covariates were included in the final feature matrix.

    Methods
    -------
    explain(X)
        Compute SHAP values for the summary features (and covariates) in `X`, returning
        (shap_values, feature_matrix, feature_names) to interpret model predictions.
    """

    def __init__(self,
                 threshold: float = 0.01,
                 task: str = 'regression',
                 covariates: np.ndarray = None,
                 include_covariates: bool = False,
                 estimator=None):
        # Step 1: edge selection
        selector = EdgeSelector(threshold=threshold,
                                task=task,
                                covariates=covariates)
        # Step 2: estimator
        if estimator is None:
            estimator = (LogisticRegression(max_iter=1000)
                         if task == 'classification'
                         else Ridge())
        super().__init__([
            ('selector', selector),
            ('estimator', estimator)
        ])
        self.include_covariates = include_covariates

    def explain(self, X: np.ndarray):
        """
        Compute SHAP explanations for the fitted CPM pipeline.

        This method:
          1. Transforms raw connectivity data `X` into summary features via the
             fitted EdgeSelector.
          2. Optionally appends covariates if `include_covariates=True`.
          3. Builds a SHAP LinearExplainer for the final estimator on these features.
          4. Returns the SHAP values alongside the feature matrix and feature names.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_edges)
            Raw connectivity data to explain.

        Returns
        -------
        shap_values : array-like
            SHAP values for each feature and sample.
        feature_matrix : np.ndarray, shape (n_samples, n_features)
            [pos_sum, neg_sum] per sample, plus covariates if included.
        feature_names : list of str
            Names corresponding to columns in `feature_matrix`, e.g. ['pos_sum', 'neg_sum', 'cov0', ...]

        Raises
        ------
        ValueError
            If called before the pipeline has been fitted (i.e., no `selector` or `estimator`).

        Examples
        --------
        >>> cpm = CPM_pipe(threshold=0.05, task='regression', covariates=covs)
        >>> cpm.fit(X, y)
        >>> shap_vals, feats, names = cpm.explain(X)
        >>> print(names)
        ['pos_sum', 'neg_sum', 'cov0', 'cov1']
        """
        # 1) summary features
        feats = self.named_steps['selector'].transform(X)

        # 2) append covariates if requested
        if self.include_covariates:
            covs = self.named_steps['selector'].covariates
            feats = np.hstack((feats, covs))

        # 3) instantiate SHAP explainer
        explainer = shap.LinearExplainer(
            self.named_steps['estimator'],
            feats,
            feature_dependence="independent"
        )
        shap_values = explainer.shap_values(feats)

        # 4) feature names
        names = ['pos_sum', 'neg_sum']
        if self.include_covariates:
            names += [f'cov{i}' for i in range(covs.shape[1])]

        return shap_values, feats, names


