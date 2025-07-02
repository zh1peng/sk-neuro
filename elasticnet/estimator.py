"""
ElasticNet estimator and preprocessing for sk-neuro.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from glmnet import cvglmnet, cvglmnetPredict, cvglmnetCoef
import matplotlib.pyplot as plt
import scipy


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Transformer that clips each feature to the [quantile, 1-quantile] range.

    Parameters
    ----------
    quantile : float, default=0.05
        The lower and upper quantile for clipping (e.g., 0.05 clips to 5th and 95th percentiles).
    copy : bool, default=True
        If False, try to perform operation in-place.
    """
    def __init__(self, quantile: float = 0.05, copy: bool = True):
        self.quantile = quantile
        self.copy = copy

    def _reset(self):
        if hasattr(self, 'data_lb_'):
            del self.data_lb_
            del self.data_ub_

    def fit(self, X: np.ndarray, y=None):
        """Compute quantile bounds for each feature."""
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        self._reset()
        self.data_lb_ = np.percentile(X, 100 * self.quantile, axis=0)
        self.data_ub_ = np.percentile(X, 100 * (1 - self.quantile), axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip features to the fitted quantile range."""
        check_is_fitted(self, ['data_lb_', 'data_ub_'])
        X = check_array(X, copy=self.copy, dtype=FLOAT_DTYPES)
        X = np.clip(X, self.data_lb_, self.data_ub_)
        return X


class GlmnetElasticNetCV(BaseEstimator):
    """
    Sklearn-style cross-validated ElasticNet using glmnet.

    Parameters
    ----------
    alphas : array-like, default=np.linspace(0.1, 1, 10)
        List of alpha values to search.
    nfold : int, default=10
        Number of folds for cross-validation.
    quantile : float, default=0.01
        Quantile for winsorization (clipping outliers).
    ptype : str, default='mse'
        Loss function for glmnet.
    family : str, default='gaussian'
        'gaussian' for univariate, 'mgaussian' for multivariate prediction.
    not2preprocess : list or None
        Indices of features not to preprocess.
    """
    def __init__(self, alphas: np.ndarray = np.linspace(0.1, 1, 10), nfold: int = 10, quantile: float = 0.01,
                 ptype: str = 'mse', family: str = 'gaussian', not2preprocess=None):
        self.alphas = alphas
        self.nfold = nfold
        self.quantile = quantile
        self.ptype = ptype
        self.family = family
        self.not2preprocess = not2preprocess
        self.estimators = []
        self.best_cvm = None
        self.best_alpha = None
        self.best_lambda = None
        self.scaler_ = StandardScaler()
        self.clipper_ = QuantileClipper(quantile=self.quantile)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit ElasticNet with cross-validation over alphas and lambdas.
        """
        if self.not2preprocess is None:
            X2use = X.copy()
            X2win = self.scaler_.fit_transform(X2use)
            self.clipper_.fit(X2win)
            X_win = self.clipper_.transform(X2win)
            X_norm = X_win.copy()
        else:
            X2exclude = X[:, self.not2preprocess]
            X2use = np.delete(X, self.not2preprocess, axis=1)
            X2win = self.scaler_.fit_transform(X2use)
            self.clipper_.fit(X2win)
            X_win = self.clipper_.transform(X2win)
            X_norm = np.hstack([X_win, X2exclude])

        from sklearn.model_selection import KFold
        cv = KFold(n_splits=self.nfold, shuffle=True)
        foldid2use = y[:, 0].copy()
        foldid = -1
        for train_index, test_index in cv.split(X):
            foldid += 1
            foldid2use[test_index] = foldid
        foldid2use = foldid2use.astype(int)
        if y.shape[1] > 1:
            self.family = 'mgaussian'

        alphas = self.alphas
        cvms = []
        lambdas = []
        for alpha2use in alphas:
            clf_obj = cvglmnet(x=X_norm.copy(), y=y.copy(),
                               foldid=foldid2use, alpha=alpha2use,
                               ptype=self.ptype, family=self.family)
            self.estimators.append(clf_obj)
            cvms.append(clf_obj['cvm'].min())
            lambdas.append(clf_obj['lambda_min'])
        min_cvm_idx = cvms.index(min(cvms))
        self.best_estimator = self.estimators[min_cvm_idx]
        self.best_cvm = min(cvms)
        self.best_alpha = alphas[min_cvm_idx]
        self.best_lambda = lambdas[min_cvm_idx]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the best fitted ElasticNet model.
        """
        if self.not2preprocess is None:
            X2use = X.copy()
            X2win = self.scaler_.transform(X2use)
            X_win = self.clipper_.transform(X2win)
            X_norm = X_win.copy()
        else:
            X2exclude = X[:, self.not2preprocess]
            X2use = np.delete(X, self.not2preprocess, axis=1)
            X2win = self.scaler_.transform(X2use)
            X_win = self.clipper_.transform(X2win)
            X_norm = np.hstack([X_win, X2exclude])
        pred = cvglmnetPredict(self.best_estimator, X_norm, s='lambda_min')
        if self.family == 'gaussian':
            pred2return = pred.reshape(-1)
        else:
            pred2return = np.squeeze(pred)
        return pred2return

    def get_info(self) -> dict:
        """
        Return best alpha, lambda, and coefficients for the fitted model.
        """
        info = {}
        info['best_alpha'] = self.best_alpha
        info['best_l1'] = self.best_lambda[0]
        if self.family == 'gaussian':
            info['coef'] = cvglmnetCoef(self.best_estimator, s='lambda_min').reshape(-1)
        else:
            info['coef'] = cvglmnetCoef(self.best_estimator, s='lambda_min')
        return info

    def diagnostic_plot1(self, saveto: str = None):
        """
        Plot cross-validation curves for each alpha.
        """
        fig, axn = plt.subplots(len(self.alphas), 1, figsize=(8, 60), sharey=True)
        for i, ax in enumerate(axn.flat):
            plt.axes(ax)
            from cvglmnetPlot import cvglmnetPlot
            cvglmnetPlot(self.estimators[i])
            textstr = 'alpha=' + str(np.round_(self.alphas[i], 3))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            plt.tight_layout()
        if saveto is not None:
            plt.savefig(saveto, dpi=300)

    def diagnostic_plot2(self, saveto: str = None):
        """
        Plot MSE vs log(lambda) for each alpha.
        """
        for idx, alpha2use in enumerate(self.alphas):
            obj2plot = self.estimators[idx]
            col2use = np.random.rand(3,)
            plt.plot(scipy.log(obj2plot['lambdau']), obj2plot['cvm'], color=col2use)
        plt.xlabel('log(lambda)')
        plt.ylabel('Mean-Squared Error')
        plt.title('alpha-lambda-mse')
        plt.legend(np.round_(self.alphas, 3), loc='best')
        if saveto is not None:
            plt.savefig(saveto, dpi=300)
