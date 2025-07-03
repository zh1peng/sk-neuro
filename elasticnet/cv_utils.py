"""
Cross-validation utilities for ElasticNet in sk-neuro.
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.utils import shuffle
from joblib import Parallel, delayed


def _fit_and_score_fold(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray
) -> tuple:
    """
    Fit estimator on a fold and return predictions and tuning info.
    Args:
        estimator: Estimator instance (must implement fit/predict/get_info).
        X: Feature matrix.
        y: Target matrix.
        train_index: Indices for training set.
        test_index: Indices for test set.
    Returns:
        Tuple of (fold_coef, fold_alpha, fold_l1, fold_y_pred, fold_test_idx, fold_score).
    """
    fold_test_idx = []
    fold_coef = []
    fold_alpha = []
    fold_l1 = []
    fold_y_pred = []
    fold_score = []
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    fold_y_pred.append(y_pred)
    mse_score = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    fold_score.append(mse_score)
    tuning_info = estimator.get_info()
    fold_coef.append(tuning_info['coef'])
    fold_alpha.append(tuning_info['best_alpha'])
    fold_l1.append(tuning_info['best_l1'])
    fold_test_idx.append(test_index)
    return fold_coef, fold_alpha, fold_l1, fold_y_pred, fold_test_idx, fold_score


def parallel_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    nfold: int = 10,
    n_jobs: int = -1,
    verbose: bool = False,
    pre_dispatch: str = '2*n_jobs'
) -> tuple:
    """
    Run parallel cross-validation for an ElasticNet estimator.
    Args:
        estimator: Estimator instance (must implement fit/predict/get_info).
        X: Feature matrix.
        y: Target matrix.
        nfold: Number of folds for KFold CV.
        n_jobs: Number of parallel jobs.
        verbose: Verbosity for joblib.
        pre_dispatch: Pre-dispatch parameter for joblib.
    Returns:
        Tuple of (fold_coef, fold_alpha, fold_l1, fold_score, sorted_y_pred).
    """
    cv = KFold(n_splits=nfold, shuffle=True)
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(delayed(_fit_and_score_fold)(clone(estimator), X, y, train_index, test_index)
                      for train_index, test_index in cv.split(X))
    zipped_results = list(zip(*results))
    fold_coef, fold_alpha, fold_l1, fold_y_pred, fold_test_idx, fold_score = zipped_results
    sorted_y_pred = np.empty((y.shape)) * np.nan
    if y.shape[1] > 1:
        for fold_i, fold_y_pred_i in zip(fold_test_idx, fold_y_pred):
            sorted_y_pred[fold_i] = np.array(fold_y_pred_i)
    else:
        for fold_i, fold_y_pred_i in zip(fold_test_idx, fold_y_pred):
            sorted_y_pred[fold_i] = np.array(fold_y_pred_i).T
    fold_coef_ = np.squeeze(np.vstack(fold_coef).T)
    fold_alpha_ = np.array(fold_alpha).reshape(-1)
    fold_l1_ = np.array(fold_l1).reshape(-1)
    fold_score_ = np.vstack(fold_score).T
    sorted_y_pred_ = sorted_y_pred
    return fold_coef_, fold_alpha_, fold_l1_, fold_score_, sorted_y_pred_


def repeated_parallel_cv(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    nfold: int = 10,
    n_repeats: int = 50,
    shuffle_y: bool = False
) -> dict:
    """
    Run repeated parallel cross-validation for an ElasticNet estimator.
    Args:
        estimator: Estimator instance (must implement fit/predict/get_info).
        X: Feature matrix.
        y: Target matrix.
        nfold: Number of folds for KFold CV.
        n_repeats: Number of repetitions.
        shuffle_y: Whether to shuffle y for permutation testing.
    Returns:
        results: Dictionary with all predictions, scores, and coefficients.
    """
    results = {}
    all_y_pred = []
    all_alpha = []
    all_l1 = []
    all_coef = []
    all_score = []
    seed0 = np.random.randint(1000, size=1)
    for rep_i in range(n_repeats):
        print(f'Repetition {rep_i+1}/{n_repeats}')
        if shuffle_y:
            seed1 = seed0 + np.random.randint(100, size=1)
            y2use = shuffle(y.copy(), random_state=int(seed1))
        else:
            y2use = y.copy()
        fold_coef, fold_alpha, fold_l1, fold_score, sorted_y_pred = parallel_cv(
            estimator, X, y2use, nfold=nfold, n_jobs=-1, verbose=False, pre_dispatch='2*n_jobs')
        all_y_pred.append(np.array(sorted_y_pred))
        all_alpha.append(np.array(fold_alpha))
        all_l1.append(np.array(fold_l1))
        all_coef.append(np.array(fold_coef))
        all_score.append(np.array(fold_score))
    results['all_y_pred'] = np.stack(all_y_pred, axis=-1)
    results['mean_y_pred'] = np.mean(np.array(all_y_pred), axis=0)
    results['all_alpha'] = np.array(all_alpha).T
    results['all_l1'] = np.array(all_l1).T
    results['all_coef'] = np.stack(all_coef, axis=-1)
    results['all_score'] = np.array(all_score).T
    results['true_y'] = y.copy()
    return results


def pearson_corr(y_pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Pearson correlation between predictions and true values.
    Args:
        y_pred: Predicted values.
        y: True values.
    Returns:
        r_value: Array of Pearson r values for each output.
        p_value: Array of p-values for each output.
    """
    import scipy.stats
    if np.ndim(y_pred) == 1:
        y_pred = y_pred[:, None]
    y_dim = y.shape[-1]
    r_value = np.empty((y.shape[-1])) * np.nan
    p_value = np.empty((y.shape[-1])) * np.nan
    for y_i in np.arange(y_dim):
        r_value[y_i], p_value[y_i] = scipy.stats.pearsonr(y_pred[:, y_i], y[:, y_i])
    return r_value, p_value
