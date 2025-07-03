"""
I/O and batch utilities for ElasticNet in sk-neuro.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from .cv_utils import pearson_corr, repeated_parallel_cv


def run_cv_on_csv(data_csv: str) -> tuple:
    """
    Run cross-validated prediction on a single CSV file.
    Returns r, p, and elapsed time as a tuple.
    Args:
        data_csv: Path to CSV file.
    Returns:
        Tuple (r_value, p_value, elapsed_time_str)
    """
    csv_name, ext = os.path.splitext(data_csv)
    start_time = time.time()
    all_df = pd.read_csv(data_csv)
    flag_y = [col for col in all_df.columns if 'y_' in col]
    y = np.array(all_df[flag_y])
    fs_df = all_df.drop(columns=flag_y)
    flag_col = np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs = np.array(fs_df)
    from .estimator import GlmnetElasticNetCV
    clf = GlmnetElasticNetCV(not2preprocess=flag_col)
    y_pred = cross_val_predict(clf, fs, y, cv=10, n_jobs=-1)
    r_value, p_value = pearson_corr(y_pred, y)
    e = int(time.time() - start_time)
    e_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
    with open('appending_results.txt', 'a+') as fo:
        fo.write(f'\n filename---{csv_name} r---: {r_value} p---: {p_value} time--: {e_time} \n')
    return r_value, p_value, e_time


def save_cv_diagnostic_plot(data_csv: str) -> None:
    """
    Fit GlmnetElasticNetCV and save diagnostic plot for a single CSV file.
    Args:
        data_csv: Path to CSV file.
    """
    csv_name, ext = os.path.splitext(data_csv)
    all_df = pd.read_csv(data_csv)
    flag_y = [col for col in all_df.columns if 'y_' in col]
    y = np.array(all_df[flag_y])
    fs_df = all_df.drop(columns=flag_y)
    flag_col = np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs = np.array(fs_df)
    from .estimator import GlmnetElasticNetCV
    clf = GlmnetElasticNetCV(not2preprocess=flag_col)
    clf.fit(fs, y)
    clf.diagnostic_plot1(csv_name + '1.png')


def run_cv_on_directory(data_path: str) -> pd.DataFrame:
    """
    Run run_cv_on_csv on all CSV files in a directory.
    Returns a DataFrame with r, p, and time for each file.
    Args:
        data_path: Path to directory containing CSV files.
    Returns:
        DataFrame with r, p, and time for each file.
    """
    os.chdir(data_path)
    filenames = os.listdir(data_path)
    csv2test = [filename for filename in filenames if filename.endswith('.csv')]
    all_r = []
    all_p = []
    all_time = []
    for csv in csv2test:
        tmp_r, tmp_p, tmp_time = run_cv_on_csv(csv)
        all_r.append(str(tmp_r))
        all_p.append(str(tmp_p))
        all_time.append(str(tmp_time))
    df = pd.DataFrame({'r value': all_r, 'p value': all_p, 'time': all_time}, index=csv2test)
    return df


def run_repeated_cv_on_csv(data_csv: str, n_repeats: int = 50, shuffle_y: bool = False) -> dict:
    """
    Run repeated cross-validation on a single CSV file.
    Returns the results dictionary.
    Args:
        data_csv: Path to CSV file.
        n_repeats: Number of repetitions.
        shuffle_y: Whether to shuffle y for permutation testing.
    Returns:
        results: Dictionary with all predictions, scores, and coefficients.
    """
    csv_name, ext = os.path.splitext(data_csv)
    all_df = pd.read_csv(data_csv)
    flag_y = [col for col in all_df.columns if 'y_' in col]
    y = np.array(all_df[flag_y])
    fs_df = all_df.drop(columns=flag_y)
    flag_col = np.array([idx for idx, col in enumerate(fs_df.columns) if 'flag' in col])
    fs = np.array(fs_df)
    from .estimator import GlmnetElasticNetCV
    from .cv_utils import repeated_parallel_cv
    clf = GlmnetElasticNetCV(not2preprocess=flag_col)
    results = repeated_parallel_cv(clf, fs, y, nfold=10, n_repeats=n_repeats, shuffle_y=shuffle_y)
    return results
