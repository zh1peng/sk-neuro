"""
Interpretation utilities for ElasticNet results in sk-neuro
"""
import numpy as np
import pandas as pd


def load_cv_results(true_npy: str, null_npy: str) -> tuple[dict, dict]:
    """
    Load cross-validated results from .npy files for true and null models.
    Args:
        true_npy: Path to .npy file with true model results.
        null_npy: Path to .npy file with null/permuted model results.
    Returns:
        (true_result, null_result): Tuple of result dictionaries.
    """
    true_result = np.load(true_npy, allow_pickle=True).item()
    null_result = np.load(null_npy, allow_pickle=True).item()
    return true_result, null_result


def significant_betas(true_result: dict, null_result: dict, ci: float = 95) -> np.ndarray:
    """
    Identify significant betas outside the confidence interval of null betas.
    Args:
        true_result: Result dict from load_cv_results (true model).
        null_result: Result dict from load_cv_results (null/permuted model).
        ci: Confidence interval (default 95).
    Returns:
        sig_betas: Array of significant beta values (others set to 0).
    """
    true_betas = true_result['all_coef'].mean(-1).mean(-1)
    null_betas = null_result['all_coef'].reshape(null_result['all_coef'].shape[0], -1)
    sig_betas = np.zeros(np.size(true_betas))
    lower = (100 - ci) / 2
    upper = 100 - lower
    for idx, val in enumerate(true_betas):
        if val >= np.percentile(null_betas[idx, :], upper) or val <= np.percentile(null_betas[idx, :], lower):
            sig_betas[idx] = val
    return sig_betas


def interpret_betas_by_region(
    csv_file: str,
    sig_betas: np.ndarray,
    regions: list[str],
    conditions: list[str],
    mid_stuff: str
) -> tuple[dict, dict]:
    """
    Group significant betas by region and condition for interpretation and plotting.
    Args:
        csv_file: Path to CSV file with feature names.
        sig_betas: Array of significant beta values.
        regions: List of region names (e.g., ['F','C','P']).
        conditions: List of condition codes (e.g., ['13','23','16','26']).
        mid_stuff: Pattern between region and condition in feature names.
    Returns:
        beta_interpreter: Dict mapping region/condition to betas.
        data2plot: Dict mapping condition to matrix for plotting.
    """
    all_data = pd.read_csv(csv_file)
    feature_names = list(all_data.drop(columns=['y']))
    beta_interpreter = {}
    for region_i in regions:
        for cond_i in conditions:
            test_pattern = region_i + mid_stuff + cond_i
            pattern_idx = [i for i, s in enumerate(feature_names) if str(s).startswith(test_pattern)]
            pattern_beta = sig_betas[pattern_idx]
            beta_interpreter[f'{region_i}_{cond_i}'] = pattern_beta
            beta_interpreter[f'{region_i}_{cond_i}_idx2check'] = pattern_idx
    data2plot = {}
    for cond_i in conditions:
        cond_by_regions = []
        for region_i in regions:
            cond_by_regions.append(beta_interpreter[f'{region_i}_{cond_i}'])
        data2plot[f'con_{cond_i}'] = np.vstack(cond_by_regions).T
    return beta_interpreter, data2plot
