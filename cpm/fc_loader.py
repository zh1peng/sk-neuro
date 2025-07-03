import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from joblib import Parallel, delayed

def _load_and_flatten(idx, row, bids_root, feature_subpath, band, con_type, triu):
    """
    Worker for joblib: load one .mat, extract and flatten FC.
    Returns (idx, fc_flat) or (idx, None) on failure.
    """
    mat_path = bids_root / row["source"] / feature_subpath / row["eeg_feature_file"]
    try:
        mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        res = mat["res"]
        conn = getattr(getattr(res, band), con_type)
        return idx, conn[triu]
    except Exception:
        return idx, None
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from joblib import Parallel, delayed

def _load_and_flatten(idx, row, bids_root, feature_subpath, band, con_type, triu):
    """
    Worker for joblib: load one .mat, extract and flatten FC.
    Returns (idx, fc_flat) or (idx, None) on failure.
    """
    mat_path = bids_root / row["source"] / feature_subpath / row["eeg_feature_file"]
    try:
        mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        res = mat["res"]
        conn = getattr(getattr(res, band), con_type)
        return idx, conn[triu]
    except Exception:
        return idx, None

def extract_fc_for_cpm_parallel(df_list,
                                bids_root,
                                feature_subpath,
                                band,
                                con_type,
                                n_subjects=None,
                                n_jobs=-1,
                                verbose=5):
    """
    Parallel FC extractor using joblib.Parallel, with optional subsetting to
    the first `n_subjects` rows of df_list.

    Parameters
    ----------
    df_list : pd.DataFrame
        Must contain ['participant_id','source','eeg_feature_file'].
    bids_root : str or Path
    feature_subpath : str
    band : str
    con_type : str
    n_subjects : int or None
        If int, only the first `n_subjects` rows of df_list will be processed.
    n_jobs : int
        Number of parallel jobs for joblib.Parallel.
    verbose : int
        Verbosity level for joblib.

    Returns
    -------
    X : np.ndarray, shape (n_valid_subjects, n_edges)
    df_clean : pd.DataFrame
        Sub‐DataFrame of df_list (up to `n_subjects`) where loading succeeded.
    """
    bids_root = Path(bids_root)
    df = df_list.reset_index(drop=True)

    # optionally limit to top-k rows
    if isinstance(n_subjects, int):
        df = df.iloc[:n_subjects].copy()

    # Probe first file to get node count & triu indices
    first = df.iloc[0]
    sample = loadmat(
        bids_root / first["source"] / feature_subpath / first["eeg_feature_file"],
        struct_as_record=False, squeeze_me=True
    )["res"]
    n_nodes = getattr(getattr(sample, band), con_type).shape[0]
    triu = np.triu_indices(n_nodes, k=1)

    # Run parallel loading & flattening
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_load_and_flatten)(
            idx, df.iloc[idx], bids_root, feature_subpath, band, con_type, triu
        )
        for idx in range(len(df))
    )

    # Collect successes
    fc_list, valid_idxs = [], []
    for idx, fc_flat in results:
        if fc_flat is not None:
            fc_list.append(fc_flat)
            valid_idxs.append(idx)

    if not fc_list:
        raise RuntimeError("No connectivity matrices loaded. Check your inputs.")

    X = np.vstack(fc_list)
    df_clean = df.loc[valid_idxs].reset_index(drop=True)
    return X, df_clean


def extract_fc_for_cpm(df_list,
                       bids_root,
                       feature_subpath,
                       band,
                       con_type,
                       n_subjects=None):
    """
    Extracts and flattens the upper-triangle FC matrix for a given band & con_type,
    optionally limiting to the first n_subjects rows of df_list.

    Parameters
    ----------
    df_list : pd.DataFrame
        Must contain ['participant_id','source','eeg_feature_file'].
    bids_root : str or Path
    feature_subpath : str
    band : str
    con_type : str
    n_subjects : int or None
        If int, only the first n_subjects rows of df_list will be processed.

    Returns
    -------
    X : np.ndarray, shape (n_valid_subjects, n_edges)
    df_clean : pd.DataFrame
        Sub‐DataFrame of df_list (up to n_subjects) where loading succeeded.
    """
    bids_root = Path(bids_root)
    # limit rows if requested
    if isinstance(n_subjects, int):
        df_proc = df_list.iloc[:n_subjects].copy()
    else:
        df_proc = df_list

    fc_list = []
    valid_idx = []

    for idx, row in df_proc.iterrows():
        mat_path = bids_root / row["source"] / feature_subpath / row["eeg_feature_file"]
        print(f"{idx+1} / {len(df_proc)} : {mat_path}", end="\r")
        try:
            mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
            res = mat["res"]
            conn = getattr(getattr(res, band), con_type)

            # flatten upper triangle
            triu = np.triu_indices(conn.shape[0], k=1)
            fc_list.append(conn[triu])
            valid_idx.append(idx)

        except Exception as e:
            print(f"[WARN] {row['participant_id']}: {e}")

    if not fc_list:
        raise RuntimeError("No connectivity matrices loaded. Check inputs.")

    X = np.vstack(fc_list)
    df_clean = df_proc.loc[valid_idx].reset_index(drop=True)
    return X, df_clean
