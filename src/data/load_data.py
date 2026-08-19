"""
src/data/load_data.py
---------------------
Fungsi untuk memuat dan mempersiapkan dataset mentah mpg_indo.csv.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ── Konstanta ────────────────────────────────────────────────────────────────
RAW_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'mpg_indo.csv')
TARGET       = 'konsumsi_bbm'
ORIGIN_MAP   = {1: 'Amerika', 2: 'Eropa', 3: 'Asia'}
RANDOM_STATE = 42


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """
    Muat dataset mentah dan terapkan mapping awal.

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom asal_pabrikan sudah di-map ke label,
        jumlah_silinder dan tahun_rilis sebagai string (untuk encoding).
    """
    df = pd.read_csv(path)
    df['asal_pabrikan']   = df['asal_pabrikan'].map(ORIGIN_MAP)
    df['jumlah_silinder'] = df['jumlah_silinder'].astype(str)
    df['tahun_rilis']     = df['tahun_rilis'].astype(str)
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET):
    """
    Pisahkan fitur (X) dan target (y).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = RANDOM_STATE):
    """
    Bagi data menjadi train dan test set.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_and_split(path: str = RAW_PATH,
                   test_size: float = 0.2,
                   random_state: int = RANDOM_STATE):
    """
    Shortcut: load → split fitur/target → split train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    df = load_raw(path)
    X, y = split_features_target(df)
    return split_train_test(X, y, test_size=test_size, random_state=random_state)
