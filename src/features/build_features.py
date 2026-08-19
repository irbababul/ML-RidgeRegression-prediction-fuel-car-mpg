"""
src/features/build_features.py
--------------------------------
Konfigurasi kolom, custom transformers, dan builder preprocessor sklearn.
"""

import pandas as pd
import numpy as np
from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.pipeline         import Pipeline
from sklearn.compose          import ColumnTransformer
from sklearn.impute           import SimpleImputer
from sklearn.preprocessing    import (
    OrdinalEncoder, OneHotEncoder, PowerTransformer
)


# ── Konfigurasi kolom ────────────────────────────────────────────────────────
NUM_COLS     = ['kapasitas_mesin', 'kekuatan_mesin', 'berat_mobil', 'akselerasi']
ORD_SILINDER = ['jumlah_silinder']
ORD_TAHUN    = ['tahun_rilis']
OHE_COLS     = ['asal_pabrikan']

SILINDER_CATS = [['3', '4', '5', '6', '8']]

# Dibuat dinamis saat build (bisa di-override jika perlu)
_TAHUN_CATS_CACHE = None


def get_tahun_cats(df=None):
    """
    Kembalikan urutan kategori tahun_rilis.
    Jika df diberikan, inferensi dari data; jika tidak, gunakan default 70–82.
    """
    global _TAHUN_CATS_CACHE
    if df is not None:
        years = sorted(df['tahun_rilis'].astype(int).unique())
        _TAHUN_CATS_CACHE = [[str(y) for y in years]]
    elif _TAHUN_CATS_CACHE is None:
        _TAHUN_CATS_CACHE = [[str(y) for y in range(70, 83)]]
    return _TAHUN_CATS_CACHE


MAX_TAHUN = 82  # tahun rilis terbaru di dataset (1982)


# ── Custom Transformers ──────────────────────────────────────────────────────

class PowerToWeightAdder(BaseEstimator, TransformerMixin):
    """
    Menambahkan fitur 'power_to_weight' = kekuatan_mesin / berat_mobil.
    Fitur ini menangkap rasio tenaga-per-berat yang lebih langsung
    mencerminkan efisiensi bahan bakar.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        hp = pd.to_numeric(X['kekuatan_mesin'], errors='coerce')
        wt = pd.to_numeric(X['berat_mobil'],    errors='coerce')
        X['power_to_weight'] = hp / wt
        return X


class CarAgeAdder(BaseEstimator, TransformerMixin):
    """
    Mengubah kolom 'tahun_rilis' (string) menjadi 'umur_mobil' (int).
    umur_mobil = max_tahun - tahun_rilis
    Kolom tahun_rilis asli di-drop setelah transformasi.
    """

    def __init__(self, max_tahun: int = MAX_TAHUN):
        self.max_tahun = max_tahun

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['umur_mobil'] = self.max_tahun - X['tahun_rilis'].astype(int)
        X = X.drop(columns=['tahun_rilis'])
        return X


class FullFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Gabungan semua feature engineering terbaik (Exp A + B + C):
      1. Tambah power_to_weight
      2. Ubah tahun_rilis → umur_mobil, drop tahun_rilis
      3. Drop kapasitas_mesin
    """

    def __init__(self, max_tahun: int = MAX_TAHUN):
        self.max_tahun = max_tahun

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        hp = pd.to_numeric(X['kekuatan_mesin'], errors='coerce')
        wt = pd.to_numeric(X['berat_mobil'],    errors='coerce')
        X['power_to_weight'] = hp / wt
        X['umur_mobil'] = self.max_tahun - X['tahun_rilis'].astype(int)
        X = X.drop(columns=['tahun_rilis', 'kapasitas_mesin'])
        return X


# ── Builder Preprocessor ─────────────────────────────────────────────────────

def build_preprocessor(num_cols=None, tahun_cats=None):
    """
    Buat ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    num_cols : list, optional
        Daftar kolom numerik. Default: NUM_COLS (semua fitur numerik).
    tahun_cats : list of list, optional
        Urutan kategori tahun_rilis. Default: get_tahun_cats().

    Returns
    -------
    ColumnTransformer
    """
    if num_cols is None:
        num_cols = NUM_COLS
    if tahun_cats is None:
        tahun_cats = get_tahun_cats()

    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  PowerTransformer(method='yeo-johnson', standardize=True))
        ]), num_cols),
        ('ord_sil', OrdinalEncoder(
            categories=SILINDER_CATS,
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ORD_SILINDER),
        ('ord_thn', OrdinalEncoder(
            categories=tahun_cats,
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ORD_TAHUN),
        ('ohe', OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        ), OHE_COLS),
    ], remainder='drop')


def build_preprocessor_no_tahun(num_cols_with_age=None, tahun_cats=None):
    """
    Preprocessor untuk pipeline yang sudah mengubah tahun_rilis → umur_mobil
    (Exp C & D). Tidak ada ord_thn karena kolom tahun_rilis sudah di-drop.

    Parameters
    ----------
    num_cols_with_age : list, optional
        Kolom numerik yang sudah include 'umur_mobil'.
    """
    if num_cols_with_age is None:
        num_cols_with_age = ['kekuatan_mesin', 'berat_mobil',
                             'akselerasi', 'umur_mobil']

    return ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler',  PowerTransformer(method='yeo-johnson', standardize=True))
        ]), num_cols_with_age),
        ('ord_sil', OrdinalEncoder(
            categories=SILINDER_CATS,
            handle_unknown='use_encoded_value', unknown_value=-1
        ), ORD_SILINDER),
        ('ohe', OneHotEncoder(
            drop='first', sparse_output=False, handle_unknown='ignore'
        ), OHE_COLS),
    ], remainder='drop')


def build_target_transformer():
    """Yeo-Johnson transformer untuk kolom target."""
    return PowerTransformer(method='yeo-johnson', standardize=True)
