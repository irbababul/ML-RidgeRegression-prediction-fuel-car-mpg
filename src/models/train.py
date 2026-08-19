"""
src/models/train.py
--------------------
Fungsi untuk membangun pipeline model, melatih, mengevaluasi,
menyimpan, dan memuat model.
"""

import os
import numpy as np
import joblib
from sklearn.pipeline         import Pipeline
from sklearn.compose          import TransformedTargetRegressor
from sklearn.model_selection  import cross_val_score, KFold
from sklearn.linear_model     import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing    import PolynomialFeatures
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score

from ..features.build_features import (
    build_preprocessor, build_target_transformer,
    NUM_COLS, ORD_SILINDER, ORD_TAHUN, OHE_COLS,
)

RANDOM_STATE = 42
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', '..', 'models')


# ── Builder Pipeline ─────────────────────────────────────────────────────────

def build_linear_pipeline(preprocessor=None):
    """Pipeline: Preprocessing + Linear Regression."""
    if preprocessor is None:
        preprocessor = build_preprocessor()
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor   = LinearRegression(),
            transformer = build_target_transformer()
        ))
    ])


def build_ridge_pipeline(preprocessor=None, alphas=None):
    """
    Pipeline: Preprocessing + Ridge Regression (dengan RidgeCV).

    Parameters
    ----------
    alphas : array-like, optional
        Range alpha untuk RidgeCV. Default: logspace(-3, 3, 100).
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()
    if alphas is None:
        alphas = np.logspace(-3, 3, 100)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor   = RidgeCV(alphas=alphas, cv=5),
            transformer = build_target_transformer()
        ))
    ])


def build_lasso_pipeline(preprocessor=None, alphas=None):
    """Pipeline: Preprocessing + Lasso Regression (dengan LassoCV)."""
    if preprocessor is None:
        preprocessor = build_preprocessor()
    if alphas is None:
        alphas = np.logspace(-4, 1, 100)
    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', TransformedTargetRegressor(
            regressor   = LassoCV(alphas=alphas, cv=5,
                                  max_iter=10000, random_state=RANDOM_STATE),
            transformer = build_target_transformer()
        ))
    ])


def build_polynomial_pipeline(preprocessor=None, degree=2):
    """Pipeline: Preprocessing + Polynomial Features + Linear Regression."""
    if preprocessor is None:
        preprocessor = build_preprocessor()
    return Pipeline([
        ('preprocessor', preprocessor),
        ('poly',         PolynomialFeatures(degree=degree, include_bias=False)),
        ('model', TransformedTargetRegressor(
            regressor   = LinearRegression(),
            transformer = build_target_transformer()
        ))
    ])


# ── Evaluasi ─────────────────────────────────────────────────────────────────

def evaluate(name: str, pipeline, X_train, y_train, X_test, y_test,
             cv_folds: int = 5) -> dict:
    """
    Fit pipeline dan hitung metrik evaluasi.

    Metrik yang dihitung:
    - RMSE Train / Test (dalam satuan mpg — sudah di-inverse transform)
    - MAE Test
    - R² Test
    - CV RMSE (5-fold cross-validation pada training set)

    Returns
    -------
    dict
        Dictionary berisi semua metrik dan nama model.
    """
    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test  = pipeline.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mae_test   = mean_absolute_error(y_test, y_pred_test)
    r2_test    = r2_score(y_test, y_pred_test)

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                scoring='neg_mean_squared_error', cv=cv)
    cv_rmse = np.sqrt(-cv_scores).mean()

    return {
        'Model'     : name,
        'RMSE Train': round(rmse_train, 4),
        'RMSE Test' : round(rmse_test,  4),
        'MAE Test'  : round(mae_test,   4),
        'R² Test'   : round(r2_test,    4),
        'CV RMSE'   : round(cv_rmse,    4),
        '_pipeline' : pipeline,
        '_y_pred'   : y_pred_test,
    }


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(pipeline, filename: str, models_dir: str = MODELS_DIR):
    """
    Simpan pipeline ke folder models/.

    Parameters
    ----------
    filename : str
        Nama file, misal 'ridge_final.pkl'.
    """
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    joblib.dump(pipeline, path)
    print(f'Model disimpan: {path}')
    return path


def load_model(filename: str, models_dir: str = MODELS_DIR):
    """
    Load pipeline dari folder models/.

    Parameters
    ----------
    filename : str
        Nama file, misal 'ridge_final.pkl'.
    """
    path = os.path.join(models_dir, filename)
    pipeline = joblib.load(path)
    print(f'Model dimuat: {path}')
    return pipeline


def predict(pipeline, X) -> np.ndarray:
    """
    Prediksi menggunakan pipeline yang sudah di-fit.
    Output sudah dalam satuan asli (mpg) berkat TransformedTargetRegressor.
    """
    return pipeline.predict(X)
