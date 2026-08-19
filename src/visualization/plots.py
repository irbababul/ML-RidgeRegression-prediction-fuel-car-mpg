"""
src/visualization/plots.py
---------------------------
Fungsi-fungsi plotting standar yang digunakan di seluruh notebook.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

FIG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports', 'figures')
COLORS  = sns.color_palette('muted')


def save_fig(name: str, fig_dir: str = FIG_DIR, dpi: int = 150):
    """Simpan figure aktif ke reports/figures/<name>.png."""
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name if name.endswith('.png') else name + '.png')
    plt.savefig(path, dpi=dpi, bbox_inches='tight')


def plot_actual_vs_pred(y_true, y_pred, name: str,
                        color_idx: int = 0, ax=None, show: bool = True):
    """
    Scatter plot aktual vs prediksi dengan garis ideal y = ŷ.

    Parameters
    ----------
    y_true, y_pred : array-like
        Nilai aktual dan prediksi dalam satuan asli (mpg).
    name : str
        Judul plot.
    color_idx : int
        Index warna dari palet COLORS.
    ax : matplotlib.axes.Axes, optional
        Axes yang akan digunakan. Jika None, buat figure baru.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(y_true, y_pred, alpha=0.6,
               color=COLORS[color_idx % len(COLORS)], edgecolors='none', s=40)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Ideal (y = ŷ)')
    ax.set_xlabel('Aktual (mpg)')
    ax.set_ylabel('Prediksi (mpg)')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    ax.set_title(f'{name}\nRMSE={rmse:.4f} | R²={r2:.4f}', fontweight='bold')
    ax.legend()

    if standalone and show:
        plt.tight_layout()
        plt.show()


def plot_residuals(y_true, y_pred, name: str,
                   color_idx: int = 1, ax=None, show: bool = True):
    """
    Residual plot: residual vs nilai prediksi.
    Pola acak di sekitar 0 menunjukkan model yang baik.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 5))

    residuals = np.array(y_true) - np.array(y_pred)
    ax.scatter(y_pred, residuals, alpha=0.6,
               color=COLORS[color_idx % len(COLORS)], edgecolors='none', s=40)
    ax.axhline(0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Prediksi (mpg)')
    ax.set_ylabel('Residual (Aktual − Prediksi)')
    ax.set_title(f'{name}\nResidual Plot', fontweight='bold')

    if standalone and show:
        plt.tight_layout()
        plt.show()


def plot_model_comparison(results_df, metric_cols=None, figname: str = None):
    """
    Bar chart perbandingan beberapa model berdasarkan metrik evaluasi.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame dengan nama model sebagai index dan kolom metrik.
    metric_cols : list, optional
        Kolom metrik yang akan diplot. Default: ['RMSE Test', 'MAE Test', 'R² Test'].
    figname : str, optional
        Jika diberikan, simpan figure ke reports/figures/<figname>.png.
    """
    if metric_cols is None:
        metric_cols = ['RMSE Test', 'MAE Test', 'R² Test']

    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    model_names = results_df.index.tolist()
    x = np.arange(len(model_names))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(model_names))]

    for ax, col in zip(axes, metric_cols):
        bars = ax.bar(x, results_df[col], color=bar_colors, edgecolor='gray')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
        ax.set_title(col, fontweight='bold')
        ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=8)
        if 'R²' in col:
            ax.set_ylim(0, 1.15)

    fig.suptitle('Perbandingan Performa Model', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if figname:
        save_fig(figname)
    plt.show()


def plot_train_test_gap(results_df, figname: str = None):
    """
    Bar chart RMSE Train vs Test untuk mendeteksi overfitting.

    Gap besar (RMSE Test >> RMSE Train) mengindikasikan overfitting.
    """
    model_names = results_df.index.tolist()
    x = np.arange(len(model_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, results_df['RMSE Train'], w,
                label='RMSE Train', color=COLORS[0], edgecolor='gray')
    b2 = ax.bar(x + w/2, results_df['RMSE Test'],  w,
                label='RMSE Test',  color=COLORS[1], edgecolor='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.set_ylabel('RMSE (mpg)')
    ax.set_title('RMSE Train vs Test — Deteksi Overfitting', fontweight='bold')
    ax.legend()
    ax.bar_label(b1, fmt='%.3f', padding=2, fontsize=8)
    ax.bar_label(b2, fmt='%.3f', padding=2, fontsize=8)
    plt.tight_layout()

    if figname:
        save_fig(figname)
    plt.show()


def plot_feature_importance(coef_series, title: str = 'Koefisien Model',
                             figname: str = None):
    """
    Horizontal bar chart koefisien model.
    Positif = koefisien positif, negatif = koefisien negatif.
    Koefisien = 0 (Lasso) berwarna abu-abu.
    """
    bar_colors = ['#d9534f' if v < 0 else '#5cb85c' if v > 0 else '#cccccc'
                  for v in coef_series.values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(coef_series) * 0.45)))
    ax.barh(coef_series.index, coef_series.values,
            color=bar_colors, edgecolor='gray')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Nilai Koefisien')
    plt.tight_layout()

    if figname:
        save_fig(figname)
    plt.show()


def plot_fe_comparison(results_df, baseline_key: str = None, figname: str = None):
    """
    Bar chart perbandingan eksperimen feature engineering,
    termasuk kolom delta RMSE vs baseline.

    Parameters
    ----------
    results_df : pd.DataFrame
        Harus memiliki kolom 'RMSE Test' dan 'Δ RMSE Test'.
    baseline_key : str, optional
        Nama baseline di index results_df untuk garis referensi.
    """
    exp_names   = results_df.index.tolist()
    x           = np.arange(len(exp_names))
    bar_colors  = [COLORS[i % len(COLORS)] for i in range(len(exp_names))]
    baseline_rmse = results_df.loc[baseline_key, 'RMSE Test'] if baseline_key else None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: RMSE Test
    bars = axes[0].bar(x, results_df['RMSE Test'], color=bar_colors, edgecolor='gray')
    if baseline_rmse is not None:
        axes[0].axhline(baseline_rmse, color='red', linestyle='--', linewidth=1.5,
                        label=f'Baseline = {baseline_rmse:.4f}')
        axes[0].legend()
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(exp_names, rotation=25, ha='right', fontsize=9)
    axes[0].set_ylabel('RMSE Test (mpg)')
    axes[0].set_title('RMSE Test per Eksperimen', fontweight='bold')
    axes[0].bar_label(bars, fmt='%.4f', padding=2, fontsize=8)

    # Plot 2: Delta RMSE
    delta_vals   = results_df['Δ RMSE Test'].values
    delta_colors = ['#5cb85c' if v < 0 else '#d9534f' if v > 0 else '#aaaaaa'
                    for v in delta_vals]
    bars2 = axes[1].bar(x, delta_vals, color=delta_colors, edgecolor='gray')
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(exp_names, rotation=25, ha='right', fontsize=9)
    axes[1].set_ylabel('Δ RMSE Test vs Baseline (mpg)')
    axes[1].set_title('Perubahan RMSE vs Baseline\n(hijau = lebih baik)', fontweight='bold')
    axes[1].bar_label(bars2, fmt='%+.4f', padding=2, fontsize=8)

    fig.suptitle('Perbandingan Eksperimen Feature Engineering',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if figname:
        save_fig(figname)
    plt.show()
