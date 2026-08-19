"""
app.py — Streamlit App: Prediksi Konsumsi BBM Mobil
Jalankan: streamlit run app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st

# ── Konfigurasi path ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'ridge_final.pkl')
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'mpg_indo.csv')


# ── Load model & data (cache agar tidak reload tiap interaksi) ───────────────
@st.cache_resource
def load_model():
    """Load the pre-trained model saved with scikit-learn 1.6.*.
    scikit-learn 1.7 removed some private helpers (e.g. _RemainderColsList)
    that are referenced in the pickled pipeline.  To keep backward
    compatibility we patch a minimal stub of the missing class before
    calling ``joblib.load``.  This avoids having to pin the exact
    scikit-learn version or rebuild the model.
    """
    # --- Backward-compat patch -------------------------------------------------
    try:
        from sklearn.compose import _column_transformer as _ct  # pylint: disable=import-error
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                """Minimal replacement for the removed private helper."""
                pass
            _ct._RemainderColsList = _RemainderColsList  # type: ignore[attr-defined]
    except Exception:
        # If the internal path has changed we silently ignore; the load may
        # still succeed or raise a clear error that the user can act on.
        pass
    # -------------------------------------------------------------------------
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['asal_pabrikan'] = df['asal_pabrikan'].map({1: 'Amerika', 2: 'Eropa', 3: 'Asia'})
    return df


# ── Helper ───────────────────────────────────────────────────────────────────
def make_input_df(jumlah_silinder, kekuatan_mesin, berat_mobil,
                  akselerasi, tahun_rilis, asal_pabrikan):
    """Buat DataFrame input untuk prediksi (sesuai format training)."""
    return pd.DataFrame([{
        'jumlah_silinder': str(int(jumlah_silinder)),
        'kekuatan_mesin' : float(kekuatan_mesin),
        'berat_mobil'    : float(berat_mobil),
        'akselerasi'     : float(akselerasi),
        'tahun_rilis'    : str(int(tahun_rilis)),
        'asal_pabrikan'  : asal_pabrikan,
    }])


def bbm_category(mpg):
    """Klasifikasi konsumsi BBM."""
    if mpg >= 35:
        return '🟢 Sangat Irit', '#2ecc71'
    elif mpg >= 25:
        return '🟡 Irit', '#f1c40f'
    elif mpg >= 15:
        return '🟠 Sedang', '#e67e22'
    else:
        return '🔴 Boros', '#e74c3c'


# ════════════════════════════════════════════════════════════════════════════
# LAYOUT UTAMA
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title='Prediksi Konsumsi BBM Mobil',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='expanded'
)

model = load_model()
df    = load_data()

# ── Header ───────────────────────────────────────────────────────────────────
st.title('🚗 Prediksi Konsumsi BBM Mobil')
st.markdown(
    'Model **Ridge Regression** yang dilatih pada dataset Auto MPG. '
    'Masukkan spesifikasi kendaraan di panel kiri untuk mendapatkan estimasi konsumsi BBM.'
)
st.divider()

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Input Spesifikasi Kendaraan
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header('⚙️ Spesifikasi Kendaraan')

    jumlah_silinder = st.selectbox(
        'Jumlah Silinder',
        options=[3, 4, 5, 6, 8],
        index=1,
        help='Jumlah silinder mesin kendaraan'
    )

    kekuatan_mesin = st.slider(
        'Kekuatan Mesin (HP)',
        min_value=40, max_value=250, value=100, step=5,
        help='Tenaga kuda (horsepower) mesin'
    )

    berat_mobil = st.slider(
        'Berat Mobil (lbs)',
        min_value=1500, max_value=5500, value=2800, step=50,
        help='Berat kendaraan dalam pounds (1 kg ≈ 2.2 lbs)'
    )

    akselerasi = st.slider(
        'Akselerasi 0–60 mph (detik)',
        min_value=8.0, max_value=25.0, value=15.0, step=0.5,
        help='Waktu yang dibutuhkan untuk mencapai 60 mph dari posisi diam'
    )

    tahun_rilis = st.selectbox(
        'Tahun Rilis',
        options=list(range(70, 83)),
        index=6,
        format_func=lambda x: f'19{x}',
        help='Tahun produksi kendaraan (19xx)'
    )

    asal_pabrikan = st.selectbox(
        'Asal Pabrikan',
        options=['Amerika', 'Eropa', 'Asia'],
        index=0,
        help='Negara asal produsen kendaraan'
    )

    st.divider()
    predict_btn = st.button('🔮 Prediksi Sekarang', type='primary', use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# MAIN — Hasil Prediksi
# ════════════════════════════════════════════════════════════════════════════

col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.subheader('📋 Spesifikasi Input')

    spec_df = pd.DataFrame({
        'Spesifikasi': [
            'Jumlah Silinder', 'Kekuatan Mesin', 'Berat Mobil',
            'Akselerasi', 'Tahun Rilis', 'Asal Pabrikan'
        ],
        'Nilai': [
            f'{jumlah_silinder} silinder',
            f'{kekuatan_mesin} HP',
            f'{berat_mobil:,} lbs ({berat_mobil/2.205:.0f} kg)',
            f'{akselerasi} detik',
            f'19{tahun_rilis}',
            asal_pabrikan
        ]
    })
    st.dataframe(spec_df, hide_index=True, use_container_width=True)

with col2:
    st.subheader('🎯 Hasil Prediksi')

    if predict_btn:
        X_input = make_input_df(
            jumlah_silinder, kekuatan_mesin, berat_mobil,
            akselerasi, tahun_rilis, asal_pabrikan
        )

        try:
            pred_mpg = model.predict(X_input)[0]
            pred_lkm = 235.21 / pred_mpg  # konversi mpg → L/100km

            label, color = bbm_category(pred_mpg)

            st.markdown(
                f"""
                <div style="
                    background-color: {color}22;
                    border-left: 5px solid {color};
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 15px;
                ">
                    <h2 style="color: {color}; margin: 0;">{pred_mpg:.2f} mpg</h2>
                    <p style="font-size: 1.1rem; margin: 5px 0 0 0;">
                        ≈ {pred_lkm:.1f} L/100 km &nbsp;|&nbsp; {label}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Gauge-style progress bar
            max_mpg = 50
            pct = min(pred_mpg / max_mpg, 1.0)
            st.markdown('**Posisi relatif terhadap rentang dataset (9–47 mpg):**')
            st.progress(pct)
            st.caption(f'Semakin ke kanan = semakin irit ({pred_mpg:.1f} dari maks ~{max_mpg} mpg)')

        except Exception as e:
            st.error(f'Error prediksi: {e}')
    else:
        st.info('👈 Isi spesifikasi di panel kiri dan klik **Prediksi Sekarang**.')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Analisis & Konteks
# ════════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader('📊 Konteks: Posisi Prediksi vs Dataset')

# Ambil nilai prediksi dari session (agar chart muncul otomatis setelah predict)
if predict_btn:
    try:
        X_input = make_input_df(
            jumlah_silinder, kekuatan_mesin, berat_mobil,
            akselerasi, tahun_rilis, asal_pabrikan
        )
        pred_val = model.predict(X_input)[0]

        col3, col4 = st.columns([1, 1], gap='large')

        with col3:
            # Histogram distribusi target + garis prediksi
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df['konsumsi_bbm'], bins=25, color='#3498db', alpha=0.7,
                    edgecolor='white', label='Distribusi data historis')
            ax.axvline(pred_val, color='#e74c3c', linewidth=2.5, linestyle='--',
                       label=f'Prediksi kamu: {pred_val:.2f} mpg')
            ax.axvline(df['konsumsi_bbm'].mean(), color='#2ecc71', linewidth=1.5,
                       linestyle=':', label=f'Rata-rata dataset: {df["konsumsi_bbm"].mean():.2f} mpg')
            ax.set_xlabel('Konsumsi BBM (mpg)')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Konsumsi BBM Dataset', fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col4:
            # Scatter: berat mobil vs BBM + titik prediksi
            fig, ax = plt.subplots(figsize=(7, 4))
            scatter = ax.scatter(
                df['berat_mobil'], df['konsumsi_bbm'],
                c=df['konsumsi_bbm'], cmap='RdYlGn',
                alpha=0.5, s=25, edgecolors='none'
            )
            ax.scatter(berat_mobil, pred_val, color='#e74c3c', s=200,
                       zorder=5, marker='*', label=f'Input kamu ({pred_val:.1f} mpg)')
            plt.colorbar(scatter, ax=ax, label='mpg')
            ax.set_xlabel('Berat Mobil (lbs)')
            ax.set_ylabel('Konsumsi BBM (mpg)')
            ax.set_title('Berat Mobil vs Konsumsi BBM', fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Statistik perbandingan
        st.markdown('**Perbandingan dengan dataset historis:**')
        pct_rank = (df['konsumsi_bbm'] < pred_val).mean() * 100
        comp_cols = st.columns(4)
        comp_cols[0].metric('Prediksi', f'{pred_val:.2f} mpg')
        comp_cols[1].metric('Rata-rata dataset', f'{df["konsumsi_bbm"].mean():.2f} mpg',
                            delta=f'{pred_val - df["konsumsi_bbm"].mean():.2f} mpg')
        comp_cols[2].metric('Median dataset', f'{df["konsumsi_bbm"].median():.2f} mpg',
                            delta=f'{pred_val - df["konsumsi_bbm"].median():.2f} mpg')
        comp_cols[3].metric('Persentil',
                            f'{pct_rank:.0f}%',
                            help=f'Lebih irit dari {pct_rank:.0f}% kendaraan di dataset')

    except Exception as e:
        st.error(f'Error visualisasi: {e}')
else:
    st.caption('Visualisasi konteks akan muncul setelah klik Prediksi.')

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Info Model
# ════════════════════════════════════════════════════════════════════════════

st.divider()

with st.expander('ℹ️ Tentang Model'):
    st.markdown("""
    **Algoritma:** Ridge Regression (regularisasi L2)

    **Performa pada data uji:**
    | Metrik | Nilai |
    |---|---|
    | RMSE | 2.3552 mpg |
    | MAE | ~1.77 mpg |
    | R² | 0.8968 |

    **Preprocessing pipeline:**
    - Missing value: Median imputer (`kekuatan_mesin`)
    - Encoding: Ordinal (`jumlah_silinder`, `tahun_rilis`) + One-Hot (`asal_pabrikan`)
    - Scaling: Yeo-Johnson Power Transform (sudah termasuk standardisasi)
    - Target transform: Yeo-Johnson (inverse transform otomatis saat prediksi)

    **Fitur yang digunakan:** `kekuatan_mesin`, `berat_mobil`, `akselerasi`, `jumlah_silinder`, `tahun_rilis`, `asal_pabrikan`
    (`kapasitas_mesin` di-drop — redundan dengan `kekuatan_mesin`, meningkatkan performa)

    **Dataset:** 398 kendaraan produksi 1970–1982 dari Amerika, Eropa, dan Asia.
    Prediksi paling akurat untuk kendaraan dengan spesifikasi dalam rentang tersebut.
    """)

with st.expander('📖 Panduan Konversi Satuan'):
    st.markdown("""
    | Satuan | Konversi |
    |---|---|
    | 1 mpg | ≈ 0.425 km/L atau 235.21 / mpg L/100km |
    | 1 lbs | ≈ 0.454 kg |
    | 1 HP (horsepower) | ≈ 0.746 kW |

    **Contoh interpretasi:**
    - 25 mpg ≈ 10.6 km/L ≈ 9.4 L/100km (cukup irit)
    - 15 mpg ≈ 6.4 km/L ≈ 15.7 L/100km (agak boros)
    """)
