# 📊Project Report — Prediksi Konsumsi BBM Mobil
**👤kontributor: Muhammad Irbabul Salas**
---

**📂Dataset:** Auto MPG (versi Bahasa Indonesia) — `mpg_indo.csv`  
**🎯Tipe Problem:** Supervised Learning — Regresi  
**🏆Model Final:** Ridge Regression  

---

## 1. 📌Gambaran Proyek

Proyek ini bertujuan membangun model machine learning untuk memprediksi konsumsi bahan bakar minyak (BBM) kendaraan dalam satuan **miles per gallon (mpg)**. Dataset berisi spesifikasi teknis 398 kendaraan dari tiga wilayah produksi: Amerika, Eropa, dan Asia, diproduksi antara tahun 1970–1982.

### 🏗️Struktur Proyek

```
ML_autompg/
├── 📂data/
│   ├── 📄raw/           ← mpg_indo.csv (data asli, tidak diubah)
│   ├── 📄interim/       ← data_encoded.csv (setelah encoding, sebelum scaling)
│   └── 📄processed/     ← data_preprocessed.csv (siap modeling)
├── 📂models/
│   └── 📦ridge_final.pkl
├── 📂notebooks/
│   ├── 📓eda.ipynb
│   ├── 📓preprocessing.ipynb
│   └── 📓feature_engineering.ipynb
├── 📂reports/
│   └── 🖼️figures/       ← semua visualisasi tersimpan di sini
├── 📂src/
│   ├── ⚙️data/          ← load_data.py
│   ├── ⚙️features/      ← build_features.py (transformers + preprocessor)
│   ├── ⚙️models/        ← train.py (pipeline builder + evaluasi)
│   └── ⚙️visualization/ ← plots.py
└── 📂docs/
    └── 📄project_report.md (dokumen ini)
```

---

## 2. 📑Dataset

| Kolom | Tipe | Keterangan |
|---|---|---|
| `kapasitas_mesin` | Numerik | Kapasitas silinder (cubic inches) |
| `jumlah_silinder` | Kategorikal Ordinal | Jumlah silinder: 3, 4, 5, 6, 8 |
| `kekuatan_mesin` | Numerik | Tenaga kuda (horsepower) |
| `berat_mobil` | Numerik | Berat kendaraan (lbs) |
| `akselerasi` | Numerik | Waktu 0–60 mph (detik) |
| `tahun_rilis` | Kategorikal Ordinal | Tahun produksi (70–82) |
| `asal_pabrikan` | Kategorikal Nominal | Amerika / Eropa / Asia |
| `konsumsi_bbm` | **Target** | Konsumsi BBM (mpg) |

**⚠️Missing value:** 6 nilai pada `kekuatan_mesin` (~1.5%)

---

## 3. 🔍Exploratory Data Analysis (EDA)

### Temuan Utama

**Distribusi Target**
- `konsumsi_bbm` berdistribusi right-skewed (skewness ≈ +0.46)
- Rentang: 9–46.6 mpg, Mean: 23.5 mpg
- Transformasi Yeo-Johnson diperlukan untuk model linear

**Korelasi dengan Target**
| Fitur | Korelasi | Arah |
|---|---|---|
| `berat_mobil` | ~−0.83 | Negatif kuat |
| `kapasitas_mesin` | ~−0.81 | Negatif kuat |
| `kekuatan_mesin` | ~−0.78 | Negatif kuat |
| `akselerasi` | ~+0.42 | Positif moderat |
| `tahun_rilis` | ~+0.58 | Positif (mobil baru lebih irit) |

**Multikolinearitas**
- `kapasitas_mesin` dan `kekuatan_mesin`: r ≈ 0.85+ → kandidat untuk ditangani

**Perbedaan antar Kelompok**
- Mobil **Asia** rata-rata paling irit, diikuti Eropa, kemudian Amerika
- Silinder lebih sedikit → konsumsi BBM lebih irit secara signifikan

**Outlier**
- Terdeteksi terutama di `kekuatan_mesin` dan `akselerasi`
- Setelah Yeo-Johnson, outlier tidak signifikan mempengaruhi model

---

## 4. 🛠️Preprocessing Pipeline

### Strategi

```
Raw Data
  ↓ Median Imputer (kekuatan_mesin)
  ↓ OrdinalEncoder (jumlah_silinder: 3<4<5<6<8)
  ↓ OrdinalEncoder (tahun_rilis: 70<71<...<82)
  ↓ OneHotEncoder (asal_pabrikan: drop=first)
  ↓ Yeo-Johnson PowerTransformer (numerik, standardize=True)
  ↓ Yeo-Johnson PowerTransformer (target konsumsi_bbm)
Processed Data
```

### Keputusan Teknis

| Keputusan | Alasan |
|---|---|
| Split sebelum preprocessing | Cegah data leakage — parameter fit hanya dari training data |
| Median imputer | Robust terhadap outlier di `kekuatan_mesin` |
| Yeo-Johnson (bukan Log/Box-Cox) | Bisa handle nilai positif maupun negatif |
| `standardize=True` di PowerTransformer | Sudah mencakup standardisasi — tidak perlu StandardScaler terpisah |
| Transform target | Asumsi residual normal untuk model linear |
| Pipeline sklearn | Preprocessing + model = 1 objek, CV tidak leaky |

### Implementasi

Seluruh preprocessing diimplementasikan sebagai `sklearn.Pipeline` dengan `TransformedTargetRegressor` sehingga:
- `pipeline.fit(X_train, y_train)` → fit preprocessing + model
- `pipeline.predict(X_test)` → output langsung dalam satuan mpg (inverse transform otomatis)

---

## 5. 🤖Modeling — Phase 1–3

### Model yang Diuji

| Model | Deskripsi |
|---|---|
| Linear Regression | Baseline — garis linear tanpa penalti |
| Polynomial (degree=2) | Tambah fitur kuadrat dan interaksi |
| Ridge (L2) | Penalti λΣwᵢ² — kurangi efek multikolinearitas |
| Lasso (L1) | Penalti λΣ\|wᵢ\| — seleksi fitur otomatis |

### Hasil

| Model | RMSE Train | RMSE Test | MAE Test | R² Test | CV RMSE |
|---|---|---|---|---|---|
| Linear Regression | 2.9267 | 2.3715 | 1.7849 | 0.8954 | 3.0022 |
| Polynomial (deg=2) | 2.4987 | 2.3900 | 1.7989 | 0.8938 | 3.1341 |
| **Ridge Regression** | **2.9260** | **2.3691** | **1.7823** | **0.8956** | 3.0203 |
| Lasso Regression | 2.9263 | 2.3711 | 1.7838 | 0.8954 | 3.0459 |

**Pemenang: Ridge Regression** — RMSE Test 2.3691 mpg, R² 0.8956

### Analisis

- **Ridge vs Linear:** Ridge menang karena regularisasi L2 meredam multikolinearitas `kapasitas_mesin`–`kekuatan_mesin`
- **Polynomial gagal mengalahkan Ridge:** Overfitting ringan — degree=2 menghasilkan terlalu banyak fitur untuk 398 data
- **Lasso tidak men-drop fitur:** Alpha optimal sangat kecil → semua fitur berkontribusi, penalti hampir tidak aktif
- **Outlier handling (Phase 3):** Tidak memberikan perubahan — Yeo-Johnson sudah cukup mereduksi efek outlier

---

## 6. 🔬Feature Engineering — Phase 4

### Eksperimen

| Eksperimen | Perubahan | RMSE Test | Δ RMSE |
|---|---|---|---|
| Baseline | Ridge tanpa perubahan | 2.3691 | +0.0000 |
| **Exp A** | Drop `kapasitas_mesin` | **2.3552** | **−0.0139** |
| Exp B | Tambah `power_to_weight` | 2.3830 | +0.0139 |
| Exp C | `tahun_rilis` → `umur_mobil` | 2.3927 | +0.0236 |
| Exp D | A + B + C | 2.3980 | +0.0289 |

### Interpretasi

**Exp A berhasil:** Drop `kapasitas_mesin` meningkatkan performa. Artinya `kapasitas_mesin` meskipun berkorelasi dengan target, informasinya sudah terwakili penuh oleh `kekuatan_mesin` dan adanya keduanya sekaligus menciptakan noise multikolinearitas.

**Exp B & C gagal:** Fitur baru tidak menambah informasi yang belum ditangkap model. Ordinal encoding untuk `tahun_rilis` sudah cukup representatif.

**Exp D (kombinasi) lebih buruk:** Efek negatif B dan C mendominasi efek positif A.

---

## 7. 🥇Model Final

### Spesifikasi

| Item | Nilai |
|---|---|
| Algoritma | Ridge Regression (RidgeCV) |
| Fitur input | `kekuatan_mesin`, `berat_mobil`, `akselerasi` (numerik) + `jumlah_silinder` (ordinal) + `tahun_rilis` (ordinal) + `asal_pabrikan` (one-hot) |
| Fitur di-drop | `kapasitas_mesin` |
| Transform fitur | Yeo-Johnson (standardize=True) |
| Transform target | Yeo-Johnson (standardize=True) |
| **RMSE Test** | **2.3552 mpg** |
| **R² Test** | **0.8968** |
| File | `models/ridge_final.pkl` |

### Interpretasi Performa

- RMSE 2.36 mpg pada rentang data 9–46 mpg → error rata-rata ~6.4% dari range
- R² 0.8968 → model menjelaskan **89.7% variasi** konsumsi BBM dari fitur yang ada
- Gap RMSE train/test kecil → tidak ada overfitting

### Cara Menggunakan Model

```python
import joblib

# Load model
model = joblib.load('models/ridge_final.pkl')

# Prediksi (input DataFrame dengan kolom yang sama seperti training)
# Kolom: jumlah_silinder (str), kekuatan_mesin, berat_mobil, akselerasi,
#        tahun_rilis (str), asal_pabrikan
y_pred = model.predict(X_new)  # output dalam satuan mpg
```

---

## 8. 🔄Alur Eksperimen Keseluruhan

```
EDA (eda.ipynb)
  ↓ Insight: distribusi, korelasi, missing value, multikolinearitas
  
Preprocessing + Modeling (preprocessing.ipynb)
  ↓ Pipeline: imputer → encoding → Yeo-Johnson → Ridge/Lasso/Linear/Poly
  ↓ Baseline: Ridge RMSE=2.3691, R²=0.8956
  ↓ Outlier handling (Phase 3): tidak mengubah hasil
  
Feature Engineering (feature_engineering.ipynb)
  ↓ Exp A (drop kapasitas_mesin): RMSE=2.3552 ✅ terbaik
  ↓ Model final disimpan → models/ridge_final.pkl
```

---

## 9. 🧩Struktur src/

| File | Isi |
|---|---|
| `src/data/load_data.py` | `load_raw()`, `split_features_target()`, `split_train_test()`, `load_and_split()` |
| `src/features/build_features.py` | Konfigurasi kolom, `PowerToWeightAdder`, `CarAgeAdder`, `FullFeatureEngineer`, `build_preprocessor()`, `build_target_transformer()` |
| `src/models/train.py` | `build_linear/ridge/lasso/polynomial_pipeline()`, `evaluate()`, `save_model()`, `load_model()`, `predict()` |
| `src/visualization/plots.py` | `plot_actual_vs_pred()`, `plot_residuals()`, `plot_model_comparison()`, `plot_train_test_gap()`, `plot_feature_importance()`, `plot_fe_comparison()` |

---

## 10. 🚀Potensi Improvement Selanjutnya

| Ide | Ekspektasi |
|---|---|
| Model non-linear (Random Forest, Gradient Boosting) | Potensi R² > 0.93, tangkap hubungan non-linear |
| Feature engineering lanjut (interaksi fitur manual) | Improvement kecil-sedang |
| Tambah data dari tahun lebih baru | Generalisasi lebih baik |
| Hyperparameter tuning Polynomial + Ridge (RandomSearchCV) | Optimasi degree + alpha sekaligus |
