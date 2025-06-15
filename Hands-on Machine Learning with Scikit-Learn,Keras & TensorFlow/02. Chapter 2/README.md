# Penjelasan Teoritis

---

## 🗃️ 1. Persiapan dan Eksplorasi Data

### • Fetching & Loading

* Data diunduh dari GitHub A. Géron.
* Dibaca ke dalam `pandas.DataFrame`.

### • Eksplorasi Awal

* Menampilkan info dataset, statistik deskriptif, dan visualisasi histogram.
* Scatter plot digunakan untuk memahami hubungan geografis dan harga.

### • Visualisasi Tambahan:

* Warna mencerminkan harga rumah (`cmap="jet"`), ukuran titik menunjukkan populasi.
* Scatter Matrix dan korelasi numerik digunakan untuk feature engineering awal.

---

## 🤸‍♂️ 2. Feature Engineering

### • Fitur Baru:

* `rooms_per_household`
* `bedrooms_per_room`
* `population_per_household`

Fitur ini menunjukkan korelasi lebih baik dengan target (`median_house_value`).

---

## ✅ 3. Splitting Dataset

Beberapa strategi splitting digunakan:

* Random sampling
* Deterministik (menggunakan `crc32`)
* Stratified sampling berdasarkan `income_cat` (kategorisasi `median_income`)

Stratifikasi digunakan untuk menjaga proporsi data penting.

---

## 🔄 4. Preprocessing Pipeline

### • Numerical:

* Imputasi nilai hilang (median)
* Feature engineering tambahan
* Standardisasi

### • Kategori:

* Encoding ordinal
* One-hot encoding

Pipeline dibangun dengan `ColumnTransformer` untuk menggabungkan pipeline numerik dan kategori.

---

## 📊 5. Model Dasar

### a. Linear Regression

* Performa cukup buruk (underfitting)
* RMSE tinggi

### b. Decision Tree

* RMSE sangat rendah di training (overfitting)
* Cross-validation menunjukkan performa buruk

### c. Random Forest

* Performa lebih baik, overfitting tapi masih generalisasi lebih baik dari Decision Tree

---

## 🔢 6. Evaluasi dengan Cross Validation

Cross-validation dilakukan (10-fold):

* Mengukur variabilitas model
* Metrik: RMSE dari `mean_squared_error`
* Model terbaik: Random Forest

---

## ⚖️ 7. Hyperparameter Tuning

Menggunakan `GridSearchCV`:

* Dicoba berbagai kombinasi `n_estimators` dan `max_features`
* Hasil: estimator terbaik dan parameter optimal

---

## 🔹 8. Feature Importance

Setelah training, ditampilkan fitur-fitur terpenting untuk model terbaik (RandomForest).

Fitur paling penting:

* `median_income`
* `INLAND` (hasil dari OneHotEncoding dari ocean proximity)
* `rooms_per_household`

---

## 📊 9. Evaluasi Akhir di Test Set

* RMSE akhir dihitung pada `strat_test_set`
* Confidence interval (95%) untuk generalisasi error dihitung menggunakan distribusi `t`

```python
confidence_interval = np.sqrt(stats.t.interval(...))
```

---

## 💼 Kesimpulan

* Proyek ini menunjukkan pipeline lengkap Machine Learning:

  * Mulai dari pengumpulan data
  * Eksplorasi dan visualisasi
  * Feature engineering
  * Training model dan tuning hyperparameter
  * Evaluasi akhir

* Random Forest adalah model paling andal dari eksperimen ini.

---

## 📖 Referensi

* A. Géron, *Hands-On Machine Learning*, Chapter 2
* Dataset: California Housing (from StatLib)
