# Penjelasan Teoritis

---

## 1. Pembuatan Dataset Time Series Sintetis

```python
def generate_time_series(batch_size, n_steps):
```

* Fungsi ini membuat data deret waktu dengan kombinasi dua gelombang sinus dan sedikit noise acak.
* Hal ini mensimulasikan dinamika time series nyata seperti sensor atau suhu.

---

## 2. Prediksi Naif sebagai Baseline

```python
y_pred_naive = X_valid[:, -1]
```

* Prediksi nilai masa depan menggunakan titik terakhir dari data observasi.
* Ini adalah baseline sederhana yang digunakan untuk membandingkan performa model.

---

## 3. Model Linear

```python
keras.layers.Flatten() → keras.layers.Dense(1)
```

* Flatten mengubah input 2D menjadi vektor datar.
* Dense melakukan regresi linier pada input yang telah diflatkan.

---

## 4. Simple RNN

```python
keras.layers.SimpleRNN(1)
```

* RNN klasik yang hanya memiliki satu unit, menyimpan informasi sekuensial dari masa lalu secara sederhana.
* Berguna untuk perbandingan, meskipun tidak optimal untuk data time series yang kompleks.

---

## 5. Deep RNN (Stacked RNN)

```python
SimpleRNN(20, return_sequences=True) → SimpleRNN(20)
```

* Layer pertama mengembalikan seluruh urutan untuk dilanjutkan ke layer berikutnya.
* Layer kedua menyaring dan memproses urutan tersebut untuk membuat prediksi akhir.

---

## 6. LSTM (Long Short-Term Memory)

```python
keras.layers.LSTM(...)
```

* Memiliki mekanisme gate untuk menangani dependensi jangka panjang.
* Lebih baik dari RNN klasik dalam banyak kasus time series.

---

## 7. GRU (Gated Recurrent Unit)

* Alternatif dari LSTM yang lebih ringan (fewer parameters).
* Menawarkan performa sebanding dengan arsitektur lebih cepat dalam beberapa kasus.

---

## 8. WaveNet (Causal Dilated CNN)

```python
keras.layers.Conv1D(..., padding="causal", dilation_rate=rate)
```

* CNN 1D dengan padding kausal memastikan tidak terjadi "leak" informasi masa depan.
* Dilation rate berlipat ganda menghasilkan receptive field yang sangat besar.
* Cocok untuk time series yang sangat panjang.

---

## 9. EarlyStopping Callback

```python
callbacks = [keras.callbacks.EarlyStopping(...)]
```

* Menghentikan pelatihan saat model tidak lagi membaik di data validasi.
* Menghindari overfitting dan mempercepat pelatihan.

---

## 10. Windowed Dataset

```python
def windowed_dataset(...)
```

* Membagi deret waktu panjang menjadi jendela pendek (sliding window).
* Setiap jendela digunakan untuk memprediksi langkah berikutnya.
* Strategi penting dalam multi-step forecasting.

---

## 11. Visualisasi Prediksi

```python
plt.plot(y_valid[:100], label="True")
plt.plot(y_pred[:100], label="Predicted")
```

* Memungkinkan evaluasi visual dari akurasi model.
* Berguna untuk membandingkan model secara intuitif.

---

