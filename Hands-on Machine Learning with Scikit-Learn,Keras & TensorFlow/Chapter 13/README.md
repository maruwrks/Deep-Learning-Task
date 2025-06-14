# Penjelasan Teoritis

---

## 1. Data Preparation dan Normalisasi

```python
housing = fetch_california_housing()
scaler = StandardScaler()
```

📖 **Teori:**

* Dataset *California Housing* digunakan sebagai kasus regresi.
* `StandardScaler` digunakan untuk menormalkan fitur, sangat penting agar model konvergen lebih cepat dan stabil.
* Data dibagi menjadi train, validation, dan test set untuk menghindari overfitting dan memastikan evaluasi yang adil.

---

## 2. Menyimpan Data ke File CSV dalam Banyak Bagian

```python
save_to_multiple_csv_files(...)
```

📖 **Teori:**

* Sharding data ke dalam banyak file kecil lebih efisien untuk pemrosesan paralel oleh `tf.data.Dataset`.
* Format CSV digunakan agar mudah dibaca dan kompatibel dengan berbagai sistem.

---

## 3. Membaca CSV sebagai Dataset TensorFlow

```python
def parse_csv_line(...):
    fields = tf.io.decode_csv(...)
```

📖 **Teori:**

* `tf.io.decode_csv` digunakan untuk membaca dan mengurai baris CSV menjadi tensor.
* Fungsi ini memastikan semua data diubah menjadi format numerik TensorFlow.

```python
def csv_reader_dataset(...):
    dataset = tf.data.Dataset.list_files(...)
```

📖 **Teori:**

* `tf.data.Dataset.list_files()` membuat dataset dari daftar file.
* `interleave()` memungkinkan membaca banyak file secara paralel, sangat penting untuk efisiensi I/O.
* `.map()` digunakan untuk parsing dan preprocessing.
* `.shuffle()`, `.batch()`, dan `.prefetch()` digunakan untuk membuat pipeline yang efisien dan scalable.

---

## 4. Membuat Custom Preprocessing Layer: Standardization

```python
class Standardization(keras.layers.Layer):
    ...
```

📖 **Teori:**

* Membuat preprocessing layer sendiri memastikan preprocessing dapat dimasukkan dalam model.
* Keuntungan: preprocessing tidak perlu diulang saat deployment, model tetap self-contained.
* Layer ini menghitung `(input - mean) / std` sesuai standar normalisasi.

---

## 5. Membangun Model

```python
model = keras.models.Sequential([...])
```

📖 **Teori:**

* Model Sequential dibangun dengan preprocessing di dalam pipeline.
* Hidden layer menggunakan ReLU karena sifatnya yang non-linear dan sparsity-nya.
* Output layer hanya 1 unit karena ini adalah tugas regresi.

---

## 6. Training Model

```python
model.compile(loss="mse", optimizer="nadam", metrics=["RootMeanSquaredError"])
```

📖 **Teori:**

* Loss function: **MSE (Mean Squared Error)** cocok untuk regresi.
* Optimizer: **Nadam** menggabungkan RMSprop dan momentum Nesterov.
* RMSE sebagai metrik membuat interpretasi lebih dekat ke satuan data asli (harga rumah).

---

## 7. Evaluasi dan Prediksi

```python
model.evaluate(test_set)
model.predict(X_new)
```

📖 **Teori:**

* Evaluasi dilakukan pada `test_set` untuk mengukur generalisasi model.
* Prediksi dilakukan pada data baru yang belum pernah dilihat model.

---

## ✅ Kesimpulan

Kode ini mendemonstrasikan:

* Penggunaan pipeline data dengan `tf.data`
* Membaca dan parsing CSV secara efisien
* Membangun preprocessing sebagai bagian dari model
* Pelatihan dan evaluasi regresi menggunakan TensorFlow

Pendekatan ini cocok untuk data besar dan produksi karena scalable, efisien, dan mudah di-deploy.
