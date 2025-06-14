# Penjelasan Teoritis

---

## 1. **Pemeriksaan dan Pengaturan GPU**

```python
tf.config.experimental.list_physical_devices('GPU')
```

Kode ini mengecek apakah GPU tersedia. Jika tersedia, dapat dikonfigurasi untuk digunakan atau tidak (dalam contoh ini GPU dinonaktifkan untuk menjalankan model di CPU).

---

## 2. **Persiapan Data MNIST**

MNIST adalah dataset gambar tangan angka 0–9 berukuran 28x28 piksel grayscale. Data di-normalisasi menjadi \[0,1] dan dibagi menjadi data training, validasi, dan testing.

---

## 3. **Training Model Dasar**

Model sederhana Sequential dengan dua `Dense` layers dilatih menggunakan `sparse_categorical_crossentropy`. Model ini digunakan sebagai baseline sebelum disimpan dan diekspor.

---

## 4. **Export SavedModel Format**

```python
model.export(model_dir)
```

Model disimpan ke direktori dalam format `SavedModel`, format standar TensorFlow untuk deployment. Direktori berisi arsitektur, bobot, dan metadata signature.

---

## 5. **Menjalankan TensorFlow Serving**

* Menginstal `tensorflow-model-server` dari repository Google.
* Menjalankan server REST API di port 8501 secara background dengan model yang telah diekspor.
* Melakukan request HTTP POST ke endpoint `/v1/models/{model_name}:predict` dengan data input JSON.
* Mendapatkan prediksi dari model dan memvisualisasikan hasilnya.

```python
requests.post(SERVER_URL, data=request_json)
```

---

## 6. **Distribusi Pelatihan dengan MirroredStrategy**

`tf.distribute.MirroredStrategy()` memungkinkan pelatihan model menggunakan beberapa GPU secara paralel.

Semua definisi model, optimizer, dan kompilasi model harus berada dalam konteks `strategy.scope()` agar TensorFlow tahu bahwa distribusi training sedang digunakan.

```python
with strategy.scope():
    # define model here
```

### Kelebihan:

* Meningkatkan kecepatan training.
* Dapat memanfaatkan semua GPU pada satu mesin.

---

## 7. **Evaluasi dan Visualisasi**

* Menghitung `loss` dan `accuracy` pada test set.
* Menampilkan plot learning curve dari training dan validasi menggunakan `pandas.DataFrame.plot()`.

---
