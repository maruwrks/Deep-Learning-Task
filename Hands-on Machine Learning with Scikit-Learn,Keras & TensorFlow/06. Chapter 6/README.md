# Penjelasan Teoritis

---

## Konsep Dasar

Decision Tree adalah algoritma pembelajaran yang menyusun aturan keputusan dalam struktur berbentuk pohon. Setiap node mewakili kondisi berdasarkan fitur input, dan cabangnya menentukan hasil keputusan.

Model ini mudah dipahami dan sangat berguna untuk eksplorasi awal dan interpretasi model.

---

## Pelatihan dan Visualisasi

Scikit-Learn menyediakan `DecisionTreeClassifier` dan `DecisionTreeRegressor`.

* **Training:**

  * Menggunakan algoritma **CART (Classification and Regression Tree)**.
  * Membagi dataset menggunakan satu fitur dan threshold yang meminimalkan impurity.

* **Visualisasi:**

  * Menggunakan `export_graphviz` untuk menghasilkan file `.dot`.
  * Dapat dikonversi ke format PNG atau SVG menggunakan Graphviz.

Setiap node menunjukkan:

* `gini` (untuk klasifikasi) atau `mse` (untuk regresi).
* Jumlah sampel yang mencapai node tersebut.
* Distribusi kelas atau nilai rata-rata target.

---

## Prediksi dan Probabilitas

* **Prediksi klasifikasi:** Traversal pohon berdasarkan kondisi fitur sampai mencapai *leaf node*, lalu pilih kelas mayoritas.
* **Probabilitas kelas:** Proporsi kelas pada *leaf node*.

---

## Algoritma CART

### Untuk Klasifikasi:

* Tujuan: membagi data agar impurity sekecil mungkin.
* Ukuran impurity:

  * **Gini Impurity**:
    $G = 1 - \sum p_k^2$
  * **Entropy** (opsional):
    $H = - \sum p_k \log_2(p_k)$

### Untuk Regresi:

* Gunakan **Mean Squared Error (MSE)** sebagai fungsi biaya:
  $\text{MSE} = \sum (y_i - \bar{y})^2$

CART bersifat **greedy**, artinya hanya memilih pembagian terbaik lokal pada setiap node, bukan global.

---

## Regularisasi dan Overfitting

Decision Tree cenderung overfit jika tidak dibatasi. Untuk mengontrolnya, beberapa hyperparameter penting:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_leaf_nodes`
* `max_features`

Semakin ketat regularisasi (misalnya, depth kecil), semakin kecil risiko overfitting.

---

## Regresi dengan Decision Tree

* Sama seperti klasifikasi, tetapi setiap *leaf node* menyimpan nilai rata-rata target.
* Fungsi biaya default: **MSE**.
* Model juga rentan terhadap overfitting.

