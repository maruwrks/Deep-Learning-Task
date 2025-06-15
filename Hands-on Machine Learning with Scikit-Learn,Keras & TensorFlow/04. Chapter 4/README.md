# Penjelasan Teoritis

---

## 1. Linear Regression

Model linier digunakan untuk memodelkan hubungan antara satu atau lebih fitur numerik (X) dengan target kontinu (y).

### • Metode:

* **Normal Equation**:

  * Solusi langsung dari persamaan linier:
    \$\theta = (X^TX)^{-1}X^Ty\$\
* **Scikit-Learn**: Menggunakan `LinearRegression()`
* **SVD (Singular Value Decomposition)**: Alternatif lebih stabil terhadap inversi matriks

---

## 2. Gradient Descent

### • Batch Gradient Descent (BGD)

* Menggunakan seluruh dataset untuk satu update parameter.

### • Stochastic Gradient Descent (SGD)

* Update dilakukan tiap satu instance data.
* Lebih cepat dan cocok untuk big data.

### • SGDRegressor dari Scikit-Learn

* Optimisasi regresi linier menggunakan SGD.

---

## 3. Polynomial Regression

Menggunakan `PolynomialFeatures` untuk menambah derajat polinomial pada fitur:

* Mampu menangkap pola non-linier.
* Digabung dengan `LinearRegression`.

### • Learning Curves

* Menilai performa model dengan plot error pada data latih vs validasi.
* Berguna untuk mengenali overfitting/underfitting.

---

## 4. Regularized Linear Models

### • Ridge Regression (L2 Penalty)

* Menambahkan penalti kuadrat terhadap magnitude koefisien.

### • Lasso Regression (L1 Penalty)

* Menghasilkan model sparse: bisa menyetel beberapa koefisien jadi nol.

### • Elastic Net

* Kombinasi L1 dan L2 (parameter `l1_ratio` mengontrol rasio).

---

## ⏱️ 5. Early Stopping

Strategi regulasi untuk menghindari overfitting:

* Hentikan pelatihan saat error validasi tidak membaik.
* Digunakan bersama model polinomial derajat tinggi.

---

## 6. Logistic Regression (Binary Classification)

Model linier probabilistik untuk klasifikasi:

* Estimasi probabilitas menggunakan fungsi sigmoid.

### • Contoh:

* Memprediksi apakah bunga iris adalah "virginica" berdasarkan lebar petal.

### • Plot Probabilitas:

* Menampilkan batas keputusan pada \$x=1.6\$ (probabilitas = 0.5).

---

## 7. Softmax Regression (Multiclass)

Generalisasi Logistic Regression untuk multi-kelas:

* Setiap kelas memiliki skor dan probabilitas dikalkulasi dengan fungsi softmax.

### • Implementasi:

```python
LogisticRegression(multi_class="multinomial", solver="lbfgs")
```

* Memprediksi kelas dan probabilitas untuk input dua fitur (panjang & lebar petal).

---
