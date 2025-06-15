# Penjelasan Teoritis

---

## Persiapan Dataset

Dataset MNIST diambil menggunakan `fetch_openml`, terdiri dari 70.000 gambar grayscale berukuran 28x28 piksel. Gambar pertama divisualisasikan menggunakan matplotlib untuk eksplorasi awal.

Data dibagi menjadi dua bagian:

* 60.000 data untuk pelatihan (`X_train`, `y_train`)
* 10.000 data untuk pengujian (`X_test`, `y_test`)

---

## Pelatihan Klasifikasi Biner (Apakah Angka 5?)

Label target diubah menjadi biner `True` jika digit adalah 5 (`y_train_5`) dan `False` untuk digit lainnya. Model `SGDClassifier` dilatih menggunakan `X_train` dan `y_train_5`. Model kemudian diuji untuk memprediksi apakah suatu gambar adalah angka 5.

---

## Evaluasi Model: Akurasi dan Validasi Silang

Model dievaluasi menggunakan:

* **StratifiedKFold** untuk menjaga proporsi label selama pembagian data.
* **`cross_val_score`** untuk menghitung akurasi rata-rata pada 3 fold.

Model pembanding sederhana `Never5Classifier` yang selalu memprediksi `False` menunjukkan bahwa akurasi bukan metrik yang dapat diandalkan dalam kasus dataset tidak seimbang.

---

## Matriks Kebingungan dan Skor Klasifikasi

* Matriks kebingungan dibuat menggunakan `confusion_matrix` untuk menghitung TP, FP, FN, dan TN.
* Diikuti dengan perhitungan:

  * **Presisi**: Rasio TP / (TP + FP)
  * **Recall**: Rasio TP / (TP + FN)
  * **F1 Score**: Harmonic mean antara presisi dan recall

---

## Trade-off Precision dan Recall

Model dapat disesuaikan untuk mencapai presisi atau recall tinggi dengan mengatur ambang batas keputusan (*decision threshold*).

* `decision_function` memberikan skor kepercayaan.
* `precision_recall_curve` digunakan untuk membangun grafik presisi vs recall terhadap ambang.

Ambang dengan presisi minimal 90% diekstrak, dan performa model dengan threshold tersebut dievaluasi kembali.

---

## Kurva ROC dan AUC

* Kurva ROC menunjukkan hubungan antara **False Positive Rate (FPR)** dan **True Positive Rate (TPR)**.
* Area Under Curve (**ROC AUC Score**) digunakan untuk membandingkan model.

Model `SGDClassifier` dan `RandomForestClassifier` dibandingkan, di mana Random Forest menunjukkan skor AUC yang lebih tinggi.

---

## Klasifikasi Multikelas (Multiclass)

### Menggunakan SVM (Support Vector Machine)

Model `SVC` secara otomatis melakukan klasifikasi multikelas menggunakan metode *One-vs-One (OvO)*, menghasilkan skor untuk semua kelas dan memilih yang tertinggi.

* `decision_function` digunakan untuk melihat skor semua kelas.
* `classes_` menyimpan label untuk masing-masing skor.

### One-vs-Rest (OvR)

Model `OneVsRestClassifier(SVC())` membuat satu pengklasifikasi per kelas dan mengembalikan skor tertinggi.

---

## Evaluasi Akurasi Multiclass

Model `SGDClassifier` juga digunakan untuk klasifikasi multikelas:

* `cross_val_score` mengukur akurasi klasifikasi multikelas.
* `StandardScaler` digunakan untuk normalisasi fitur sebelum pelatihan untuk meningkatkan akurasi.
