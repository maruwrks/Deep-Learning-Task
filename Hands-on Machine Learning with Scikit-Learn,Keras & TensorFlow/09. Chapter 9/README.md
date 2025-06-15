# Penjelasan Teoritis

---

## 1. Pengantar

* **Clustering**: Mengelompokkan data berdasarkan kesamaan.
* **Estimasi Kepadatan**: Menentukan distribusi probabilitas dari data.
* **Deteksi Anomali**: Mengidentifikasi data yang berbeda dari mayoritas.
* **Reduksi Dimensi & Visualisasi**: Menyederhanakan representasi data untuk analisis atau tampilan visual (dibahas lebih rinci di Bab 8).

Fokus utama bab ini adalah algoritma clustering seperti K-Means, DBSCAN, dan Gaussian Mixture Models (GMM), beserta aplikasi lainnya.

## 2. Clustering

### a. K-Means

**Cara kerja:**

1. Inisialisasi \$k\$ pusat cluster (centroid) secara acak.
2. Setiap data diberi label ke centroid terdekat.
3. Update centroid sebagai rata-rata dari data dalam setiap cluster.
4. Ulangi langkah 2 dan 3 hingga konvergen.

**Tujuan optimasi (Inertia):** Meminimalkan jumlah kuadrat jarak dari tiap data ke centroid-nya.

**Permasalahan:**

* Hasil bisa terjebak dalam solusi lokal buruk akibat inisialisasi acak.

**Solusi:**

* Gunakan `n_init` untuk mencoba beberapa inisialisasi.
* K-Means++ sebagai metode inisialisasi yang lebih baik.

**Menentukan jumlah cluster (\$k\$):**

* Gunakan metode elbow (plot inersia vs \$k\$).
* Hitung silhouette score untuk menilai seberapa baik pemisahan antar cluster.

**Keterbatasan:**

* Perlu menentukan \$k\$.
* Kurang cocok untuk bentuk cluster non-linier.
* Tidak efektif pada kepadatan cluster yang berbeda.

### b. Mini-Batch K-Means

Versi K-Means yang menggunakan mini-batch untuk mempercepat proses pada dataset besar. Performa mendekati K-Means biasa, namun jauh lebih efisien secara waktu.

## 3. DBSCAN

**Definisi Komponen:**

* **\$\epsilon\$-neighborhood:** Radius di sekitar sebuah data.
* **Core point:** Titik dengan jumlah tetangga minimal dalam radius.
* **Border point:** Titik dalam radius core point, tetapi bukan core point.
* **Noise point:** Titik yang tidak memenuhi dua kategori di atas.

**Proses:**

1. Mulai dari titik acak.
2. Jika titik adalah core, bangun cluster baru.
3. Border point ditambahkan ke cluster terdekat.
4. Titik yang tidak dapat dijangkau dianggap noise.

**Kelebihan:**

* Tidak perlu menentukan jumlah cluster.
* Dapat mendeteksi bentuk cluster arbitrer.
* Menemukan outlier dengan mudah.

**Kekurangan:**

* Sangat sensitif terhadap nilai `eps` dan `min_samples`.
* Tidak optimal untuk data berdimensi tinggi.

## 4. Gaussian Mixture Models (GMM)

Model yang memodelkan data sebagai kombinasi dari beberapa distribusi Gaussian.

**Langkah utama (EM algorithm):**

1. Inisialisasi parameter Gaussian.
2. **E-Step:** Hitung probabilitas data berasal dari masing-masing Gaussian.
3. **M-Step:** Perbarui parameter Gaussian berdasarkan estimasi pada E-Step.

**Karakteristik:**

* Berbeda dari K-Means yang bersifat *hard clustering*, GMM bersifat *soft clustering*.
* Mengestimasi probabilitas data milik sebuah cluster.

**Parameter penting:**

* `n_components`: jumlah distribusi Gaussian.
* `covariance_type`: bentuk kovarians (misal: `full`, `diag`).

**Penggunaan dalam deteksi anomali:**

* Data dengan probabilitas sangat rendah dianggap sebagai anomali.

**Kelebihan:**

* Fleksibel terhadap bentuk dan ukuran cluster.
* Menyediakan probabilitas keanggotaan.

**Kekurangan:**

* Perlu `n_components` ditentukan sebelumnya.
* Lebih lambat dibanding K-Means.

## 5. Tugas Lain dalam Pembelajaran Tanpa Pengawasan

### a. Estimasi Kepadatan

Menentukan bentuk distribusi dari data. GMM dapat digunakan untuk hal ini.

### b. Deteksi Anomali

Mengidentifikasi data yang tidak sesuai dengan distribusi mayoritas. Contoh pendekatan:

* **PCA**: Melihat besar *reconstruction error*.
* **GMM**: Titik dengan kepadatan rendah.
* **Isolation Forest**: Mengisolasi outlier dengan pohon.
* **One-Class SVM**: Menentukan batas sekeliling data normal.

### c. Novelty Detection

Berbeda dari deteksi anomali umum. Dilatih hanya pada data "normal" dan mengevaluasi *instance* baru apakah masuk kategori aneh.
