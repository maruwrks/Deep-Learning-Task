# Penjelasan Teoritis

---

## Reduksi Dimensi: Latar Belakang

Ruang berdimensi tinggi sering kali menyulitkan proses pembelajaran mesin karena data menjadi jarang dan jarak antar instance menjadi besar. Ini dikenal sebagai *curse of dimensionality*, yang dapat menyebabkan kesulitan dalam generalisasi model dan peningkatan risiko *overfitting*.

**Manfaat dari Reduksi Dimensi:**

* Proses pelatihan menjadi lebih cepat.
* Penggunaan memori menjadi lebih efisien.
* Dapat membantu dalam visualisasi data kompleks.
* Dapat memperbaiki performa beberapa model dengan mengurangi *noise*.

**Tantangan dan Kekurangannya:**

* Potensi kehilangan informasi penting.
* Algoritma tertentu bisa mahal secara komputasi.
* Fitur hasil transformasi bisa sulit diinterpretasikan.

Dua pendekatan utama yang digunakan adalah: **Proyeksi** dan **Manifold Learning**.

## Proyeksi

Dalam banyak kasus, data tidak tersebar merata di semua dimensi, melainkan hanya menempati subruang berdimensi lebih rendah. Proyeksi mencoba memetakan data dari ruang berdimensi tinggi ke subruang yang lebih rendah tersebut dengan tetap menjaga informasi sebanyak mungkin.

Namun, tidak semua struktur data cocok untuk proyeksi linier. Misalnya, struktur *Swiss roll* tidak bisa direduksi secara baik menggunakan proyeksi biasa karena bentuknya yang terlipat.

## Manifold Learning

*Manifold* adalah permukaan berdimensi rendah yang berada dalam ruang berdimensi tinggi. *Manifold learning* bertujuan mempelajari representasi rendah dimensi dari data yang terstruktur seperti manifold.

Pendekatan ini cocok untuk data non-linier yang tidak dapat direpresentasikan secara baik melalui proyeksi linier.

## Principal Component Analysis (PCA)

PCA merupakan metode reduksi dimensi paling umum yang bekerja dengan mencari arah dalam ruang data yang menyimpan varians terbesar. Arah tersebut disebut *principal components*.

### Fitur Utama PCA:

* Berdasarkan dekomposisi nilai singular (SVD).
* Komponen utama adalah vektor eigen dari matriks kovarians data.
* Proyeksi ke dimensi rendah dilakukan dengan mengalikan data ke komponen teratas.

### Implementasi:

* `PCA` dari Scikit-Learn dapat digunakan untuk transformasi dan kompresi data.
* `explained_variance_ratio_` menunjukkan seberapa besar varians yang dijelaskan oleh setiap komponen.
* Pemilihan jumlah komponen dapat dilakukan berdasarkan rasio varians kumulatif.

### Versi Lanjutan:

* **Incremental PCA (IPCA):** untuk dataset besar secara bertahap.
* **Randomized PCA:** menggunakan pendekatan probabilistik, lebih cepat untuk dimensi besar.

## Kernel PCA

Ketika PCA standar gagal menangkap struktur non-linier, *Kernel PCA* menggunakan *kernel trick* untuk mentransformasi data ke ruang berdimensi lebih tinggi dan kemudian melakukan PCA di sana.

Scikit-Learn mendukung berbagai kernel seperti RBF, polynomial, dan sigmoid. Pemilihan kernel dan parameter seperti `gamma` dapat dioptimalkan melalui *GridSearchCV* yang dikombinasikan dengan model *downstream*.

## Locally Linear Embedding (LLE)

LLE adalah algoritma *manifold learning* non-linier yang memodelkan hubungan lokal antara titik data dan mempertahankan hubungan tersebut dalam ruang berdimensi rendah. LLE tidak memerlukan kernel dan sangat efektif untuk data seperti *Swiss roll*. Namun, kompleksitasnya tinggi dan sensitif terhadap pemilihan jumlah tetangga (`n_neighbors`).

## Teknik Reduksi Dimensi Lainnya

Beberapa teknik tambahan untuk reduksi dimensi:

* **Multidimensional Scaling (MDS):** mempertahankan jarak antar titik.
* **Isomap:** mempertahankan jarak geodetik dalam manifold.
* **t-SNE:** sangat efektif untuk visualisasi 2D/3D, menjaga titik yang serupa tetap dekat.
* **Linear Discriminant Analysis (LDA):** pendekatan yang diawasi untuk memaksimalkan separabilitas antar kelas dalam dimensi yang lebih rendah.

---
