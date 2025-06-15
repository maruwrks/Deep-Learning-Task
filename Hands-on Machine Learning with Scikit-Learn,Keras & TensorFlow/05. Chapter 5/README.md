# Penjelasan Teoritis

---

## Soft Margin Classification

Alih-alih hanya mengandalkan batas pemisah yang keras (*hard margin*) yang tidak toleran terhadap pelanggaran (misalnya outlier atau noise), SVM mendukung konsep *soft margin*.

Dengan *soft margin*, model mencoba menyeimbangkan antara dua hal:

* Membuat margin selebar mungkin.
* Mengizinkan beberapa pelanggaran (misalnya data berada di sisi yang salah dari margin).

Hyperparameter `C` digunakan untuk mengontrol keseimbangan ini:

* **Nilai kecil `C`** → lebih banyak pelanggaran diizinkan (mengurangi overfitting).
* **Nilai besar `C`** → lebih ketat terhadap pelanggaran (berpotensi overfitting).

Scikit-Learn menyediakan implementasi dengan `LinearSVC` untuk klasifikasi linier cepat tanpa probabilitas output.

---

## Klasifikasi Non-Linier dengan Kernel

Jika data tidak bisa dipisahkan secara linier, SVM tetap bisa digunakan dengan memanfaatkan *kernel trick*. Beberapa teknik umum:

### 🔸 Kernel Polinomial

Model polinomial menambahkan fitur kompleks secara implisit tanpa eksplisit membuatnya. Cocok untuk data dengan batas keputusan melengkung.

```python
from sklearn.svm import SVC
poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)
```

### 🔸 Kernel RBF (Gaussian)

Mengukur kesamaan antara titik menggunakan fungsi eksponensial. Dua parameter penting:

* `gamma`: seberapa dekat titik harus berada untuk memengaruhi satu sama lain.
* `C`: kekuatan regularisasi.

```python
rbf_kernel_svm_clf = SVC(kernel="rbf", gamma=0.5, C=5)
```

Semakin besar `gamma`, model cenderung overfit; semakin kecil `gamma`, model bisa underfit.

---

## SVM untuk Regresi

SVM juga bisa digunakan untuk regresi, disebut *Support Vector Regression* (SVR). Alih-alih mencoba memisahkan kelas, SVR berusaha membuat model yang berada dalam margin toleransi tertentu (`epsilon`) terhadap data.

* `LinearSVR`: untuk regresi linier.
* `SVR`: untuk regresi non-linier dengan kernel (contoh: RBF).

```python
from sklearn.svm import SVR
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
```

---

## Komputasi dan Performa

| Model         | Kompleksitas Waktu               | Kernel | Skalabilitas           |
| ------------- | -------------------------------- | ------ | ---------------------- |
| LinearSVC     | \$O(mn)\$                        | Tidak  | Cepat                  |
| SVC (kernel)  | \$O(m^2 n)\$ hingga \$O(m^3 n)\$ | Ya     | Lambat jika data besar |
| SGDClassifier | \$O(mn)\$                        | Tidak  | Bisa out-of-core       |

* Gunakan `LinearSVC` untuk dataset besar.
* Gunakan `SVC` dengan kernel untuk dataset kecil hingga sedang.

---

## Mekanisme SVM di Balik Layar

SVM mengoptimalkan batas keputusan yang memaksimalkan margin antara dua kelas. Model didasarkan pada:

* Vektor bobot \$\mathbf{w}\$
* Bias \$b\$

Prediksi:

```math
\hat{y} = \text{sign}(\mathbf{w}^\intercal \mathbf{x} + b)
```

### Tujuan Optimisasi

Soft-margin SVM meminimalkan:

```math
\frac{1}{2} \|\mathbf{w}\|^2 + C \sum \zeta^{(i)}
```

dengan kendala agar setiap instance tidak melanggar margin terlalu banyak.

### Dual Problem dan Kernel

Masalah primal diubah ke *dual form* agar kernel dapat digunakan. Kernel menghitung *dot product* dalam ruang berdimensi tinggi secara implisit:

* **Linear kernel**: \$K(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\intercal \mathbf{b}\$
* **Polynomial**: \$K(\mathbf{a}, \mathbf{b}) = (\gamma \mathbf{a}^\intercal \mathbf{b} + r)^d\$
* **RBF (Gaussian)**: \$K(\mathbf{a}, \mathbf{b}) = \exp(-\gamma |\mathbf{a} - \mathbf{b}|^2)\$

Prediksi:

```math
h(\mathbf{x}_{\text{new}}) = \sum_i \alpha_i t^{(i)} K(\mathbf{x}^{(i)}, \mathbf{x}_{\text{new}}) + b
```

---

## SVM Online

SVM juga bisa dilatih secara inkremental menggunakan `SGDClassifier` dengan *hinge loss*.

```python
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(loss="hinge", alpha=0.01)
```
