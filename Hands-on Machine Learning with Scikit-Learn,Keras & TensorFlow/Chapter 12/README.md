## 1. Data Loading dan Preprocessing

```python
from sklearn.datasets import fetch_california_housing
...
scaler = StandardScaler()
```

📖 **Teori:**

* Dataset *California Housing* digunakan sebagai masalah regresi untuk memprediksi harga rumah median.
* `StandardScaler` digunakan untuk menormalisasi fitur agar distribusinya memiliki mean = 0 dan std = 1. Ini penting untuk menjaga stabilitas dan efisiensi pelatihan neural network.

---

## 2. Custom Loss Function: Huber Loss

```python
class HuberLoss(keras.losses.Loss):
    ...
```

📖 **Teori:**

* **Huber Loss** adalah kombinasi dari **Mean Squared Error (MSE)** dan **Mean Absolute Error (MAE)**:

  * Untuk error kecil, loss bersifat kuadrat (MSE), sensitif terhadap perubahan kecil.
  * Untuk error besar, loss menjadi linier (MAE), tahan terhadap outlier.
* Digunakan saat outlier cukup banyak, tapi kita tetap ingin model sensitif terhadap kesalahan kecil.

---

## 3. Custom Layers: MyDense

```python
class MyDense(keras.layers.Layer):
    ...
```

📖 **Teori:**

* Membuat layer Dense sendiri memberi fleksibilitas penuh untuk mengatur weight, bias, dan aktivasi.
* `build()` digunakan untuk membuat variabel trainable (`kernel` dan `bias`).
* `call()` mendefinisikan forward pass layer.
* Subclassing seperti ini memberi kita kontrol penuh terhadap arsitektur.

---

## 4. Custom Model: MyModel

```python
class MyModel(keras.Model):
    ...
```

📖 **Teori:**

* Dengan mewarisi `keras.Model`, kita bisa menentukan arsitektur model secara manual.
* Struktur seperti ini cocok untuk model kompleks dengan banyak cabang, kondisi if/else, atau skip connection.

---

## 5. Custom Training Loop

```python
for epoch in range(1, n_epochs + 1):
    with tf.GradientTape() as tape:
        ...
```

📖 **Teori:**

* Dengan `GradientTape()`, kita melakukan **forward pass secara manual**, menghitung loss, dan **menghitung gradien secara eksplisit**.
* `apply_gradients()` digunakan untuk mengupdate weight berdasarkan gradien.
* Keuntungan pendekatan ini:

  * Kontrol penuh atas proses pelatihan
  * Bisa digunakan untuk multi-loss, multi-optimizer, atau penyesuaian metrik yang rumit

---

## 6. TensorFlow Functions (`@tf.function`)

```python
@tf.function
def sum_up_to_graph(n):
    ...
```

📖 **Teori:**

* `@tf.function` mengubah fungsi Python menjadi **graph computation** (statis), yang jauh lebih cepat dan bisa dioptimasi.
* TensorFlow akan **men-trace** fungsi dan menghasilkan graph yang dieksekusi oleh runtime engine.
* Fitur ini disebut **AutoGraph**, mampu mengubah Python `for`, `if`, `while`, menjadi bentuk yang bisa dioptimasi dalam graph.
* Perbedaan besar antara eager mode (imperatif) dan graph mode (deklaratif) adalah performa dan portabilitas.

---
