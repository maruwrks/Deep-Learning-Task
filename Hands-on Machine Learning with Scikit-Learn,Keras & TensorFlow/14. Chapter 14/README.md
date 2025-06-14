# Penjelasan Teoritis

---

## 1. Data Preparation (tf\_flowers dari TFDS)

```python
import tensorflow_datasets as tfds
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
```

📖 **Teori:**

* `tfds.load()` digunakan untuk memuat dataset siap pakai.
* Dataset `tf_flowers` terdiri dari gambar bunga dari 5 kelas berbeda.
* Dataset dibagi menjadi training, validasi, dan test set secara proporsional.

---

## 2. Augmentasi dan Preprocessing Gambar

```python
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
```

📖 **Teori:**

* Model pretrained (seperti Xception) biasanya memerlukan input ukuran tetap (224x224).

```python
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(...)
```

📖 **Teori:**

* Augmentasi digunakan untuk meningkatkan generalisasi model dengan membuat variasi data baru.
* Teknik seperti flip, saturasi, dan brightness bersifat label-preserving.

---

## 3. Pipeline Data dengan `tf.data`

```python
dataset = dataset.shuffle(...).map(...).batch(...).prefetch(...)
```

📖 **Teori:**

* `tf.data` adalah API untuk membangun pipeline input efisien.
* `shuffle` mencegah urutan data yang bias.
* `batch` dan `prefetch` mempercepat pelatihan melalui paralelisme.

---

## 4. CNN Kustom: ResNet-34

```python
class ResidualUnit(keras.layers.Layer):
```

📖 **Teori:**

* **Residual Connections** mengatasi masalah vanishing gradients.
* Unit residual belajar perbedaan (residu) dari input dan output.
* Digunakan dalam ResNet untuk pelatihan jaringan yang sangat dalam.

```python
model.add(ResidualUnit(...))
```

📖 **Teori:**

* Arsitektur mengikuti ResNet-34: blok residual bertingkat dengan peningkatan filter.

---

## 5. Transfer Learning dengan Xception

```python
base_model = keras.applications.xception.Xception(...)
```

📖 **Teori:**

* Transfer learning memanfaatkan representasi fitur dari model besar yang dilatih pada ImageNet.
* `include_top=False` berarti tidak menyertakan classifier bawaan.

```python
layer.trainable = False
```

📖 **Teori:**

* Freezing layer menjaga bobot pretrained agar tidak rusak saat awal pelatihan.

```python
GlobalAveragePooling2D() + Dense(n_classes)
```

📖 **Teori:**

* Klasifikasi dilakukan menggunakan head baru di atas base model.
* Global pooling menyederhanakan representasi spasial sebelum klasifikasi.

---

## 6. Training dan Fine-Tuning

📖 **Tahap 1:**

* Melatih hanya layer atas (head) dengan learning rate tinggi.

📖 **Tahap 2:**

* Fine-tuning: membuka semua layer dan melatih ulang dengan learning rate kecil untuk penyesuaian halus.

---

## 7. Evaluasi dan Visualisasi Feature Map

```python
activation_model = Model(inputs=..., outputs=base_model.layers[:20])
```

📖 **Teori:**

* Visualisasi feature map berguna untuk memahami representasi internal dari model.
* Feature map pada layer awal biasanya menunjukkan deteksi tepi, tekstur, dsb.

```python
plt.imshow(feature_maps[0, :, :, i], cmap="viridis")
```

📖 **Teori:**

* Setiap channel menunjukkan aktivasi terhadap filter tertentu.
* Memvisualisasikan ini membantu interpretabilitas model CNN.

---
