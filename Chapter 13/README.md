# Chapter 13: Loading and Preprocessing Data with TensorFlow

Bab ini membahas cara memuat dan memproses data secara efisien menggunakan TensorFlow. Fokus utamanya adalah membangun pipeline data yang scalable dan optimal menggunakan `tf.data`, format `TFRecord`, dan preprocessing yang bisa diintegrasikan ke dalam model.

---

## 🌟 Tujuan

* Mengelola data berskala besar secara efisien
* Membuat pipeline data menggunakan `tf.data.Dataset`
* Mempersiapkan data dengan transformasi seperti normalisasi, encoding, dan augmentasi
* Menyimpan dan membaca data dalam format biner `TFRecord`
* Menggunakan preprocessing layer Keras dan TF Transform untuk produksi

---

## 📦 1. The `tf.data` API

API `tf.data` digunakan untuk membuat pipeline input data yang efisien dan dapat dikustomisasi.

### Contoh penggunaan:

```python
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(1000).batch(32).prefetch(1)
```

### Transformasi umum:

* `.map(fungsi)` – transformasi item
* `.shuffle(buffer_size)` – mengacak data
* `.batch(batch_size)` – batching
* `.repeat()` – mengulang dataset tanpa batas
* `.prefetch(buffer_size)` – melakukan preload untuk efisiensi

---

## 🧪 2. Preprocessing Data

Preprocessing dapat dilakukan langsung di pipeline dengan `.map()`.

### Contoh:

```python
def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

dataset = dataset.map(preprocess)
```

Transformasi umum:

* Normalisasi numerik
* One-hot encoding
* Padding sequences
* Augmentasi gambar (flip, crop, rotate, dll)

---

## 📀 3. TFRecord Format

TFRecord adalah format biner untuk menyimpan data secara efisien dalam skala besar.

### Menulis:

```python
with tf.io.TFRecordWriter('data.tfrecord') as writer:
    writer.write(example.SerializeToString())
```

### Membaca:

```python
raw_dataset = tf.data.TFRecordDataset('data.tfrecord')

def parse_example(serialized):
    feature_description = {
        'feature1': tf.io.FixedLenFeature([], tf.float32),
        ...
    }
    return tf.io.parse_single_example(serialized, feature_description)

parsed_dataset = raw_dataset.map(parse_example)
```

---

## 📦 4. Protocol Buffers

TFRecord menyimpan data dalam format protobuf (`tf.train.Example`). Untuk data sekuensial, gunakan `tf.train.SequenceExample`.

Langkah-langkah:

* Definisikan struktur feature
* Encode menggunakan `tf.train.Feature`
* Serialize sebelum tulis ke file

---

## 🔢 5. Handling Sequences: SequenceExample

Untuk data seperti:

* Teks (urutan kata)
* Sensor (urutan waktu)
* Audio (urutan frame)

Gunakan `SequenceExample` untuk menyimpan nested lists.

---

## 🌤️ 6. Encoding Kategorikal

### One-Hot Encoding

* Gunakan jika jumlah kategori kecil
* Gunakan `tf.keras.layers.CategoryEncoding`

### Embedding

* Gunakan jika kategori banyak
* Gunakan `tf.keras.layers.Embedding`

---

## 🧱 7. Keras Preprocessing Layers

Preprocessing dapat dilakukan dalam model menggunakan layer berikut:

| Layer                 | Fungsi                           |
| --------------------- | -------------------------------- |
| `Normalization()`     | Normalisasi numerik              |
| `StringLookup()`      | Konversi string ke integer index |
| `IntegerLookup()`     | Konversi integer ke index        |
| `CategoryEncoding()`  | One-hot encoding                 |
| `TextVectorization()` | Tokenisasi dan vektorisasi teks  |

Keuntungan:

* Preprocessing jadi bagian dari model
* Konsisten antara training dan inference

---

## ⚙️ 8. TF Transform (TFT)

Untuk preprocessing skala besar dalam pipeline produksi (misalnya di GCP/TFX):

* Mendukung preprocessing yang *stateless* maupun *stateful* (contoh: normalisasi dengan mean dari training set)
* Sinkronisasi transformasi antara training dan serving
* Menggunakan Apache Beam untuk eksekusi terdistribusi

---

## 🌍 9. TensorFlow Datasets (TFDS)

TFDS menyediakan dataset populer siap pakai:

* MNIST, CIFAR, IMDB, dll.
* Sudah dibagi menjadi split: `train`, `test`, `validation`
* API: `tfds.load("nama_dataset")`

### Contoh:

```python
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True
)
```
---

