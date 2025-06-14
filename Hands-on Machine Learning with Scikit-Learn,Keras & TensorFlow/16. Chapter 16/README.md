# Penjelasan Teoritis
 
 ---

## 1. **Karakter-level Language Modeling**

### Dataset & Preprocessing

* Dataset diunduh dari GitHub (`tinyshakespeare`), dibaca ke dalam string, dan di-tokenisasi sebagai karakter individual menggunakan `Tokenizer(char_level=True)`.
* Token hasil tokenisasi dikurangi 1 (karena Keras memulai dari indeks 1).
* Dataset diubah menjadi window sepanjang `n_steps + 1`, lalu diproses menjadi pasangan input (X) dan target (Y) untuk pelatihan prediksi karakter berikutnya.
* Dataset di-batch dan dikonversi ke one-hot vectors dengan `tf.one_hot` untuk digunakan oleh model.

### Model `stateful GRU`

```python
model_char_rnn = keras.models.Sequential([
    keras.layers.InputLayer(input_shape=[None, max_id], batch_size=batch_size),
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2),
    keras.layers.GRU(128, return_sequences=True, stateful=True, dropout=0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(max_id, activation="softmax"))
])
```

* GRU pertama dan kedua digunakan dengan `stateful=True` agar menyimpan state antar batch.
* TimeDistributed Dense digunakan agar setiap langkah waktu memiliki prediksi karakter.
* Digunakan `sparse_categorical_crossentropy` karena target adalah indeks karakter.

### Generate Text

```python
def generate_text(model, tokenizer, text, n_chars=50, temperature=1):
    ...
```

* Menghasilkan teks dengan melakukan sampling berdasarkan distribusi output dari model.
* Parameter `temperature` mengatur randomness dari prediksi: nilai rendah = lebih deterministik.

---

## 2. **Sentiment Analysis (IMDB)**

### Dataset & Preprocessing

* Dataset `imdb_reviews` dimuat menggunakan `tfds.load`, dipisahkan menjadi training dan test.
* Teks dibersihkan dari HTML dan simbol non-alfabet.
* Tokenisasi dan padding dilakukan melalui `tf.strings.split(...).to_tensor()`.

### Text Vectorization & Embedding

```python
text_vec_layer = keras.layers.TextVectorization(max_tokens=vocab_size)
keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
```

* TextVectorization digunakan untuk membatasi vocab dan mengubah token menjadi indeks.
* Embedding mengubah indeks menjadi vektor representasi dense.

### Model GRU Sentiment

```python
model_sentiment = keras.models.Sequential([
    text_vec_layer,
    keras.layers.Embedding(...),
    keras.layers.GRU(128, return_sequences=True),
    keras.layers.GRU(128),
    keras.layers.Dense(1, activation="sigmoid")
])
```

* Dua GRU layer memproses teks, output terakhir diklasifikasikan sebagai positif/negatif.
* Binary cross-entropy digunakan sebagai fungsi loss.

---

## 3. **Transformer Block Custom**

### Positional Encoding

```python
class PositionalEncoding(keras.layers.Layer):
    ...
```

* Membuat encoding sinusoidal untuk menyisipkan informasi posisi ke input embedding.
* Dihitung menggunakan rumus dari paper Attention Is All You Need.

### Multi-Head Attention

```python
class MultiHeadAttention(keras.layers.Layer):
    ...
```

* Membagi input ke dalam beberapa "head", masing-masing dengan Dense projection q, k, v.
* Setiap head menghitung attention dan menghasilkan hasil tersendiri.
* Output semua head digabung dan diproyeksikan kembali ke ukuran asli.

### Transformer Block

```python
class TransformerBlock(keras.layers.Layer):
    ...
```

* Kombinasi Attention, Residual Connection, Normalization, dan Feedforward.
* Meniru struktur blok dasar dalam arsitektur Transformer.

### Demonstrasi Akhir

```python
input_sequences = np.random.randint(vocab_size, size=(2, 10))
embedding_output = embedding_layer(input_sequences)
pos_encoded_output = pos_encoding_layer(embedding_output)
transformer_output = transformer_block(pos_encoded_output)
```

* Simulasi input dummy diberikan ke embedding + positional + transformer untuk menunjukkan proses aliran data.

---
