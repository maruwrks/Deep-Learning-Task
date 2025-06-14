# Chapter 12: Custom Models and Training with TensorFlow

Bab ini membahas bagaimana memanfaatkan TensorFlow secara mendalam, khususnya ketika kita perlu membuat model kustom, fungsi pelatihan, dan komponen lainnya untuk kasus yang lebih kompleks dari penggunaan `tf.keras` biasa.

---

## 🧭 Pendahuluan

Walaupun `tf.keras` mencakup 95% kasus penggunaan machine learning, ada kalanya kita memerlukan fleksibilitas tambahan:
- Custom loss
- Custom metric
- Custom training loop
- Multi-optimizer
- Intervensi manual pada gradient
- Eksplorasi dan riset arsitektur baru

---

## ⚙️ Ringkasan Fitur TensorFlow

- Core mirip NumPy, tapi dengan dukungan GPU/TPU
- Komputasi terdistribusi (multi-device, multi-machine)
- JIT compiler: mengubah Python function jadi computation graph
- Ekosistem besar: `tf.data`, `tf.image`, `tf.signal`, `tf.keras`, `TensorBoard`, dll.
- Ekspor model lintas platform (Android, Web, Server)

---

## 📦 Struktur dan Operasi Dasar

### TensorFlow seperti NumPy:
- `tf.Tensor` vs `np.ndarray`
- Mendukung slicing, broadcasting, dan operasi vektor
- `tf.Variable` untuk nilai yang bisa diubah

### Struktur data lain:
- Ragged tensors, sparse tensors, sets, string tensors, dll.

---

## 🧩 Kustomisasi Komponen Model

### 🔹 Custom Loss Function
- Fungsi sederhana atau subclass `keras.losses.Loss`
- Contoh: Huber loss

### 🔹 Custom Metric
- Fungsi biasa atau subclass `keras.metrics.Metric`

### 🔹 Custom Layer
- Subclass dari `keras.layers.Layer`
- Bisa memuat layer lain di dalamnya

### 🔹 Custom Model
- Subclass dari `keras.Model`
- Berisi logika forward pass dan layer-layer internal

---

## 🔁 Custom Training Loop

Custom loop memberi kontrol penuh atas:
- Sampling batch
- Perhitungan forward pass dan loss
- Backpropagation via `tf.GradientTape`
- Update optimizer manual
- Logging metrik dan validasi

**Contoh struktur:**
```python
for epoch in range(epochs):
    for step in range(steps_per_epoch):
        with tf.GradientTape() as tape:
            y_pred = model(X_batch)
            loss = loss_fn(y_true, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## ⚡ TF Functions dan Graph Execution
@tf.function:
- Mengkonversi fungsi Python jadi computation graph
- Efisien, portabel, dan bisa diparalelkan

AutoGraph:
- Konversi struktur Python (for, if, while) menjadi graph
- Harus gunakan tf.range, bukan range

Aturan:
- Hindari print(), random, atau Python ops yang tidak konversi
- Semua operasi harus dalam bentuk TensorFlow ops

---

## 🎯 Kapan Harus Kustom?
- Arsitektur unik (loop, skip connections, dynamic branching)
- Pelatihan non-standar (multi-loss, multi-optimizer)
- Logging dan debugging manual
- Penelitian dan eksplorasi model baru

