# Penjelasan Teoritis

---

## 1. **Policy Gradient (PG)**

### Tujuan

Melatih model secara langsung untuk mempelajari *kebijakan* (policy), yaitu distribusi probabilitas atas tindakan pada setiap keadaan.

### Arsitektur Model

```python
model_pg = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(n_outputs, activation="softmax")
])
```

Model menghasilkan distribusi probabilitas aksi (`softmax`).

### Proses Pembelajaran:

1. **play\_one\_step**:

   * Mengambil satu langkah dari environment.
   * Menghitung loss terhadap aksi yang diambil, dan memperoleh gradien parameter.

2. **play\_multiple\_episodes**:

   * Melakukan iterasi beberapa episode dan menyimpan semua reward & gradien.

3. **discount\_rewards**:

   * Menghitung reward terdiskonto untuk mempertimbangkan masa depan.

4. **discount\_and\_normalize\_rewards**:

   * Menormalkan reward untuk stabilitas pelatihan.

5. **Training Loop**:

   * Menggabungkan gradien dari semua episode secara tertimbang oleh reward terdiskonto.
   * Gradien diterapkan pada parameter model.

### Visualisasi:

Kurva reward rata-rata ditampilkan selama pelatihan untuk melihat perkembangan kebijakan.

---

## 2. **Deep Q-Network (DQN)**

### Tujuan

Belajar fungsi nilai aksi (*action-value function*, atau Q-value) dan memilih aksi berdasarkan nilai tertinggi.

### Arsitektur Model

```python
model_dqn = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])
```

Model meng-output Q-values untuk semua aksi dalam satu langkah.

### Strategi:

1. **Epsilon-Greedy Policy**:

   * Memungkinkan eksplorasi aksi secara acak untuk menghindari konvergensi lokal.
   * Epsilon menurun selama pelatihan untuk eksploitasi lebih besar.

2. **Replay Memory**:

   * Menghindari korelasi antar langkah dengan menyimpan pengalaman sebelumnya.
   * Sampling acak pengalaman untuk pembaruan model.

3. **Training Step**:

   * Target Q dihitung dari reward + diskonto \* max Q pada next state.
   * Loss antara Q saat ini vs target Q dihitung dan digunakan untuk backpropagation.

4. **Training Loop**:

   * Setiap episode dilakukan, reward dikumpulkan.
   * Setelah sejumlah episode, parameter diperbarui dari replay buffer.

---

## Perbandingan PG vs DQN

| Aspek                | Policy Gradient                     | Deep Q-Network                |
| -------------------- | ----------------------------------- | ----------------------------- |
| Output               | Distribusi probabilitas (policy)    | Nilai Q untuk setiap aksi     |
| Tipe optimisasi      | Langsung optimisasi reward          | Optimisasi Q-value (indirect) |
| Stabilitas pelatihan | Bisa tidak stabil tanpa normalisasi | Stabil dengan replay + target |
| Eksplorasi           | Terjadi melalui sampling policy     | Eksplisit via epsilon-greedy  |

---

## Evaluasi dan Visualisasi

Setelah pelatihan, kedua agen diuji secara deterministik (tanpa eksplorasi). Lingkungan divisualisasikan dengan `env.render()` untuk melihat performa.

---

## Kesimpulan

Kode ini mencakup dua metode RL:

* **Policy Gradient** untuk pendekatan berbasis optimisasi langsung kebijakan.
* **Deep Q-Network** untuk pendekatan berbasis nilai (value-based).

Keduanya memiliki peran penting dalam membangun agen cerdas yang dapat belajar dari trial-and-error di lingkungan interaktif.
