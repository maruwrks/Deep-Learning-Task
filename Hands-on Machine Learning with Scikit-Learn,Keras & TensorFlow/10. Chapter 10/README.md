## Penjelasan Teoritis

---

### Neuron Biologis vs Neuron Buatan

Neuron biologis memiliki struktur seperti dendrit (menerima sinyal), soma (mengakumulasi sinyal), dan akson (mengirimkan sinyal). Koneksi antar neuron melalui sinapsis, yang dapat diperkuat atau dilemahkan. Neuron buatan meniru perilaku ini secara matematis:

* Input diberi bobot.
* Dihitung jumlah total ditambah bias.
* Hasilnya dilewatkan melalui fungsi aktivasi.

### Perceptron dan Keterbatasannya

Perceptron adalah model neuron tunggal yang membuat keputusan biner berdasarkan fungsi aktivasi step. Bobot diperbarui berdasarkan kesalahan. Namun, model ini hanya dapat memisahkan data secara linier. Masalah seperti XOR tidak dapat diselesaikan dengan perceptron tunggal.

### Multi-Layer Perceptrons (MLPs)

MLP adalah jaringan berlapis yang terdiri dari:

* Lapisan input
* Beberapa lapisan tersembunyi (hidden layer)
* Lapisan output

Setiap neuron terhubung secara feedforward ke lapisan berikutnya. MLP dapat menyelesaikan masalah non-linier dan kompleks, terutama dengan menggunakan algoritma backpropagation.

### Algoritma Backpropagation

Pelatihan JST menggunakan backpropagation dilakukan melalui:

1. Forward pass untuk menghasilkan output.
2. Menghitung kesalahan.
3. Backward pass untuk menghitung gradien terhadap setiap parameter.
4. Update parameter menggunakan gradient descent.

Fungsi aktivasi:

* Sigmoid: antara 0-1, rawan vanishing gradient.
* tanh: antara -1 dan 1.
* ReLU: cepat, namun bisa menyebabkan neuron mati.
* Softmax: untuk klasifikasi multikelas.

### JST untuk Regresi dan Klasifikasi

* Regresi: Menggunakan satu neuron output dan fungsi aktivasi linier, serta loss function seperti MSE.
* Klasifikasi:

  * Biner: satu neuron output dengan sigmoid.
  * Multikelas: satu neuron per kelas dengan softmax.

### Membangun Model dengan Keras Sequential API

Langkah-langkah:

1. Normalisasi input.
2. `Flatten()` untuk input 2D.
3. Beberapa `Dense()` layer dengan `relu`.
4. Output layer menggunakan `softmax` untuk klasifikasi.
5. Kompilasi model dengan `compile()`.
6. Pelatihan dengan `fit()`.
7. Evaluasi dengan `evaluate()`.
8. Prediksi dengan `predict()`.

### Functional API dan Subclassing API

* Functional API memungkinkan koneksi kompleks antar layer.
* Subclassing API digunakan untuk arsitektur dinamis atau model dengan logika kompleks.

### Menyimpan dan Memuat Model

Model dapat disimpan dengan `model.save()` dan dimuat kembali dengan `load_model()` untuk prediksi lanjutan atau melanjutkan pelatihan.

### Penggunaan Callbacks

Callbacks digunakan untuk:

* Early stopping
* Menyimpan checkpoint model terbaik
* Logging

Contoh: `ModelCheckpoint`, `EarlyStopping`, dan integrasi TensorBoard.

### Visualisasi dengan TensorBoard

TensorBoard digunakan untuk:

* Melihat kurva pelatihan
* Visualisasi arsitektur model
* Monitoring metrik lainnya

Aktifkan dengan `TensorBoard()` callback dan jalankan dengan `tensorboard --logdir=...`

### Penyetelan Hyperparameter

Beberapa hyperparameter penting:

* Jumlah layer
* Jumlah neuron
* Fungsi aktivasi
* Optimizer (SGD, Adam, dll.)
* Learning rate
* Batch size dan epoch

Dapat disetel dengan:

* GridSearchCV / RandomizedSearchCV
* Keras Tuner
