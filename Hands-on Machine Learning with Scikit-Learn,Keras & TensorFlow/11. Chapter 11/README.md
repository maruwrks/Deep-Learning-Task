## Penjelasan Teoritis

---

### 1. Masalah Vanishing dan Exploding Gradients

Selama proses backpropagation, gradien yang dibawa mundur dari lapisan output ke input dapat mengecil (vanishing) atau membesar (exploding) secara drastis:

* **Vanishing gradients** menyebabkan pembaruan bobot menjadi sangat kecil, membuat pelatihan lambat atau berhenti sama sekali.
* **Exploding gradients** mengakibatkan pembaruan parameter menjadi sangat besar, sehingga pelatihan menjadi tidak stabil.

Kondisi ini umumnya diperparah oleh banyaknya lapisan dan fungsi aktivasi seperti sigmoid atau tanh.

### 2. Strategi Mengatasi Vanishing/Exploding Gradients

#### a. Inisialisasi Bobot

* **Glorot/Xavier Initialization:** Menyesuaikan varians bobot berdasarkan jumlah neuron input dan output.
* **He Initialization:** Cocok untuk aktivasi ReLU, memperhitungkan hanya jumlah neuron input.

#### b. Fungsi Aktivasi Non-saturasi

* **ReLU:** Fungsi umum, namun dapat mengalami "Dying ReLUs".
* **Leaky ReLU, PReLU:** Mengatasi kematian neuron dengan memberikan gradien kecil pada sisi negatif.
* **ELU dan SELU:** Dapat meningkatkan stabilitas pelatihan dan performa.

#### c. Normalisasi Batch

Batch Normalization menstabilkan distribusi input untuk setiap lapisan selama pelatihan. Hal ini membantu mempercepat pelatihan dan bertindak sebagai regularisasi.

#### d. Pemotongan Gradien (Gradient Clipping)

Teknik ini mencegah gradien menjadi terlalu besar dengan menetapkan batas atas tertentu, berguna khususnya pada RNN.

### 3. Transfer Learning

Memanfaatkan model yang sudah dilatih sebelumnya untuk tugas baru:

* **Lapisan awal dibekukan** karena biasanya sudah menangkap fitur-fitur generik.
* **Lapisan akhir diganti** dengan lapisan baru yang sesuai dengan tugas spesifik.
* **Fine-tuning** dilakukan dengan membuka sebagian lapisan untuk dilatih ulang dengan learning rate rendah.

### 4. Optimizer Canggih

Beberapa algoritma optimasi yang mempercepat pelatihan:

* **Momentum:** Menambahkan kecepatan dari gradien sebelumnya.
* **Nesterov Accelerated Gradient (NAG):** Memperkirakan posisi di masa depan untuk penghitungan gradien.
* **AdaGrad:** Mengatur learning rate berdasarkan frekuensi pembaruan bobot.
* **RMSProp:** Mengatasi penurunan learning rate berlebihan pada AdaGrad.
* **Adam:** Menggabungkan ide Momentum dan RMSProp.
* **Adamax, Nadam:** Varian lain dengan adaptasi khusus.

### 5. Regularisasi untuk Mencegah Overfitting

#### a. L1 dan L2

Penalti bobot pada fungsi loss:

* **L1 (Lasso):** Dorong bobot menjadi nol.
* **L2 (Ridge):** Dorong bobot menjadi kecil.

#### b. Dropout

Selama pelatihan, neuron acak dinonaktifkan sementara, mendorong jaringan menjadi lebih tangguh dan menghindari overfitting.

#### c. Alpha Dropout

Dropout yang dirancang agar tetap menjaga properti normalisasi diri pada SELU.

#### d. Monte Carlo Dropout

Melakukan prediksi beberapa kali dengan dropout aktif, digunakan untuk estimasi ketidakpastian.

#### e. Max-Norm Regularization

Membatasi norma bobot maksimum per neuron.

### 6. Panduan Kinerja

| Hyperparameter      | Nilai Umum         | Untuk Underfitting | Untuk Overfitting |
| ------------------- | ------------------ | ------------------ | ----------------- |
| Inisialisasi        | He (ReLU/ELU/SELU) |                    |                   |
| Fungsi Aktivasi     | ReLU, Softmax      |                    |                   |
| Optimizer           | Adam               | Tingkatkan LR      | Turunkan LR       |
| Learning Rate       | 1e-3 s/d 3e-3      | Naikkan            | Turunkan          |
| Batch Size          | 32                 |                    | Kurangi           |
| Hidden Layer        | 2 - 5              | Tambah             | Kurangi           |
| Neuron per Layer    | 10 - 100           | Tambah             | Kurangi           |
| L1/L2               | Tidak aktif        | Kurangi            | Tambah            |
| Dropout Rate        | 0.2 - 0.5          | Kurangi            | Tambah            |
| Batch Normalization | Direkomendasikan   |                    |                   |
| Early Stopping      | Direkomendasikan   |                    |                   |
