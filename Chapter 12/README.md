Bab ini membahas bagaimana menggunakan TensorFlow secara lebih fleksibel dan mendalam dibandingkan pendekatan standar menggunakan tf.keras. Tujuan utamanya adalah memberikan kontrol penuh kepada pengguna terhadap proses pembuatan model, fungsi aktivasi, metrik, algoritma pelatihan, dan sebagainya.

1. Pengantar TensorFlow
TensorFlow adalah framework untuk komputasi numerik yang mendukung eksekusi pada GPU/TPU dan komputasi terdistribusi. Di bab ini, TensorFlow digunakan untuk:

Membuat model secara manual

Mengelola tensor dan variabel seperti di NumPy

Menyusun fungsi-fungsi komputasi sebagai graph (menggunakan @tf.function)

2. TensorFlow seperti NumPy
TensorFlow mendukung operasi numerik seperti NumPy, namun dengan kelebihan berupa kemampuan menjalankan operasi di berbagai perangkat keras. Konsep dasar seperti tensor, operasi matematis, slicing, dan broadcasting dibahas di sini. TensorFlow juga memiliki struktur data tambahan seperti tf.Variable (untuk nilai yang dapat berubah).

3. Kustomisasi Komponen Model
Bagian ini menjelaskan cara membuat komponen-komponen berikut secara manual:

Fungsi Loss Kustom: Digunakan saat loss standar tidak mencukupi, misalnya Huber loss.

Fungsi Aktivasi, Inisialisasi, dan Regularisasi Kustom: Dibuat dengan mendefinisikan fungsi atau class baru.

Metrik Kustom: Berguna untuk evaluasi model berdasarkan logika yang lebih spesifik.

Layer Kustom: Dengan mewarisi tf.keras.layers.Layer, pengguna dapat membuat arsitektur unik.

Model Kustom: Dengan mewarisi tf.keras.Model, pengguna bisa mendefinisikan model yang kompleks, misalnya dengan banyak cabang atau residual connections.

4. Training Loop Kustom
Bagian ini menjelaskan bagaimana membuat loop pelatihan manual, alih-alih menggunakan .fit() dari Keras. Ini memberi fleksibilitas penuh dalam mengatur:

Forward pass

Perhitungan loss

Perhitungan dan penerapan gradien

Logging metrik dan checkpoint

5. Autodiff dan Gradient Tape
TensorFlow menyediakan tf.GradientTape, sebuah API untuk menghitung gradien secara otomatis. Ini penting saat membuat loop training manual dan memungkinkan proses backpropagation dilakukan secara eksplisit.

6. TensorFlow Function dan Graph
Dengan @tf.function, pengguna dapat mengubah fungsi Python biasa menjadi computation graph untuk optimisasi performa. Fitur ini mendukung "AutoGraph" yang mampu mengkonversi struktur Python seperti if, for, dan while.

7. Kapan Harus Menggunakan Pendekatan Kustom?
Pendekatan ini direkomendasikan saat:

Membangun model arsitektur unik yang tidak bisa diwujudkan dengan Sequential atau Functional API

Membutuhkan logika pelatihan khusus, misalnya penggunaan beberapa optimizer

Melakukan riset atau eksplorasi arsitektur baru

Melakukan debugging model dengan cara manual
