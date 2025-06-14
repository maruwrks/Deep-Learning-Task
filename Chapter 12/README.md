Konsep-konsep utama yang dibahas meliputi:

Penggunaan TensorFlow seperti NumPy: Bab ini dimulai dengan pengenalan tensor, variabel, dan operasi dasar di TensorFlow, menunjukkan bagaimana ia dapat digunakan untuk komputasi numerik yang efisien, terutama dengan dukungan GPU.
Komponen Kustom: Anda belajar cara membuat komponen tf.keras Anda sendiri dari awal:
Custom Loss & Metrics: Membuat fungsi loss dan metrik yang spesifik untuk masalah Anda, di luar yang sudah disediakan Keras.
Custom Layers: Merancang layer dengan bobot dan logika unik yang tidak ada di Keras.
Custom Models: Membangun model yang kompleks dengan arsitektur non-sekuensial (misalnya, yang memiliki banyak input/output atau skip connections) dengan mewarisi kelas keras.Model.
Custom Training Loop: Anda diajarkan cara menulis loop pelatihan secara manual. Ini memberi Anda kontrol penuh atas setiap langkah, seperti bagaimana gradien dihitung dan diterapkan, yang berguna untuk implementasi algoritma riset tingkat lanjut.
TensorFlow Functions dan Graphs: Konsep paling penting untuk kinerja adalah @tf.function. Dekorator ini mengubah fungsi Python menjadi grafik komputasi (computation graph) TensorFlow yang sangat dioptimalkan. Fitur AutoGraph secara otomatis mengubah logika kontrol Python (seperti for dan if) menjadi operasi grafik yang efisien, sehingga menghasilkan peningkatan kecepatan yang drastis.
Secara keseluruhan, bab ini adalah jembatan antara penggunaan Keras yang praktis dan kekuatan penuh TensorFlow untuk riset dan implementasi tingkat lanjut.


Hasil dan Penjelasan

Waktu Eager Execution (Python Loop): 1.1527 detik
Waktu TF Function (Graph Loop): 0.3078 detik
Hasil di atas (angka pastinya akan bervariasi tergantung perangkat keras Anda, namun trennya akan sama) dengan jelas menunjukkan kekuatan dari @tf.function.

Waktu Eager Execution (Python Loop): Versi ini berjalan lambat karena loop for dieksekusi oleh interpreter Python. Pada setiap iterasi, Python harus berkomunikasi dengan backend TensorFlow untuk menjalankan operasi total += i. Komunikasi bolak-balik ini menimbulkan overhead yang signifikan.
Waktu TF Function (Graph Loop): Versi ini jauh lebih cepat karena dekorator @tf.function menggunakan AutoGraph untuk mengubah seluruh perulangan Python menjadi satu operasi tf.while_loop tunggal di dalam grafik TensorFlow. Seluruh proses perulangan kemudian dieksekusi di backend C++ yang sangat dioptimalkan, tanpa perlu kembali ke Python sama sekali. Biaya tracing di awal menjadi tidak signifikan dibandingkan keuntungan kecepatan yang didapat dari eksekusi grafik.
