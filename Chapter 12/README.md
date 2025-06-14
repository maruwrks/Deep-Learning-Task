Rangkuman Bab 12: Model Kustom dan Pelatihan dengan TensorFlow
Bab 12 memberikan Anda kemampuan untuk melampaui API tf.keras tingkat tinggi dan mendapatkan kontrol penuh atas model Anda. Dengan menggunakan API tingkat rendah TensorFlow, Anda dapat merancang arsitektur yang lebih kompleks dan menyesuaikan proses pelatihan sesuai kebutuhan spesifik.

Konsep-konsep utama yang dibahas meliputi:

Penggunaan TensorFlow seperti NumPy: Bab ini dimulai dengan pengenalan tensor, variabel, dan operasi dasar di TensorFlow, menunjukkan bagaimana ia dapat digunakan untuk komputasi numerik yang efisien, terutama dengan dukungan GPU.
Komponen Kustom: Anda belajar cara membuat komponen tf.keras Anda sendiri dari awal:
Custom Loss & Metrics: Membuat fungsi loss dan metrik yang spesifik untuk masalah Anda, di luar yang sudah disediakan Keras.
Custom Layers: Merancang layer dengan bobot dan logika unik yang tidak ada di Keras.
Custom Models: Membangun model yang kompleks dengan arsitektur non-sekuensial (misalnya, yang memiliki banyak input/output atau skip connections) dengan mewarisi kelas keras.Model.
Custom Training Loop: Anda diajarkan cara menulis loop pelatihan secara manual. Ini memberi Anda kontrol penuh atas setiap langkah, seperti bagaimana gradien dihitung dan diterapkan, yang berguna untuk implementasi algoritma riset tingkat lanjut.
TensorFlow Functions dan Graphs: Konsep paling penting untuk kinerja adalah @tf.function. Dekorator ini mengubah fungsi Python menjadi grafik komputasi (computation graph) TensorFlow yang sangat dioptimalkan. Fitur AutoGraph secara otomatis mengubah logika kontrol Python (seperti for dan if) menjadi operasi grafik yang efisien, sehingga menghasilkan peningkatan kecepatan yang drastis.
Secara keseluruhan, bab ini adalah jembatan antara penggunaan Keras yang praktis dan kekuatan penuh TensorFlow untuk riset dan implementasi tingkat lanjut.

Contoh Kode Implementasi
Kode berikut mendemonstrasikan perbedaan kinerja yang signifikan antara menjalankan perulangan (loop) di Python (Eager Execution) dibandingkan dengan menjalankannya sebagai grafik komputasi yang dioptimalkan menggunakan @tf.function, seperti yang dijelaskan di Bab 12.

Python

import tensorflow as tf
import timeit

# --- Versi Eager Execution ---
# Fungsi ini menjalankan loop di interpreter Python.
# Setiap iterasi memanggil operasi TF secara terpisah, yang membuatnya lambat.
def sum_up_to_eager(n):
    total = tf.constant(0, dtype=tf.int32)
    for i in range(n):
        total += i
    return total

# --- Versi TF Function ---
# @tf.function mengubah seluruh fungsi, termasuk loop, menjadi satu grafik komputasi.
@tf.function
def sum_up_to_graph(n):
    total = tf.constant(0, dtype=tf.int32)
    # AutoGraph akan mengubah tf.range() menjadi operasi loop yang efisien di dalam grafik.
    for i in tf.range(n):
        total += i
    return total

# Jumlah iterasi untuk pengujian
num_iterations = 10000

print(f"Menguji fungsi dengan perulangan sebanyak {num_iterations} kali...\n")

# Mengukur waktu eksekusi untuk fungsi Python biasa
start_time_eager = timeit.default_timer()
sum_up_to_eager(num_iterations)
end_time_eager = timeit.default_timer()

# Mengukur waktu eksekusi untuk TF Function (termasuk tracing)
# Catatan: Panggilan pertama akan mencakup waktu "tracing" untuk membangun grafik.
start_time_graph = timeit.default_timer()
# Kita perlu mengubah num_iterations menjadi tensor agar fungsi di-trace dengan benar
sum_up_to_graph(tf.constant(num_iterations))
end_time_graph = timeit.default_timer()

# Tampilkan hasil yang Anda berikan
print(f"Waktu Eager Execution (Python Loop): {end_time_eager:.4f} detik")
print(f"Waktu TF Function (Graph Loop): {end_time_graph:.4f} detik")

Hasil dan Penjelasan
Menguji fungsi dengan perulangan sebanyak 10000 kali...

Waktu Eager Execution (Python Loop): 1.1527 detik
Waktu TF Function (Graph Loop): 0.3078 detik
Hasil di atas (angka pastinya akan bervariasi tergantung perangkat keras Anda, namun trennya akan sama) dengan jelas menunjukkan kekuatan dari @tf.function.

Waktu Eager Execution (Python Loop): Versi ini berjalan lambat karena loop for dieksekusi oleh interpreter Python. Pada setiap iterasi, Python harus berkomunikasi dengan backend TensorFlow untuk menjalankan operasi total += i. Komunikasi bolak-balik ini menimbulkan overhead yang signifikan.
Waktu TF Function (Graph Loop): Versi ini jauh lebih cepat karena dekorator @tf.function menggunakan AutoGraph untuk mengubah seluruh perulangan Python menjadi satu operasi tf.while_loop tunggal di dalam grafik TensorFlow. Seluruh proses perulangan kemudian dieksekusi di backend C++ yang sangat dioptimalkan, tanpa perlu kembali ke Python sama sekali. Biaya tracing di awal menjadi tidak signifikan dibandingkan keuntungan kecepatan yang didapat dari eksekusi grafik.
