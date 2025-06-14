# Penjelasan Teoritis

---

## 1. **Persiapan Dataset**

Dataset Fashion MNIST dimuat dan dinormalisasi menjadi rentang \[0, 1]. Data dilatih, divalidasi, dan diuji. Fungsi bantu seperti `plot_image()` dan `show_reconstructions()` digunakan untuk visualisasi gambar asli dan hasil rekonstruksi.

---

## 2. **Stacked Denoising Autoencoder (DAE)**

### Arsitektur

* **Encoder**:

  * Flatten → GaussianNoise (penambahan noise) → Dense(100, SELU) → Dense(30, SELU)
* **Decoder**:

  * Dense(100, SELU) → Dense(784, sigmoid) → Reshape (28x28)

### Tujuan

Membangun autoencoder yang mampu menghilangkan noise dari input (denoising). Model dilatih menggunakan `binary_crossentropy` sebagai fungsi loss dan metrik MSE.

---

## 3. **Variational Autoencoder (VAE)**

### Komponen

* **Sampling Layer**: Mengambil sampel dari distribusi Gaussian berdasarkan mean dan log-variance (VAE trick).
* **Encoder**:

  * Input → Flatten → Dense(150) → Dense(100) → Output `codings_mean` dan `codings_log_var`
  * Gunakan `Sampling()` untuk menghasilkan latent vector `z`
* **Decoder**:

  * Dense(100) → Dense(150) → Dense(784, sigmoid) → Reshape (28x28)
* **Latent Loss**:

  * Regularisasi distribusi latent menggunakan KL-divergence:
    $\text{Loss}_{\text{KL}} = -\frac{1}{2} \sum(1 + \log \sigma^2 - \mu^2 - \sigma^2)$

### Tujuan

VAE belajar merepresentasikan data dalam distribusi probabilistik dan menghasilkan sampel baru dari distribusi tersebut.

---

## 4. **Deep Convolutional GAN (DCGAN)**

### Komponen

#### a. **Generator**

* Input: vektor acak 100 dimensi (noise).
* Dense → Reshape(7x7x128) → Conv2DTranspose (upscale) → BatchNorm → Conv2DTranspose → Output: 28x28x1 gambar.
* Aktivasi terakhir `tanh`, karena data diskalakan ke \[-1, 1].

#### b. **Discriminator**

* Input: gambar 28x28x1.
* Dua Conv2D dengan stride 2 → Dropout → Flatten → Dense(1, sigmoid)

### Proses Training

1. Generator menghasilkan gambar palsu dari noise.
2. Discriminator diberi gambar asli dan palsu, lalu dilatih untuk membedakannya.
3. Generator dilatih untuk "menipu" discriminator agar menganggap gambar palsu sebagai asli.

```text
Loss_discriminator: Binary crossentropy antara [real=1, fake=0]
Loss_generator: Binary crossentropy antara [fake images predicted as real=1]
```

### Loop Training GAN

* Per epoch:

  * Generator membuat batch gambar.
  * Discriminator dilatih dengan gabungan gambar asli dan palsu.
  * Generator dilatih untuk meningkatkan kemampuannya menghasilkan gambar meyakinkan.

### Visualisasi

Setelah setiap epoch, gambar dari generator divisualisasikan untuk mengevaluasi kualitas.

---
