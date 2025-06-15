# Penjelasan Teoritis

---

## 1. Persiapan dan Visualisasi Data

* Menggunakan data dari OECD dan GDP World Bank tahun 2015.
* Data difilter dan digabung berdasarkan negara.
* Visualisasi scatterplot menggambarkan korelasi positif antara GDP per kapita dan Life Satisfaction.

---

## 2. Linear Regression (Model Dasar)

* Model linier sederhana dilatih pada sebagian data (sample).
* Model digunakan untuk memprediksi Life Satisfaction negara Cyprus.

```python
lin_reg = LinearRegression()
lin_reg.fit(X_sample, y_sample)
```

* Visualisasi garis regresi menunjukkan kemampuan generalisasi model dasar.

---

## 3. Perbandingan Model

### a. Linear Regression (seluruh data)

* Model linier dilatih menggunakan seluruh dataset (bukan subset).

### b. Ridge Regression

* Regularisasi L2 diterapkan untuk menghindari overfitting pada data kecil.

```python
ridge = Ridge(alpha=10**9.5)
```

### c. Polynomial Regression (Overfitting)

* Model polinomial derajat tinggi (30) diterapkan.
* Pipeline preprocessing mencakup standarisasi fitur.

```python
PolynomialFeatures(degree=30) + StandardScaler + LinearRegression
```

* Hasil: Model terlalu cocok dengan data (overfit) dan buruk untuk generalisasi.

---

## Evaluasi Visual

* Visualisasi kurva prediksi membantu memahami kinerja dan kompleksitas model.
* Ridge menunjukkan trade-off antara bias dan varians.

---

## Konsep Kunci

| Konsep                | Penjelasan                                            |
| --------------------- | ----------------------------------------------------- |
| Regresi Linier        | Model dasar prediktif dengan satu fitur               |
| Regularisasi          | Teknik untuk mencegah overfitting                     |
| Overfitting           | Ketika model terlalu cocok dengan data latih          |
| Polynomial Regression | Model kompleks yang bisa overfit jika tidak dikontrol |

---
