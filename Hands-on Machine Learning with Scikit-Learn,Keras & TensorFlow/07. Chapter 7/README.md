# Penjelasan Teoritis

---

## Voting Classifiers

*Voting Classifier* adalah model ansambel yang menggabungkan prediksi dari beberapa model (disebut estimator dasar) untuk membuat keputusan akhir. Dua jenis voting yang umum digunakan:

* **Hard Voting:** Klasifikasi berdasarkan mayoritas suara dari prediksi model-model individu.
* **Soft Voting:** Menggunakan probabilitas prediksi dari setiap model, lalu memilih kelas dengan probabilitas rata-rata tertinggi.

Contoh diberikan menggunakan dataset "moons" untuk mengilustrasikan perbedaan performa antara model individual seperti `LogisticRegression`, `RandomForestClassifier`, dan `SVC`, serta `VotingClassifier` baik untuk hard maupun soft voting.

## Bagging dan Pasting

*Bagging (Bootstrap Aggregating)* melatih model yang sama pada subset data pelatihan yang dipilih secara acak dengan pengembalian. *Pasting* adalah varian dari bagging tanpa pengembalian.

`BaggingClassifier` digunakan untuk membungkus `DecisionTreeClassifier` sebanyak 500 estimator. Dibandingkan dengan satu pohon keputusan, model bagging menunjukkan akurasi lebih tinggi karena mengurangi varians.

### Out-of-Bag Evaluation

Fitur `oob_score=True` memungkinkan evaluasi performa model menggunakan data yang tidak termasuk dalam sampel bootstrap. Ini memberikan estimasi akurasi tanpa memerlukan data validasi terpisah.

## Random Forest

Random Forest adalah ansambel dari pohon keputusan yang dibentuk dengan bagging dan penambahan losion fitur acak saat pemilihan split. Hal ini membuat model lebih kuat terhadap overfitting.

`RandomForestClassifier` digunakan dengan parameter `max_leaf_nodes` untuk membatasi kompleksitas model.

### Feature Importance

Model Random Forest dapat menghitung pentingnya fitur berdasarkan seberapa besar kontribusi fitur dalam mengurangi impurity secara rata-rata.

Contoh diberikan pada dataset Iris dan MNIST. Pada MNIST, importance divisualisasikan sebagai heatmap untuk melihat area pixel yang berpengaruh dalam klasifikasi.

## Boosting

*Boosting* adalah teknik ansambel di mana model dilatih secara berurutan dan setiap model baru berusaha mengoreksi kesalahan model sebelumnya.

### AdaBoost

Menggunakan model dasar yang lemah (contoh: pohon keputusan kedalaman 1) dan memberi bobot lebih pada instance yang sulit diklasifikasikan. Implementasi menggunakan `AdaBoostClassifier` dengan algoritma "SAMME".

### Gradient Boosting

Gradient Boosting membangun model secara bertahap, dengan model baru difit ke residual dari model sebelumnya. `GradientBoostingRegressor` digunakan untuk regresi dengan data sintetis berbentuk kuadratik.

Perbandingan antara boosting cepat (sedikit estimator, learning rate besar) dan boosting lambat (lebih banyak estimator, learning rate kecil) juga ditunjukkan.

### Early Stopping

Early stopping menghentikan proses pelatihan ketika performa pada data validasi mulai memburuk. Ini dilakukan dengan memantau error dari `staged_predict()` dan memilih jumlah estimator optimal (`bst_n_estimators`). Model terbaik kemudian dilatih ulang dengan jumlah estimator tersebut.
