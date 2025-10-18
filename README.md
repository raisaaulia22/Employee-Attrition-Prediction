# Employee-Attrition-Prediction
Project SML

# Daftar Isi

- [Domain Projek: Sumber Daya Manusia](#domain-projek)
  - [Referensi](#referensi)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Statements](#solution-statements)
  - [Project Benefits](#project-benefits)
- [Data Understanding](#data-understanding)
  - [Sumber Data](#sumber-data)
  - [Deskripsi Fitur](#deskripsi-fitur)
  - [Penjelasan Kontekstual Fitur](#penjelasan-kontekstual-fitur)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    - [Missing Value & Outliers](#missing-value--outliers)
    - [Univariate Analysis](#univariate-analysis)
    - [Multivariate Analysis](#multivariate-analysis)
    - [Kesimpulan EDA](#kesimpulan-eda)
- [Data Preparation](#data-preparation)
  - [Label Encoding dengan Mapping pada Fitur Target](#label-encoding-dengan-mapping-pada-fitur-target)
  - [Splitting Dataset](#splitting-dataset)
  - [Feature Engineering, Data Cleaning and Preprocessing](#feature-engineering-data-cleaning-and-preprocessing)
- [Model Training, Comparison, Selection and Tuning](#model-training-comparison-selection-and-tuning)
  - [Model Selection](#model-selection)
  - [Feature Selection](#feature-selection)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Testing and Evaluation](#model-testing-and-evaluation)
  - [Data Test Predict](#data-test-predict)
  - [Best Model Evaluation](#best-model-evaluation)
    - [Classification Report](#classification-report)
    - [Metode Evaluasi Lanjutan](#metode-evaluasi-lanjutan)
    - [Confusion Matrix](#confusion-matrix)
    - [Plot ROC-AUC Curve](#plot-roc-auc-curve)
    - [Plot PR-AUC Curve](#plot-pr-auc-curve)
- [Save Best Model](#save-best-model)
- [Model Interpretation](#model-interpretation)
  - [Interpretation](#interpretation)
  - [Feature Importance](#feature-importance)
- [Financial Result](#financial-result)
- [Conclusions](#conclusions)
  - [Ringkasan Proyek](#ringkasan-proyek)
  - [Hasil dan Evaluasi Model](#hasil-dan-evaluasi-model)
  - [Penanganan Ketidakseimbangan Data](#penanganan-ketidakseimbangan-data)
  - [Interpretasi dan Validasi Model](#interpretasi-dan-validasi-model)
  - [Estimasi Nilai Finansial](#estimasi-nilai-finansial)
  - [Langkah Selanjutnya](#langkah-selanjutnya)



## Domain Projek: Sumber Daya Manusia
<a id="domain-projek"></a>
cldjnehbejhw

### Referensi
<a id="referensi"></a>
Isi konten referensi di sini...

## Business Understanding
<a id="business-understanding"></a>

### Problem Statements
<a id="problem-statements"></a>
Perusahaan mengalami tantangan dalam mempertahankan talenta berharga akibat tingginya angka employee attrition yang tidak terprediksi, mengakibatkan kerugian finansial yang signifikan dan gangguan operasional, sementara faktor-faktor penyebab utama keputusan karyawan untuk mengundurkan diri belum sepenuhnya dipahami.

Berdasarkan hal tersebut, berikut adalah pernyataan masalah yang diangkat:
- **Pernyataan Masalah 1** : Bagaimana mengidentifikasi faktor-faktor penting yang memengaruhi keputusan karyawan untuk keluar dari perusahaan?
- **Pernyataan Masalah 2** : Bagaimana membangun model prediksi yang mampu memperkirakan kemungkinan seorang karyawan akan keluar dari perusahaan dengan tingkat akurasi tinggi?
- **Pernyataan Masalah 3** : Bagaimana menyusun strategi berbasis data untuk menurunkan angka attrition berdasarkan profil dan karakteristik karyawan berisiko tinggi?

### Goals
<a id="goals"></a>
Untuk menjawab pernyataan masalah di atas, tujuan proyek ini dirumuskan sebagai berikut :
- **Tujuan 1** : Melakukan eksplorasi dan analisis data karyawan untuk mengidentifikasi pola dan fitur yang memengaruhi angka attrition karyawan.
- **Tujuan 2** : Membangun model prediktif berbasis machine learning yang mampu menghitung probabilitas attrition dari masing-masing karyawan.
- **Tujuan 3** : Memberikan rekomendasi dan rencana aksi yang berbasis pada hasil prediksi model untuk menurunkan angka attrition karyawan.

### Solution Statements
<a id="solution-statements"></a>
Untuk mencapai tujuan-tujuan tersebut, solusi yang akan diimplementasikan meliputi:
- **Eksperimen Berbagai Algoritma Klasifikasi :**
  Membangun dan membandingkan performa beberapa algoritma seperti :
  -  A
  -  B
  -  C
- **Optimasi Model dengan Hyperparameter Tuning :**
  Menggunakan pendekatan seperti Bayesian Optimization dengan optuna untuk mendapatkan   konfigurasi model terbaik.
- **Evaluasi Model dengan Metrik yang Relevan :**
  Menggunakan metrik seperti :
  - Accuracy untuk mengukur prediksi keseluruhan
  - Precision, Recall, F1-Score untuk menilai performa pada kelas attrition
  - ROC-AUC untuk mengevaluasi kemampuan model dalam membedakan kelas
  - Confusion Matrix untuk melihat distribusi hasil prediksi
- **Analisis Fitur dan Visualisasi :**
  Menyajikan visualisasi seperti feature importance dan correlation heatmap untuk menginterpretasikan fitur-fitur utama yang berkontribusi terhadap attrition.

### Project Benefits
<a id="project-benefits"></a>
Dengan implementasi solusi ini, manfaat utama yang diharapkan antara lain:
- **Pengurangan Biaya Operasional :** Mengurangi biaya rekrutmen dan training melalui pencegahan dini karyawan berisiko keluar.
- **Peningkatan Produktivitas :** Mempertahankan talenta kunci dan mengurangi gangguan kerja akibat pergantian karyawan, meningkatkan stabilitas tim dan kontinuitas operasional bisnis.
- **Data-Driven Decision Making :** Memberikan insights berbasis data untuk pengambilan keputusan strategis departemen HR dalam merancang program retensi yang lebih efektif dan terukur.

## Data Understanding
<a id="data-understanding"></a>

### Sumber Data
<a id="sumber-data"></a>
Dataset yang digunakan dalam proyek ini diperoleh dari situs [Kaggle](https://www.kaggle.com/competitions/tugas-1-sml-a-2025/data). Dataset ini mencakup informasi tentang 1.173 karyawan, yang mencatat berbagai aspek demografis dan profil karakteristik karyawan.

Dataset ini memiliki 34 fitur, yang mencakup .............. dan lainnya. Di antara seluruh karyawan, hanya sekitar 16,07% yang termasuk dalam kategori churn (berhenti menggunakan layanan). Ketidakseimbangan kelas ini menjadikan proses pelatihan model prediktif sebagai tantangan tersendiri.

### Deskripsi Fitur
<a id="deskripsi-fitur"></a>
Isi deskripsi fitur di sini...

### Penjelasan Kontekstual Fitur
<a id="penjelasan-kontekstual-fitur"></a>
Isi penjelasan kontekstual di sini...

## Exploratory Data Analysis (EDA)
<a id="exploratory-data-analysis-eda"></a>

### Missing Value & Outliers
<a id="missing-value--outliers"></a>
Isi analisis missing value & outliers di sini...

### Univariate Analysis
<a id="univariate-analysis"></a>
Isi univariate analysis di sini...

### Multivariate Analysis
<a id="multivariate-analysis"></a>
Isi multivariate analysis di sini...

### Kesimpulan EDA
<a id="kesimpulan-eda"></a>
Isi kesimpulan EDA di sini...


## Data Preparation
<a id="data-preparation"></a>

### Label Encoding dengan Mapping pada Fitur Target
<a id="label-encoding-dengan-mapping-pada-fitur-target"></a>
Konten label encoding...

### Splitting Dataset
<a id="splitting-dataset"></a>
Konten splitting dataset...

### Feature Engineering, Data Cleaning and Preprocessing
<a id="feature-engineering-data-cleaning-and-preprocessing"></a>
Konten feature engineering...

## Model Training, Comparison, Selection and Tuning
<a id="model-training-comparison-selection-and-tuning"></a>

### Model Selection
<a id="model-selection"></a>
Konten model selection...

### Feature Selection
<a id="feature-selection"></a>
Konten feature selection...

### Hyperparameter Tuning
<a id="hyperparameter-tuning"></a>
Konten hyperparameter tuning...

## Model Testing and Evaluation
<a id="model-testing-and-evaluation"></a>

### Data Test Predict
<a id="data-test-predict"></a>
Konten data test predict...

### Best Model Evaluation
<a id="best-model-evaluation"></a>

#### Classification Report
<a id="classification-report"></a>
Konten classification report...

#### Metode Evaluasi Lanjutan
<a id="metode-evaluasi-lanjutan"></a>
Konten metode evaluasi lanjutan...

#### Confusion Matrix
<a id="confusion-matrix"></a>
Konten confusion matrix...

#### Plot ROC-AUC Curve
<a id="plot-roc-auc-curve"></a>
Konten plot ROC-AUC curve...

#### Plot PR-AUC Curve
<a id="plot-pr-auc-curve"></a>
Konten plot PR-AUC curve...

## Save Best Model
<a id="save-best-model"></a>
Konten save best model...

## Model Interpretation
<a id="model-interpretation"></a>

### Interpretation
<a id="interpretation"></a>
Konten interpretation...

### Feature Importance
<a id="feature-importance"></a>
Konten feature importance...

## Financial Result
<a id="financial-result"></a>
Konten financial result...

## Conclusions
<a id="conclusions"></a>

### Ringkasan Proyek
<a id="ringkasan-proyek"></a>
Konten ringkasan proyek...

### Hasil dan Evaluasi Model
<a id="hasil-dan-evaluasi-model"></a>
Konten hasil dan evaluasi model...

### Penanganan Ketidakseimbangan Data
<a id="penanganan-ketidakseimbangan-data"></a>
Konten penanganan ketidakseimbangan data...

### Interpretasi dan Validasi Model
<a id="interpretasi-dan-validasi-model"></a>
Konten interpretasi dan validasi model....

### Estimasi Nilai Finansial
<a id="estimasi-nilai-finansial"></a>
Konten estimasi nilai finansial...

### Langkah Selanjutnya
<a id="langkah-selanjutnya"></a>
Konten langkah selanjutnya...
