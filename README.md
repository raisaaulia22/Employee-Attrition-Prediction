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
Attrition karyawan merujuk pada keluarnya karyawan dari suatu organisasi akibat pengunduran diri, pensiun, atau pemecatan. Meskipun sebagian pengurangan karyawan tidak dapat dihindari, tingkat pergantian karyawan yang berlebihan dapat mengganggu operasional bisnis secara signifikan, menghambat pertumbuhan, dan berdampak pada kesuksesan jangka panjang. Pengurangan karyawan telah menjadi salah satu masalah paling mendesak bagi organisasi modern, terutama di industri yang kompetitif di mana mempertahankan talenta terbaik sangat penting untuk mempertahankan keunggulan kompetitif. [1](https://www.questjournals.org/jrhss/papers/vol13-issue3/1303122126.pdf)

Faktor penyebab utama keputusan karyawan untuk mengundurkan diri meliputi ketidakpuasan terhadap kompensasi, tingginya tingkat stres kerja, terbatasnya peluang pengembangan karier, serta manajemen yang kurang efektif. Selain itu, budaya kerja yang tidak mendukung dan kurangnya pengakuan atas kontribusi karyawan turut memperparah tingkat attrition. Oleh karena itu, perusahaan perlu menerapkan strategi retensi holistik yang mencakup peningkatan kesejahteraan karyawan, pengembangan kepemimpinan, serta program pengembangan karier yang terstruktur agar dapat mempertahankan tenaga kerja yang berkualitas dan mendukung keberlanjutan bisnisnya secara efektif. [1](https://www.questjournals.org/jrhss/papers/vol13-issue3/1303122126.pdf)

Proyek ini berada dalam domain analisis karakteristik karyawan dan manajemen risiko keuangan, dengan fokus pada pengembangan model prediksi berbasis data untuk mengidentifikasi karyawan yang berpotensi mengundurkan diri dari perusahaan. Dengan memanfaatkan teknik data science dan machine learning, bank dapat memprediksi employee attrition secara lebih akurat dan melakukan intervensi yang bersifat proaktif untuk mempertahankan karyawan dan meminimalisir kerugian finansial yang mungkin timbul.

### Referensi
<a id="referensi"></a>
[1] Kaushik, R. D. & Dubey, N. (2025). _Study Employee Attrition and Its Footprint on Organization Performance_. Quest Journals Journal of Research in Humanities and Social Science, 13(3), 122-126. https://doi.org/10.35629/9467-1303122126

## Business Understanding
<a id="business-understanding"></a>

### Problem Statements
<a id="problem-statements"></a>
Karyawan dinilai menjadi aset perusahaan karena kemampuan, pengetahuan, dan keterampilan mereka sangat penting untuk menggerakkan dan mengembangkan perusahaan. Namun, perusahaan mengalami tantangan dalam mempertahankan talenta berharga karyawan. Tingginya angka employee attrition tidak hanya berdampak pada kerugian finansial dan gangguan operasional, tetapi juga meningkatkan beban biaya perekrutan karyawan baru.

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
Data yang digunakan pada Tugas 1 SML A 2025 merupakan dataset didapat dari kaggle (https://www.kaggle.com/competitions/tugas-1-sml-a-2025). Data berfokus pada analisis attrition karyawan. Dataset ini terbagi menjadi dua bagian utama, yaitu data training yang terdiri dari 1.176 sampel dengan 36 fitur dan data testing yang berisi 294 sampel dengan 35 fitur. Perbedaan jumlah fitur tersebut disebabkan oleh adanya variabel target "Attrition" yang hanya terdapat pada data training. Variabel target ini memiliki format biner (0 dan 1) yang menunjukkan status karyawan tetap atau keluar dari perusahaan.

Dataset ini merupakan kumpulan data sumber daya manusia yang komprehensif untuk analisis attrition karyawan. Data tersebut mencakup berbagai aspek karakteristik personal, kondisi pekerjaan, serta faktor-faktor organisasional yang berpotensi memengaruhi keputusan karyawan untuk bertahan atau keluar dari perusahaan. Variabel-variabel dalam dataset merepresentasikan dimensi demografis, kompensasi finansial, perkembangan karir, kepuasan kerja, work-life balance, serta hubungan dengan manajemen. Keberagaman atribut ini memungkinkan dilakukannya analisis mendalam terhadap faktor-faktor prediktif yang berkontribusi terhadap fenomena attrition dalam lingkungan organisasi.

## Deskripsi Variabel Dataset

| Nama Fitur | Deskripsi | Tipe Data |
|------------|-----------|-----------|
| id | ID unik karyawan untuk identifikasi | int64 |
| Age | Usia karyawan dalam tahun | int64 |
| BusinessTravel | Frekuensi perjalanan dinas karyawan | object |
| DailyRate | Gaji harian karyawan | int64 |
| Department | Departemen tempat karyawan bekerja | object |
| DistanceFromHome | Jarak tempat tinggal ke kantor (km) | int64 |
| Education | Tingkat pendidikan terakhir (skala 1-5) | int64 |
| EducationField | Bidang studi terakhir karyawan | object |
| EmployeeCount | Jumlah karyawan (selalu 1 dalam dataset) | int64 |
| EmployeeNumber | Nomor unik karyawan dalam sistem HR | int64 |
| EnvironmentSatisfaction | Tingkat kepuasan terhadap lingkungan kerja (skala 1-4) | int64 |
| Gender | Jenis kelamin karyawan | object |
| HourlyRate | Upah per jam karyawan | int64 |
| JobInvolvement | Tingkat keterlibatan pekerjaan (skala 1-4) | int64 |
| JobLevel | Level jabatan karyawan | int64 |
| JobRole | Posisi/jabatan spesifik karyawan | object |
| JobSatisfaction | Tingkat kepuasan pekerjaan (skala 1-4) | int64 |
| MaritalStatus | Status pernikahan karyawan | object |
| MonthlyIncome | Gaji bulanan karyawan | int64 |
| MonthlyRate | Tarif bulanan karyawan | int64 |
| NumCompaniesWorked | Jumlah perusahaan tempat pernah bekerja | int64 |
| Over18 | Status usia di atas 18 tahun (selalu Y dalam dataset) | object |
| OverTime | Status lembur karyawan | object |
| PercentSalaryHike | Persentase kenaikan gaji tahunan terakhir | int64 |
| PerformanceRating | Penilaian kinerja terakhir (skala 1-4) | int64 |
| RelationshipSatisfaction | Tingkat kepuasan terhadap hubungan kerja (skala 1-4) | int64 |
| StandardHours | Jam kerja standar (selalu 80 dalam dataset) | int64 |
| StockOptionLevel | Level kepemilikan saham perusahaan | int64 |
| TotalWorkingYears | Total tahun pengalaman kerja | int64 |
| TrainingTimesLastYear | Jumlah pelatihan yang diikuti dalam setahun terakhir | int64 |
| WorkLifeBalance | Tingkat keseimbangan kerja-hidup (skala 1-4) | int64 |
| YearsAtCompany | Total tahun bekerja di perusahaan saat ini | int64 |
| YearsInCurrentRole | Total tahun di posisi saat ini | int64 |
| YearsSinceLastPromotion | Tahun sejak promosi terakhir | int64 |
| YearsWithCurrManager | Tahun bekerja dengan manajer saat ini | int64 |
| Attrition | Status karyawan keluar dari perusahaan (1=Yes/0=No) | int64 |


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
