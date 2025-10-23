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
<a id="Deskripsi Variabel Dataset"></a>

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
<a id="Penjelasan Kontekstual Fitur"></a>
**A. Kelompok Karir dan Kompensasi:**
- *JobLevel*, *MonthlyIncome*, *StockOptionLevel* - merepresentasikan hierarki jabatan dan struktur kompensasi
- *Department*, *JobRole* - menggambarkan divisi organisasi dan spesialisasi peran
- *PercentSalaryHike*, *PerformanceRating* - menunjukkan sistem reward dan pengakuan kinerja

**B. Kelompok Kepuasan dan Engagement:**
- *JobSatisfaction*, *EnvironmentSatisfaction*, *RelationshipSatisfaction* - mengukur tingkat engagement karyawan dari berbagai aspek
- *JobInvolvement* - mencerminkan keterlibatan dalam pekerjaan

**C. Kelompok Pengembangan Karir:**
- *YearsAtCompany*, *YearsInCurrentRole*, *YearsSinceLastPromotion* - menunjukkan mobilitas dan perkembangan karir
- *TrainingTimesLastYear* - merepresentasikan investasi pengembangan kompetensi
- *NumCompaniesWorked*, *TotalWorkingYears* - menggambarkan pengalaman kerja eksternal

**D. Kelompok Work-Life Balance:**
- *WorkLifeBalance*, *OverTime* - mengukur tekanan beban kerja dan keseimbangan hidup
- *BusinessTravel* - mempengaruhi intensitas mobilitas pekerjaan

**E. Kelompok Demografis dan Personal:**
- *Age*, *Gender*, *MaritalStatus* - memberikan konteks latar belakang personal
- *Education*, *EducationField* - merepresentasikan kualifikasi akademik
- *DistanceFromHome* - mempengaruhi faktor logistik harian

**F. Kelompok Kompensasi Finansial:**
- *MonthlyRate*, *DailyRate*, *HourlyRate* - berbagai metrik kompensasi finansial

Setiap kelompok fitur saling berinteraksi dalam mempengaruhi keputusan karyawan untuk bertahan atau keluar dari perusahaan (*Attrition*).

## Exploratory Data Analysis (EDA)
<a id="exploratory-data-analysis-eda"></a>
| Fitur | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
|-------|-------|------|-----|-----|-----|-----|-----|-----|
| Age | 1,176.00 | 36.9983 | 9.1781 | 18.0000 | 30.0000 | 36.0000 | 43.0000 | 60.0000 |
| DailyRate | 1,176.00 | 803.9915 | 401.3394 | 103.0000 | 467.7500 | 799.5000 | 1,157.00 | 1,499.00 |
| DistanceFromHome | 1,176.00 | 9.3580 | 8.1798 | 1.0000 | 2.0000 | 7.0000 | 14.0000 | 29.0000 |
| Education | 1,176.00 | 2.9065 | 1.0280 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 5.0000 |
| EmployeeCount | 1,176.00 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| EmployeeNumber | 1,176.00 | 1,015.83 | 599.6574 | 1.0000 | 487.7500 | 1,004.50 | 1,547.25 | 2,062.00 |
| EnvironmentSatisfaction | 1,176.00 | 2.7168 | 1.0887 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 4.0000 |
| HourlyRate | 1,176.00 | 65.5000 | 20.3733 | 30.0000 | 48.0000 | 66.0000 | 83.0000 | 100.0000 |
| JobInvolvement | 1,176.00 | 2.7372 | 0.7037 | 1.0000 | 2.0000 | 3.0000 | 3.0000 | 4.0000 |
| JobLevel | 1,176.00 | 2.0765 | 1.0920 | 1.0000 | 1.0000 | 2.0000 | 3.0000 | 5.0000 |
| JobSatisfaction | 1,176.00 | 2.7194 | 1.1106 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 4.0000 |
| MonthlyIncome | 1,176.00 | 6,544.02 | 4,653.74 | 1,009.00 | 2,948.00 | 5,004.50 | 8,420.50 | 19,973.00 |
| MonthlyRate | 1,176.00 | 14,390.24 | 7,192.83 | 2,094.00 | 8,051.00 | 14,373.00 | 20,770.75 | 26,999.00 |
| NumCompaniesWorked | 1,176.00 | 2.6930 | 2.4861 | 0.0000 | 1.0000 | 2.0000 | 4.0000 | 9.0000 |
| PercentSalaryHike | 1,176.00 | 15.2398 | 3.6791 | 11.0000 | 12.0000 | 14.0000 | 18.0000 | 25.0000 |
| PerformanceRating | 1,176.00 | 3.1573 | 0.3643 | 3.0000 | 3.0000 | 3.0000 | 3.0000 | 4.0000 |
| RelationshipSatisfaction | 1,176.00 | 2.7389 | 1.0872 | 1.0000 | 2.0000 | 3.0000 | 4.0000 | 4.0000 |
| StandardHours | 1,176.00 | 80.0000 | 0.0000 | 80.0000 | 80.0000 | 80.0000 | 80.0000 | 80.0000 |
| StockOptionLevel | 1,176.00 | 0.7908 | 0.8458 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 3.0000 |
| TotalWorkingYears | 1,176.00 | 11.3648 | 7.8014 | 0.0000 | 6.0000 | 10.0000 | 15.0000 | 40.0000 |
| TrainingTimesLastYear | 1,176.00 | 2.7602 | 1.2563 | 0.0000 | 2.0000 | 3.0000 | 3.0000 | 6.0000 |
| WorkLifeBalance | 1,176.00 | 2.7577 | 0.7181 | 1.0000 | 2.0000 | 3.0000 | 3.0000 | 4.0000 |
| YearsAtCompany | 1,176.00 | 7.0502 | 6.0866 | 0.0000 | 3.0000 | 5.0000 | 10.0000 | 37.0000 |
| YearsInCurrentRole | 1,176.00 | 4.2313 | 3.5695 | 0.0000 | 2.0000 | 3.0000 | 7.0000 | 17.0000 |
| YearsSinceLastPromotion | 1,176.00 | 2.1828 | 3.2153 | 0.0000 | 0.0000 | 1.0000 | 3.0000 | 15.0000 |
| YearsWithCurrManager | 1,176.00 | 4.1964 | 3.5648 | 0.0000 | 2.0000 | 3.0000 | 7.0000 | 17.0000 |
| Attrition | 1,176.00 | 0.1616 | 0.3682 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |


Berdasarkan hasil statistik deskriptif yang diperoleh, dataset ini merepresentasikan profil 1.176 karyawan dengan karakteristik yang beragam. Rata-rata usia karyawan adalah 37 tahun dengan masa kerja di perusahaan sekitar 7 tahun. Sebagian besar karyawan memiliki pengalaman kerja total 11 tahun dan telah menempati posisi saat ini selama 4 tahun. Dari aspek kompensasi, gaji bulanan rata-rata sebesar $6,544 dengan variasi yang cukup signifikan antar level jabatan.

Tingkat kepuasan karyawan secara umum berada pada level sedang hingga baik, dengan nilai rata-rata sekitar 2.7 dari skala 4 untuk berbagai aspek kepuasan kerja. Meskipun sebagian besar karyaman menunjukkan keterlibatan kerja yang baik, terdapat sekitar 16% karyawan yang mengalami attrition atau keluar dari perusahaan. Pola kerja menunjukkan bahwa karyawan rata-rata mengikuti 2-3 kali pelatihan dalam setahun terakhir dan memiliki kenaikan gaji tahunan sebesar 15%. Temuan ini mengindikasikan adanya variasi dalam pengalaman dan persepsi karyawan yang dapat dijadikan dasar untuk menganalisis faktor-faktor yang mempengaruhi turnover dalam perusahaan.


## Rata-rata Fitur per Kategori Attrition

|                          | 0         | 1         |
|:-------------------------|:----------|:----------|
| Age                      | 37.74     | 33.13     |
| DailyRate                | 816.56    | 738.74    |
| DistanceFromHome         | 9.05      | 10.97     |
| Education                | 2.92      | 2.84      |
| EmployeeCount            | 1.00      | 1.00      |
| EmployeeNumber           | 1,009.80  | 1,047.15  |
| EnvironmentSatisfaction  | 2.77      | 2.44      |
| HourlyRate               | 65.67     | 64.60     |
| JobInvolvement           | 2.78      | 2.54      |
| JobLevel                 | 2.16      | 1.66      |
| JobSatisfaction          | 2.77      | 2.46      |
| MonthlyIncome            | 6,871.64  | 4,843.88  |
| MonthlyRate              | 14,321.48 | 14,747.08 |
| NumCompaniesWorked       | 2.64      | 2.97      |
| PercentSalaryHike        | 15.25     | 15.16     |
| PerformanceRating        | 3.16      | 3.16      |
| RelationshipSatisfaction | 2.76      | 2.63      |
| StandardHours            | 80.00     | 80.00     |
| StockOptionLevel         | 0.84      | 0.52      |
| TotalWorkingYears        | 12.00     | 8.06      |
| TrainingTimesLastYear    | 2.78      | 2.65      |
| WorkLifeBalance          | 2.78      | 2.63      |
| YearsAtCompany           | 7.46      | 4.90      |
| YearsInCurrentRole       | 4.49      | 2.87      |
| YearsSinceLastPromotion  | 2.24      | 1.87      |
| YearsWithCurrManager     | 4.46      | 2.83      |
| Attrition                | 0.00      | 1.00      |


Berdasarkan hasil analisis rata-rata fitur per kategori attrition, terlihat pola yang jelas antara karyawan yang bertahan (0) dan yang keluar (1). Karyawan yang mengalami attrition cenderung lebih muda dengan rata-rata usia 33 tahun dibandingkan yang bertahan (38 tahun). Mereka juga memiliki tingkat kepuasan kerja yang lebih rendah across semua aspek - lingkungan kerja, pekerjaan, dan hubungan kerja. Dari sisi karir, karyawan yang keluar memiliki level jabatan yang lebih rendah, penghasilan bulanan yang lebih kecil ($4,844 vs $6,872), dan masa kerja yang lebih singkat di perusahaan (4.9 tahun vs 7.5 tahun).

Pola menarik terlihat dalam mobilitas kerja dimana karyawan yang attrition justru memiliki jarak tempuh ke kantor yang lebih jauh (11 km vs 9 km) dan pernah bekerja di lebih banyak perusahaan sebelumnya (2.97 vs 2.64). Meskipun mendapat kenaikan gaji yang hampir sama, mereka memiliki tingkat kepemilikan saham perusahaan yang lebih rendah dan lebih jarang mengikuti pelatihan. Temuan ini mengindikasikan bahwa karyawan yang keluar cenderung berada pada fase karir awal, kurang terikat secara finansial dengan perusahaan, dan mengalami tingkat engagement yang lebih rendah dalam berbagai aspek pekerjaan.

### Missing Value & Outliers
<a id="missing-value--outliers"></a>
Dalam tahap awal pembersihan data, dilakukan pengecekan terhadap duplikasi data dan missing value. Hasilnya menunjukkan bahwa tidak terdapat duplikasi data maupun missing value di seluruh kolom fitur maupun target. Hal ini mengindikasikan bahwa dataset sudah lengkap dan tidak memerlukan teknik imputasi lebih lanjut.
## Analisis Missing Value

| Fitur | Missing Value | Persentase Missing (%) |
|-------|---------------|------------------------|
| id | 0 | 0.0% |
| Age | 0 | 0.0% |
| NumCompaniesWorked | 0 | 0.0% |
| Over18 | 0 | 0.0% |
| OverTime | 0 | 0.0% |
| PercentSalaryHike | 0 | 0.0% |
| PerformanceRating | 0 | 0.0% |
| RelationshipSatisfaction | 0 | 0.0% |
| StandardHours | 0 | 0.0% |
| StockOptionLevel | 0 | 0.0% |
| TotalWorkingYears | 0 | 0.0% |
| TrainingTimesLastYear | 0 | 0.0% |
| WorkLifeBalance | 0 | 0.0% |
| YearsAtCompany | 0 | 0.0% |
| YearsInCurrentRole | 0 | 0.0% |
| YearsSinceLastPromotion | 0 | 0.0% |
| YearsWithCurrManager | 0 | 0.0% |
| MonthlyRate | 0 | 0.0% |
| MonthlyIncome | 0 | 0.0% |
| MaritalStatus | 0 | 0.0% |
| EmployeeCount | 0 | 0.0% |
| BusinessTravel | 0 | 0.0% |
| DailyRate | 0 | 0.0% |
| Department | 0 | 0.0% |
| DistanceFromHome | 0 | 0.0% |
| Education | 0 | 0.0% |
| EducationField | 0 | 0.0% |
| EmployeeNumber | 0 | 0.0% |
| JobSatisfaction | 0 | 0.0% |
| EnvironmentSatisfaction | 0 | 0.0% |
| Gender | 0 | 0.0% |
| HourlyRate | 0 | 0.0% |
| JobInvolvement | 0 | 0.0% |
| JobLevel | 0 | 0.0% |
| JobRole | 0 | 0.0% |
| Attrition | 0 | 0.0% |


Selanjutnya, dilakukan deteksi outlier menggunakan metode Interquartile Range (IQR) untuk setiap fitur numerik. 
## Deteksi Outlier dengan Metode IQR

| Fitur                    |   Jumlah Outlier |   Persentase Outlier (%) |   Lower Bound |   Upper Bound |   Min |   Max |
|:-------------------------|-----------------:|-------------------------:|--------------:|--------------:|------:|------:|
| Attrition                |              190 |                    16.16 |          0    |          0    |     0 |     1 |
| PerformanceRating        |              185 |                    15.73 |          3    |          3    |     3 |     4 |
| TrainingTimesLastYear    |              174 |                    14.8  |          0.5  |          4.5  |     0 |     6 |
| MonthlyIncome            |               86 |                     7.31 |      -5260.75 |      16629.2  |  1009 | 19973 |
| YearsSinceLastPromotion  |               85 |                     7.23 |         -4.5  |          7.5  |     0 |    15 |
| StockOptionLevel         |               66 |                     5.61 |         -1.5  |          2.5  |     0 |     3 |
| YearsAtCompany           |               52 |                     4.42 |         -7.5  |         20.5  |     0 |    37 |
| TotalWorkingYears        |               52 |                     4.42 |         -7.5  |         28.5  |     0 |    40 |
| NumCompaniesWorked       |               36 |                     3.06 |         -3.5  |          8.5  |     0 |     9 |
| YearsInCurrentRole       |               16 |                     1.36 |         -5.5  |         14.5  |     0 |    17 |
| YearsWithCurrManager     |               10 |                     0.85 |         -5.5  |         14.5  |     0 |    17 |
| JobLevel                 |                0 |                     0    |         -2    |          6    |     1 |     5 |
| JobSatisfaction          |                0 |                     0    |         -1    |          7    |     1 |     4 |
| DistanceFromHome         |                0 |                     0    |        -16    |         32    |     1 |    29 |
| Education                |                0 |                     0    |         -1    |          7    |     1 |     5 |
| WorkLifeBalance          |                0 |                     0    |          0.5  |          4.5  |     1 |     4 |
| EmployeeCount            |                0 |                     0    |          1    |          1    |     1 |     1 |
| EmployeeNumber           |                0 |                     0    |      -1101.5  |       3136.5  |     1 |  2062 |
| EnvironmentSatisfaction  |                0 |                     0    |         -1    |          7    |     1 |     4 |
| StandardHours            |                0 |                     0    |         80    |         80    |    80 |    80 |
| RelationshipSatisfaction |                0 |                     0    |         -1    |          7    |     1 |     4 |
| HourlyRate               |                0 |                     0    |         -4.5  |        135.5  |    30 |   100 |
| PercentSalaryHike        |                0 |                     0    |          3    |         27    |    11 |    25 |
| DailyRate                |                0 |                     0    |       -566.12 |       2190.88 |   103 |  1499 |
| MonthlyRate              |                0 |                     0    |     -11028.6  |      39850.4  |  2094 | 26999 |
| JobInvolvement           |                0 |                     0    |          0.5  |          4.5  |     1 |     4 |
| Age                      |                0 |                     0    |         10.5  |         62.5  |    18 |    60 |


Berdasarkan hasil analisis outlier dengan metode IQR, teridentifikasi beberapa pola menarik dalam distribusi data. Variabel **Attrition** dan **PerformanceRating** menunjukkan jumlah outlier tertinggi dengan masing-masing 190 (16.16%) dan 185 (15.73%) observasi, yang dapat dijelaskan oleh sifat distribusi data yang tidak normal pada variabel kategori ini. 

Beberapa variabel finansial dan karir juga menunjukkan keberadaan outlier yang signifikan, seperti **MonthlyIncome** dengan 86 outlier (7.31%), **YearsSinceLastPromotion** (85 outlier, 7.23%), dan **StockOptionLevel** (66 outlier, 5.61%). Hal ini mengindikasikan adanya variasi ekstrem dalam kompensasi dan perkembangan karir karyawan, dimana sebagian kecil karyawan memiliki pendapatan yang sangat tinggi atau masa tunggu promosi yang sangat lama dibandingkan dengan mayoritas populasi.

Variabel terkait pengalaman kerja seperti **YearsAtCompany** dan **TotalWorkingYears** masing-masing memiliki 52 outlier (4.42%), mencerminkan adanya karyawan dengan masa kerja yang sangat panjang di perusahaan atau total pengalaman kerja yang jauh melampaui rata-rata. Sementara itu, **NumCompaniesWorked** dengan 36 outlier (3.06%) menunjukkan variasi dalam mobilitas karir antar karyawan.

Menariknya, sebagian besar variabel kepuasan kerja (**JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction**) serta variabel demografis seperti **Age** dan **Education** tidak mengandung outlier sama sekali, mengindikasikan distribusi yang relatif stabil dan terpusat untuk karakteristik-karakteristik tersebut dalam populasi karyawan.
<img width="1990" height="3065" alt="image" src="https://github.com/user-attachments/assets/989838d7-4f78-4c15-906e-dc4f67a7560d" />
Visualisasi melalui boxplot memperjelas sebaran data dan keberadaan outlier pada hampir seluruh variabel numerik. Sebagian besar variabel seperti MonthlyIncome, TotalWorkingYears, YearsAtCompany, dan YearsInCurrentRole menunjukkan adanya outlier di bagian atas, menandakan terdapat individu dengan nilai yang jauh lebih tinggi dibandingkan mayoritas karyawan. Hal ini mengindikasikan adanya variasi ekstrem, misalnya perbedaan besar dalam masa kerja atau penghasilan.

Sebaliknya, beberapa variabel seperti Education, JobInvolvement, WorkLifeBalance, dan PerformanceRating memiliki sebaran yang relatif sempit tanpa outlier mencolok, menandakan distribusi data yang cenderung homogen di antara karyawan.

Meskipun ditemukan sejumlah outlier, data tersebut tidak dihapus dari dataset untuk menjaga keutuhan informasi. Nilai-nilai ekstrem ini mungkin mencerminkan kondisi nyata seperti karyawan senior dengan pengalaman panjang atau jabatan tinggi yang wajar memiliki pendapatan besar. Menghapus outlier justru dapat menghilangkan variasi penting yang relevan dengan analisis terkait kinerja atau loyalitas karyawan. Sebagai langkah mitigasi terhadap pengaruh outlier, analisis lanjutan dapat menggunakan metode atau model yang lebih robust terhadap nilai ekstrem.


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
