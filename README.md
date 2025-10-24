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

Tingkat kepuasan karyawan secara umum berada pada level sedang hingga baik, dengan nilai rata-rata sekitar 2.7 dari skala 4 untuk berbagai aspek kepuasan kerja. Meskipun sebagian besar karyawan menunjukkan keterlibatan kerja yang baik, terdapat sekitar 16% karyawan yang mengalami attrition atau keluar dari perusahaan. Pola kerja menunjukkan bahwa karyawan rata-rata mengikuti 2-3 kali pelatihan dalam setahun terakhir dan memiliki kenaikan gaji tahunan sebesar 15%. Temuan ini mengindikasikan adanya variasi dalam pengalaman dan persepsi karyawan yang dapat dijadikan dasar untuk menganalisis faktor-faktor yang mempengaruhi turnover dalam perusahaan.


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


Berdasarkan hasil analisis outlier dengan metode IQR, teridentifikasi beberapa pola menarik dalam distribusi data. Variabel **PerformanceRating** menunjukkan jumlah outlier tertinggi dengan  185 (15.73%) observasi, yang dapat dijelaskan oleh sifat distribusi data yang tidak normal pada variabel kategori ini. 

Beberapa variabel finansial dan karir juga menunjukkan keberadaan outlier yang signifikan, seperti **MonthlyIncome** dengan 86 outlier (7.31%), **YearsSinceLastPromotion** (85 outlier, 7.23%), dan **StockOptionLevel** (66 outlier, 5.61%). Hal ini mengindikasikan adanya variasi ekstrem dalam kompensasi dan perkembangan karir karyawan, dimana sebagian kecil karyawan memiliki pendapatan yang sangat tinggi atau masa tunggu promosi yang sangat lama dibandingkan dengan mayoritas populasi.

Variabel terkait pengalaman kerja seperti **YearsAtCompany** dan **TotalWorkingYears** masing-masing memiliki 52 outlier (4.42%), mencerminkan adanya karyawan dengan masa kerja yang sangat panjang di perusahaan atau total pengalaman kerja yang jauh melampaui rata-rata. Sementara itu, **NumCompaniesWorked** dengan 36 outlier (3.06%) menunjukkan variasi dalam mobilitas karir antar karyawan.

Menariknya, sebagian besar variabel kepuasan kerja (**JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction**) serta variabel demografis seperti **Age** dan **Education** tidak mengandung outlier sama sekali, mengindikasikan distribusi yang relatif stabil dan terpusat untuk karakteristik-karakteristik tersebut dalam populasi karyawan.
<img width="1990" height="3065" alt="image" src="https://github.com/user-attachments/assets/989838d7-4f78-4c15-906e-dc4f67a7560d" />
Visualisasi melalui boxplot memperjelas sebaran data dan keberadaan outlier pada hampir seluruh variabel numerik. Sebagian besar variabel seperti MonthlyIncome, TotalWorkingYears, YearsAtCompany, dan YearsInCurrentRole menunjukkan adanya outlier di bagian atas, menandakan terdapat individu dengan nilai yang jauh lebih tinggi dibandingkan mayoritas karyawan. Hal ini mengindikasikan adanya variasi ekstrem, misalnya perbedaan besar dalam masa kerja atau penghasilan.

Sebaliknya, beberapa variabel seperti Education, JobInvolvement, WorkLifeBalance, dan PerformanceRating memiliki sebaran yang relatif sempit tanpa outlier mencolok, menandakan distribusi data yang cenderung homogen di antara karyawan.

Meskipun ditemukan sejumlah outlier, data tersebut tidak dihapus dari dataset untuk menjaga keutuhan informasi. Nilai-nilai ekstrem ini mungkin mencerminkan kondisi nyata seperti karyawan senior dengan pengalaman panjang atau jabatan tinggi yang wajar memiliki pendapatan besar. Menghapus outlier justru dapat menghilangkan variasi penting yang relevan dengan analisis terkait kinerja atau loyalitas karyawan. Sebagai langkah mitigasi terhadap pengaruh outlier, analisis lanjutan dapat menggunakan metode atau model yang lebih robust terhadap nilai ekstrem.


### Univariate Analysis
<a id="univariate-analysis"></a>
Grafik 1 : Distribusi kategori Churn Karyawan
<img width="794" height="702" alt="image" src="https://github.com/user-attachments/assets/48fd42ce-fdff-4d22-810a-7d904019509c" />
Berdasarkan distribusi kategori attrition pada data training, teridentifikasi bahwa **16.2%** karyawan termasuk dalam kategori churn (keluar dari perusahaan), sementara **83.8%** karyawan bertahan. Proporsi ini mengindikasikan bahwa sebagian besar karyawan dalam dataset tetap berada di perusahaan, dengan hanya sekitar 1 dari 6 karyawan yang memutuskan untuk keluar. Tingkat attrition sebesar 16.2% ini memberikan baseline penting untuk pengembangan model prediktif dalam mengidentifikasi faktor-faktor yang memengaruhi keputusan karyawan untuk meninggalkan perusahaan.
Grafik 2 : Distribusi Fitur Numerik
<img width="1988" height="3065" alt="image" src="https://github.com/user-attachments/assets/4901b65f-1d03-47a7-816a-de1f9f04cbd5" />
Visualisasi distribusi seluruh variabel numerik memberikan gambaran menyeluruh mengenai bentuk sebaran data, kecenderungan pusat, serta potensi skewness (kemencengan) dari masing-masing variabel. Secara umum, sebagian besar variabel menunjukkan distribusi yang tidak normal (non-simetris) dengan kecenderungan right-skewed (miring ke kanan), terutama pada variabel yang berhubungan dengan masa kerja dan pendapatan.

Beberapa temuan penting dari distribusi tersebut adalah sebagai berikut:

Variabel dengan Distribusi Relatif Simetris
Beberapa variabel seperti Age, DailyRate, HourlyRate, dan MonthlyRate menunjukkan distribusi yang relatif menyebar merata di sekitar nilai rata-rata. Hal ini menandakan tidak adanya dominasi nilai ekstrem tertentu, serta menggambarkan keragaman yang cukup seimbang antar individu dalam hal usia dan kompensasi harian/jam.

Variabel yang Miring ke Kanan (Right-Skewed)
Variabel seperti DistanceFromHome, MonthlyIncome, TotalWorkingYears, YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, dan YearsWithCurrManager memiliki ekor panjang di sisi kanan. Ini menunjukkan sebagian besar karyawan memiliki nilai rendah–sedang (misalnya masa kerja atau pendapatan yang lebih pendek/rendah), sementara hanya sedikit yang memiliki nilai tinggi (masa kerja atau pendapatan jauh di atas rata-rata).
Pola ini umum dijumpai dalam konteks ketenagakerjaan, di mana hanya sebagian kecil karyawan yang berpengalaman lama atau memiliki jabatan tinggi.

Variabel Kategorikal Bertipe Skala Ordinal (Diskret 1–4 atau 1–5)
Variabel seperti Education, EnvironmentSatisfaction, JobInvolvement, JobSatisfaction, RelationshipSatisfaction, WorkLifeBalance, dan PerformanceRating cenderung menunjukkan distribusi dengan puncak-puncak tertentu pada nilai diskritnya. Hal ini menunjukkan bahwa sebagian besar responden memberi penilaian pada tingkat menengah (skor 2 atau 3), dengan sedikit responden di ekstrem (skor 1 atau 4). Distribusi seperti ini menggambarkan penilaian karyawan yang cenderung moderat terhadap aspek-aspek pekerjaan.

Variabel dengan Distribusi Acak atau Merata
Variabel seperti EmployeeNumber dan MonthlyRate menunjukkan pola yang tampak acak atau merata tanpa kecenderungan bentuk distribusi tertentu, yang dapat disebabkan karena variabel tersebut bersifat identifikasi unik atau bergantung pada sistem penggajian yang bervariasi.

Grafik 3: Distribusi Fitur Kategorik
<img width="1980" height="1535" alt="image" src="https://github.com/user-attachments/assets/6e9346fd-d0c9-4e97-a21e-e3b8c9635ba4" />
Visualisasi distribusi variabel kategorik memberikan gambaran mengenai proporsi dan dominasi tiap kategori dalam dataset karyawan. Secara umum, sebagian besar variabel menunjukkan ketimpangan distribusi antar kategori, yang mencerminkan karakteristik umum tenaga kerja dalam perusahaan.

1. BusinessTravel

Mayoritas karyawan (70,4%) hanya melakukan perjalanan bisnis Travel_Rarely, sementara hanya sebagian kecil yang Travel_Frequently (19,1%) atau Non-Travel (10,5%). Hal ini menunjukkan bahwa sebagian besar pekerjaan tidak membutuhkan perjalanan dinas intensif, kemungkinan besar pekerjaan yang bersifat administratif atau berbasis kantor.

2. Department

Distribusi menunjukkan dominasi departemen Research & Development (65,0%), diikuti oleh Sales (31,0%), dan sisanya Human Resources (4,1%). Artinya, perusahaan ini memiliki fokus besar pada penelitian dan pengembangan produk, sementara bagian penjualan dan HR relatif lebih kecil.

3. EducationField

Sebagian besar karyawan berasal dari bidang Life Sciences (40,7%) dan Medical (31,0%), sedangkan bidang lain seperti Marketing, Technical Degree, dan Human Resources hanya memiliki proporsi kecil. Ini menggambarkan bahwa perusahaan bergerak di bidang industri yang berkaitan dengan sains dan kesehatan.

4. Gender

Komposisi gender relatif seimbang dengan kecenderungan dominan laki-laki (59,9%) dibanding perempuan (40,1%). Perbedaan ini bisa disebabkan oleh jenis pekerjaan yang mungkin lebih banyak membutuhkan tenaga kerja laki-laki, terutama pada posisi teknis atau lapangan.

5. JobRole

Peran pekerjaan paling banyak adalah Sales Executive (22,7%) dan Research Scientist (18,7%), diikuti Laboratory Technician (18,3%). Jabatan manajerial seperti Manager, Director, dan Healthcare Representative memiliki proporsi kecil di bawah 10%. Hal ini menunjukkan bahwa mayoritas tenaga kerja berada pada level pelaksana atau teknis, bukan manajemen puncak.

6. MaritalStatus

Karyawan yang menikah (45,3%) mendominasi, diikuti oleh single (31,7%) dan divorced (22,4%). Distribusi ini dapat mencerminkan bahwa sebagian besar tenaga kerja berada pada rentang usia produktif dengan stabilitas keluarga yang relatif tinggi.

7. OverTime

Sebagian besar karyawan tidak lembur (71,1%), sementara 28,9% sisanya sering bekerja lembur. Hal ini menunjukkan bahwa beban kerja tambahan tidak merata di seluruh karyawan, kemungkinan besar hanya terjadi pada divisi tertentu dengan target waktu yang ketat seperti produksi atau pengembangan.



### Multivariate Analysis
<a id="multivariate-analysis"></a>
Grafik 1 :
<img width="1547" height="1389" alt="image" src="https://github.com/user-attachments/assets/eb0bf3f3-46ad-4b2b-a776-be0d518cdd52" />
Korelasi Positif yang Tinggi

MonthlyIncome – JobLevel (r = 0.95):
Terdapat korelasi sangat kuat dan positif antara level jabatan dan pendapatan bulanan. Hal ini menunjukkan bahwa semakin tinggi jabatan seseorang, semakin besar pula pendapatan yang diterima. Korelasi ini wajar karena sistem penggajian biasanya mengikuti tingkat tanggung jawab jabatan.

TotalWorkingYears – MonthlyIncome (r = 0.79):
Hubungan positif yang kuat antara total masa kerja dan pendapatan menunjukkan bahwa karyawan dengan pengalaman kerja lebih lama cenderung memiliki gaji lebih tinggi. Ini menggambarkan adanya kompensasi yang meningkat seiring bertambahnya pengalaman.

TotalWorkingYears – JobLevel (r = 0.77):
Korelasi ini mengindikasikan bahwa semakin lama seseorang bekerja, semakin besar peluangnya untuk mencapai posisi jabatan yang lebih tinggi. Hubungan ini sejalan dengan sistem promosi berdasarkan senioritas dan pengalaman kerja.

YearsWithCurrManager – YearsAtCompany (r = 0.78):
Korelasi yang tinggi antara lama bekerja dengan manajer saat ini dan lama bekerja di perusahaan menunjukkan stabilitas struktur manajemen. Artinya, karyawan yang sudah lama di perusahaan umumnya bekerja lama pula dengan manajer yang sama.

YearsWithCurrManager – YearsInCurrentRole (r = 0.71):
Hubungan positif yang kuat ini menunjukkan bahwa karyawan yang lama menjabat di posisi tertentu juga cenderung memiliki hubungan kerja jangka panjang dengan atasan langsungnya.

YearsAtCompany – TotalWorkingYears (r = 0.69):
Korelasi ini menunjukkan bahwa masa kerja di perusahaan saat ini menyumbang sebagian besar terhadap total pengalaman kerja karyawan. Dengan kata lain, banyak karyawan yang menghabiskan sebagian besar kariernya di perusahaan yang sama.

PerformanceRating – PercentSalaryHike (r = 0.77):
Korelasi positif yang kuat antara penilaian kinerja dan kenaikan gaji menandakan sistem penghargaan yang baik. Karyawan dengan performa tinggi mendapatkan peningkatan gaji yang lebih besar, menunjukkan adanya konsistensi dalam sistem kompensasi berbasis kinerja.

Korelasi Negatif Sedang

Age – Attrition (r = -0.16):
Korelasi negatif sedang ini menunjukkan bahwa semakin bertambah usia karyawan, semakin kecil kemungkinan mereka keluar dari perusahaan. Sebaliknya, karyawan yang lebih muda cenderung lebih sering berpindah pekerjaan.

Grafik 2 :
<img width="2167" height="634" alt="image" src="https://github.com/user-attachments/assets/f356f934-0306-4821-99f6-3a33f107d747" />
1. Job Level vs Monthly Income

Terlihat adanya hubungan positif yang kuat antara Job Level dan Monthly Income.

Semakin tinggi Job Level, semakin besar pula Monthly Income yang diterima karyawan.

Ukuran bubble yang mewakili Total Working Years juga meningkat seiring naiknya Job Level, menunjukkan bahwa masa kerja lebih lama cenderung berbanding lurus dengan posisi dan gaji.

Warna bubble menunjukkan Attrition: karyawan dengan gaji dan level rendah memiliki tingkat Attrition lebih tinggi.

Hal ini menunjukkan bahwa karyawan di posisi bawah lebih rentan keluar dari perusahaan dibandingkan mereka yang berposisi tinggi.

2. Years at Company vs Job Level

Terlihat pola bahwa semakin lama seseorang bekerja di perusahaan, semakin tinggi pula jabatan yang diperoleh.

Bubble besar (yang menunjukkan Monthly Income tinggi) terkonsentrasi pada Job Level tinggi.

Warna bubble yang lebih terang menandakan Total Working Years yang juga panjang, sehingga masa kerja panjang → jabatan tinggi → gaji besar.

Pola ini menegaskan bahwa sistem promosi di perusahaan berjalan konsisten berdasarkan pengalaman dan loyalitas kerja.

Karyawan yang telah lama bekerja cenderung menikmati peningkatan pendapatan yang signifikan.

3. Performance Rating vs Percent Salary Hike

Terlihat hubungan positif antara Performance Rating dan Percent Salary Hike.

Karyawan dengan Performance Rating tinggi (rating 4) cenderung memperoleh kenaikan gaji yang lebih besar.

Bubble besar (gaji tinggi) dan warna cerah (jabatan tinggi) didominasi oleh karyawan berperforma baik.

Hal ini menunjukkan bahwa perusahaan menerapkan sistem penghargaan berbasis kinerja, di mana performa tinggi menghasilkan imbalan finansial yang lebih besar.

Dengan demikian, motivasi dan produktivitas karyawan dapat terjaga melalui insentif yang adil.

Grafik 3 :
Distribusi Fitur Kategorik dengan Attrition 
<img width="2014" height="3065" alt="image" src="https://github.com/user-attachments/assets/c7829671-ce50-4124-b24f-484fb315218a" />
1. Variabel Demografis dan Karakteristik Umum

Age: Karyawan yang keluar (Churn) cenderung berusia lebih muda (rata-rata 33 tahun) dibandingkan yang bertahan (rata-rata 37 tahun). Ini menunjukkan bahwa usia muda lebih rentan mengalami turnover, kemungkinan karena mereka masih mencari pengalaman atau peluang yang lebih baik.

DistanceFromHome: Nilai rata-rata jarak rumah lebih tinggi untuk karyawan yang Churn (10,97) dibanding Non-Churn (9,09). Hal ini menunjukkan bahwa jarak tempat tinggal yang jauh dapat meningkatkan risiko keluar dari perusahaan.

Education: Tidak terdapat perbedaan mencolok antara Churn dan Non-Churn, dengan rata-rata sekitar 3. Artinya, tingkat pendidikan tidak berpengaruh signifikan terhadap keputusan keluar.

2. Variabel Kepuasan dan Lingkungan Kerja

EnvironmentSatisfaction, JobSatisfaction, dan WorkLifeBalance: Ketiga variabel ini menunjukkan rata-rata lebih rendah pada karyawan yang Churn. Artinya, ketidakpuasan terhadap lingkungan kerja, pekerjaan, dan keseimbangan hidup–kerja berperan besar dalam keputusan resign.

RelationshipSatisfaction: Nilai rata-rata Churn juga sedikit lebih rendah, mengindikasikan bahwa hubungan sosial dan interaksi dengan rekan kerja atau atasan memengaruhi loyalitas.

TrainingTimesLastYear: Karyawan Non-Churn sedikit lebih sering mendapat pelatihan (rata-rata 2,78) dibandingkan Churn (2,56), yang berarti program pengembangan karyawan dapat membantu menekan tingkat turnover.

3. Variabel Karier dan Pengalaman Kerja

JobLevel dan TotalWorkingYears: Karyawan dengan Job Level dan pengalaman kerja lebih rendah memiliki kecenderungan lebih besar untuk keluar. Ini menunjukkan bahwa karyawan baru atau berposisi rendah lebih rentan terhadap turnover, mungkin karena belum merasa stabil secara finansial atau karier.

YearsAtCompany dan YearsInCurrentRole: Karyawan yang bertahan memiliki masa kerja dan masa jabatan lebih lama. Hal ini menunjukkan bahwa semakin lama seseorang bekerja di posisi tertentu, semakin besar kemungkinan mereka untuk tetap loyal.

YearsSinceLastPromotion: Rata-rata Churn memiliki masa yang lebih pendek sejak promosi terakhir, menunjukkan bahwa kurangnya kesempatan promosi dapat mendorong karyawan keluar.

 4. Variabel Kompensasi dan Kinerja

MonthlyIncome dan HourlyRate: Nilai rata-rata gaji lebih rendah pada kelompok Churn, menunjukkan bahwa kompensasi yang rendah berhubungan erat dengan tingkat keluar yang tinggi.

DailyRate dan MonthlyRate: Pola distribusi juga memperkuat bahwa karyawan dengan penghasilan lebih kecil lebih rentan keluar dari perusahaan.

PercentSalaryHike dan PerformanceRating: Karyawan Non-Churn cenderung memiliki Performance Rating dan Salary Hike yang sedikit lebih tinggi, mengindikasikan bahwa sistem penghargaan dan evaluasi kinerja yang adil dapat menekan turnover.
Grafik 4 :
Distribusi fitur Kategorik dengan Attrition
<img width="1782" height="1534" alt="image" src="https://github.com/user-attachments/assets/6e75f941-b585-4b78-b729-52fda3da9169" />
1. Business Travel

Karyawan yang sering melakukan perjalanan dinas (Travel Frequently) memiliki tingkat attrition tertinggi, yaitu 25,3%.

Sebaliknya, karyawan yang tidak pernah melakukan perjalanan (Non-Travel) hanya 15% yang keluar.

Hal ini menunjukkan bahwa beban perjalanan kerja yang tinggi berpotensi meningkatkan kelelahan dan stres, sehingga memperbesar kemungkinan resign.

2. Department

Tingkat attrition cukup bervariasi antar departemen:

Sales dan Research & Development (R&D) memiliki tingkat keluar yang sama, yaitu 20,6%.

Human Resources sedikit lebih rendah, yaitu 13,6%.

Hal ini menandakan bahwa departemen dengan beban kerja tinggi atau tekanan target (Sales dan R&D) memiliki tingkat turnover yang lebih besar.

3. Education Field

Bidang pendidikan Human Resources dan Life Sciences menunjukkan tingkat keluar tertinggi (masing-masing 15,7% dan 13,4%).

Sedangkan bidang Marketing, Medical, Technical Degree, dan Other memiliki tingkat keluar yang lebih rendah.

Ini menunjukkan bahwa latar belakang pendidikan tidak terlalu berpengaruh signifikan, namun mungkin terkait dengan jenis pekerjaan dan ekspektasi terhadap karier.

4. Gender

Tingkat keluar antara perempuan (16,2%) dan laki-laki (15,9%) hampir seimbang.

Artinya, gender bukan faktor dominan dalam keputusan karyawan untuk resign, sehingga penyebab attrition lebih terkait pada faktor pekerjaan atau organisasi daripada jenis kelamin.

5. Job Role

Tingkat keluar tertinggi terjadi pada:

Sales Representative (39,7%)

Laboratory Technician (27,0%)

Human Resources (26,7%)

Sebaliknya, posisi seperti Manager, Research Director, dan Healthcare Representative memiliki tingkat keluar rendah.

Ini menunjukkan bahwa posisi dengan tanggung jawab tinggi dan gaji besar cenderung lebih stabil, sementara posisi teknis dan lapangan memiliki turnover lebih tinggi karena tekanan atau kompensasi yang lebih rendah.

6. Marital Status

Karyawan berstatus menikah (Married) memiliki tingkat keluar tertinggi (25,5%), diikuti oleh single (11,9%) dan divorced (12,2%).

Hal ini menunjukkan bahwa karyawan menikah mungkin menghadapi tekanan waktu atau tanggung jawab keluarga, sehingga lebih rentan keluar jika keseimbangan kerja-hidup tidak terpenuhi.

7. OverTime

Perbedaan mencolok terlihat pada variabel ini:

Karyawan yang sering lembur (Yes) memiliki tingkat keluar 28,6%.

Sedangkan yang tidak lembur (No) hanya 11,1%.

Ini menegaskan bahwa lembur berlebihan menjadi salah satu faktor kuat yang memicu burnout dan meningkatkan turnover.




### Kesimpulan EDA
<a id="kesimpulan-eda"></a>
Isi kesimpulan EDA di sini...


## Data Preparation
<a id="data-preparation"></a>

### Label Encoding dengan Mapping pada Fitur Target
<a id="label-encoding-dengan-mapping-pada-fitur-target"></a>
Fitur target **Attrition** sudah berupa label (0 dan 1), sehingga tidak perlu dilakukan encoding. Adapun mapping digunakan sebagai berikut : 
| Kategori Attrition | Label |
|--------------------|--------|
| No | 0 |
| Yes | 1 |

### Splitting Dataset
<a id="splitting-dataset"></a>
- Menetapkan ```stratify = y``` sehingga fungsi train_test_split memastikan bahwa proses pemisahan mempertahankan persentase yang sama dari setiap kelas target di set train dan test.
  
Dataset yang digunakan dalam analisis ini terdiri dari data pelatihan (train) dan data pengujian (test) dengan rincian sebagai berikut:
- **Ukuran data fitur (train)** : 940 observasi dengan 31 fitur.
- **Ukuran data target (train)** : 940 observasi
- **Ukuran data fitur (test)** : 236 observasi dengan 31 fitur.
- **Ukuran data target (test)** : 236 observasi.

**Proporsi Kelas pada Variabel Target**
Distribusi proporsi kelas pada variabel target ```Attrition``` untuk masing-masing data adalah sebagai berikut:
- **Data Pelatihan (Train)** :
  - Kelas 0 (tidak keluar perusahaan) : 83,83%
  - Kelas 1 (keluar perusahaan) : 16,17%
- **Data Pengujian (Test)** :
  - Kelas 0 (tidak keluar perusahaan) : 83,55%
  - Kelas 1 (keluar perusahaan) : 16,45%

Distribusi kelas yang relatif seimbang antara data pelatihan dan pengujian menunjukkan bahwa proses pembagian data telah mempertahankan proporsi kelas, sehingga model dapat dilatih dan dievaluasi secara konsisten terhadap fenomena Attrition.

### Feature Engineering, Data Cleaning and Preprocessing
<a id="feature-engineering-data-cleaning-and-preprocessing"></a>
Prepocessing untuk Model Berbasis Tree
- **Fitur Numerik** :
  Tidak akan dilakukan transformasi apa pun karena model berbasis tree tidak memerlukan feature   scaling.
- **Fitur Kategorikal** (Ordinal => BusinessTravel)
  Akan diterapkan ordinal encoding untuk mempertahankan karakteristik ordinal.
- **Fitur Kategorikal** (Nominal => Department, EducationField, JobRole, MaritalStatus)
  Akan diterapkan one hot encoding karena jumlah kategori sedikit di tiap fitur.
- **Fitur Kategorikal** (Binary => Gender, OverTime)
  Akan diterapkan label encoding karena  fitur ini akan diubah menjadi variabel biner unik,       sehingga tidak meningkatkan dimensi.

**Feature Engineering**
Untuk mendapatkan informasi maksimal dari fitur yang tersedia, dilakukan feature engineering yang sudah terintegrasi dalam preprocessing dengan membuat fitur-fitur berikut:

Fitur Rasio :
```
1. `Income_Per_Year`   = MonthlyIncome / (TotalWorkingYears +1)
2. `Promotion_Rate`    = YearsAtCompany / YearsSinceLastPromotion
3. `Salary_Hike_Ratio` =  PercentSalaryHike / MonthlyIncome
```

Fitur Progresi Karir :
```
1. `Career_Stagnation` = (YearsInCurrentRole > 3) & (YearsSinceLastPromotion > 2)
2. `Fast_Promotion`    = (YearsSinceLastPromotion < 2) & (JobLevel > 1)
```

Fitur Kepuasan :
```
1. `Overall_Satisfaction`         = (EnvironmentSatisfaction + JobSatisfaction + RelationshipSatisfaction) / 3
2. `Low_Satisfaction_High_Income` = (Overall_Satisfaction < 2) & (MonthlyIncome > MonthlyIncome.median())
```

Fitur Kehidupan Kerja :
```
1. `Work_Stress`      = (JobInvolvement > 3) & (WorkLifeBalance < 2)
2. `Overtime_Impact`  = OverTime * JobInvolvement
```

Fitur-fitur di atas dapat menangkap hubungan dan pola tersembunyi, serta relevan dalam konteks sumber daya manusia. Hal ini sangat penting untuk diperhatikan saat melakukan feature engineering.

**Penanganan Data Duplikat pada Kolom ```id```**

Terdapat 3 duplikat dalam kolom id di data train, sehingga perlu didrop agar tidak terjadi inkonsistensi target variabel untuk id yang sama.

**Variabel yang Akan Dihapus**
  1. ```id``` :
     Akan dihapus karena memiliki nilai unik untuk setiap record, sehingga tidak berguna untuk       analisis.
  2. ```EmployeeCount``` :
     Akan dihapus karena nilainya selalu 1 untuk semua karyawan, sehingga tidak berguna untuk        analisis.
  3. ```Over18``` :
     Akan dihapus karena nilainya selalu Y dalam dataset (semua karyawan berusia di atas 18          tahun), sehingga tidak berguna untuk analisis.
  4. ```StandardHours``` :
     Akan dihapus karena nilainya selalu 80 dalam dataset (semua karyawan memiliki jam kerja         standard yang sama), sehingga tidak berguna untuk analisis.

## Model Training, Comparison, Selection and Tuning
<a id="model-training-comparison-selection-and-tuning"></a>

### 1. Model Selection
<a id="model-selection"></a>
Pada tahap pengembangan model, digunakan tiga algoritma klasifikasi berbasis tree yang umum dan efektif, Random Forest (RF) dan Gradient Boost(GB). Pemilihan kedua model tersebut didasarkan pada karakteristik masing-masing serta tujuan untuk membandingkan performa secara empiris.

**Random Forest (RF)**
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)
```
[Random Forest][https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html] adalah metode ensemble yang menggabungkan banyak pohon keputusan untuk meningkatkan akurasi dan stabilitas prediksi. Dengan membangun pohon pada subset data secara acak dan menggabungkan hasilnya, model ini mampu mengurangi overfitting dan bekerja baik pada data dengan banyak fitur.

**Gradient Boosting (GB)**
```python
from sklearn.ensemble import RandomForestClassifier

gb_model = RandomForestClassifier()
gb_model.fit(X_train_scaled, y_train)
```
[Gradient Boosting][https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html] adalah metode ensemble yang membangun pohon keputusan secara berurutan, dimana setiap pohon baru belajar mengoreksi kesalahan pohon sebelumnya. Dengan pendekatan sequential ini, model secara bertahap meningkatkan akurasi dan mampu menangkap pola kompleks dalam data.

Kedua model ini digunakan dengan pengaturan parameter awal sebagai percobaan dasar
- Pada langkah ini, membandingkan kinerja model yang berbeda dengan menggunakan **stratified k-   fold cross validation** untuk melatih masing-masing model dan mengevaluasi skor ROC-AUC.        Stratified k-fold cross validation akan mempertahankan proporsi target pada setiap fold,        menangani target yang tidak seimbang.
- k-fold cross validation adalah teknik yang digunakan dalam machine learning untuk menilai       kinerja model. Teknik ini melibatkan pembagian dataset menjadi K subset, menggunakan K-1        untuk pelatihan dan satu untuk pengujian secara berulang. Hal ini membantu dalam                memperkirakan kemampuan generalisasi model dengan mengurangi risiko overfitting dan             memberikan metrik kinerja yang lebih andal.
- Tujuan tahap ini adalah untuk memilih model terbaik untuk digunakan dalam feature selection,    hyperparameter tuning, dan evaluasi model akhir. Untuk mendapatkan model terbaik ini, akan      dievaluasi skor validasi rata-rata **ROC-AUC** tertinggi dan melihat trade-off bias-varians.

**Tabel Perbandingan Performa Model**
| Metrik | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| ROC AUC (Val) | 0.7559 | 0.7671 |
| Akurasi (Val) | 0.8383 | 0.8686 |
| Recall (Val) | 0.3158 | 0.42105|
| Spesifitas (Val) | 0.8809 | 0.9545 |
| ROC AUV (Train) | 1.0 | 1.0 | 1.0 |
| Akurasi (Train) | 1.0 | 1.0 | 1.0 |
| Recall (Train) | 1.0 | 1.0 | 1.0 |
| Spesifitas (Train) | 1.0| 1.0 | 1.0 |
| Waktu Latih (detik) | 3.0 | 37.1 |

Model **Gradient Boosting** dipilih untuk feature selection, hyperparameter tuning dan evaluasi akhir karena menunjukkan performa terbaik dengan rata-rata skor ROC-AUC validasi tertinggi. Meskipun model mengalami indikasi overfitting (skor ROC-AUC sebesar 1 pada data pelatihan), hasil validasi sudah cukup tinggi (0,76), menunjukkan generalisasi yang cukup bagus. Meskipun potensi peningkatan performa lebih lanjut melalui hyperparameter tuning relatif kecil, langkah tersebut tetap akan dilakukan sebagai bagian dari proses penyempurnaan model.

### 2. Feature Selection
<a id="feature-selection"></a>
Walaupun langkah seleksi fitur sangat penting untuk meningkatkan kemampuan generalisasi model   dan membuatnya lebih sederhana, namun ada kalanya langkah ini malah menurunkan performa model, seperti yang terjadi pada model ini. Hal tersebut kemungkinan terjadi karena beberapa alasan :
- Tree-Based Model seperti Random Forest dan Gradient Boosting otomatis **melakukan seleksi       fitur internal** dengan tidak menghiraukan fitur yang kurang berpengaruh saat split serta       memprioritaskan fitur penting.
- **Ada kemungkinan terbuangnya "weak signal" saat melakukan seleksi fitur**. Beberapa fitur      mungkin terlihat tidak penting secara individual, namun mereka memberikaan sinyal yang          bernilai ketika dikombinasikan dengan fitur lainnya.
- **Kehilangan efek ensemble**. Kelebihan tree-based model berada pada keberagamannya, di mana    semakin banyak fitur akan menghasilkan tree yang beragam juga. Adapun seleksi fitur dapat       mengurangi keberagaman ini.

### 3. Hyperparameter Tuning
<a id="hyperparameter-tuning"></a>
- Dilakukan hyperparameter tuning GridSearchCV

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

### Interpretation with SHAP Values
<a id="interpretation"></a>
- Untuk menginterpretasikan hasil Gradient Boosting, akan dianalisis nilai SHAP.
- SHAP adalah library yang memungkinkan interpretasi hasil algoritma machine learning. Dengan SHAP, dapat dipahami dampak masing-masing fitur terhadap prediksi model individu, di mana f(x) = E[f(X)] + SHAP.
- Secara sederhana, nilai SHAP dari sebuah fitur (seberapa besar pengaruhnya terhadap prediksi individu) adalah penjumlahan berbobot kontribusi marjinal dengan mempertimbangkan semua kemungkinan kombinasi fitur (feature coalitions).
- Feature coalition adalah kelompok fitur, dan nilainya merupakan prediksi model individu yang hanya menggunakan fitur-fitur dalam kelompok tersebut. Kontribusi marjinal dari sebuah fitur adalah perbedaan antara nilai prediksi untuk kombinasi fitur dengan dan tanpa fitur tersebut. Nilai kontribusi marjinal dijumlahkan untuk semua kemungkinan kombinasi dengan dan tanpa fitur tersebut. Bobotnya didasarkan pada probabilitas fitur yang sedang dihitung nilai SHAP-nya untuk berada dalam kombinasi tersebut.
- Sample salah satu karyawan

<img width="1143" height="600" alt="image" src="https://github.com/user-attachments/assets/f1a9e8a5-757e-4dd1-b570-c7288d9e62b7" />

- Karyawan ini diprediksi tidak attrit, dengan probabilitas attrit sangat rendah berdasarkan hasil transformasi log-odds melalui fungsi logistik.
- Salah satu faktor utama adalah MonthlyIncome yang mencapai 15,972, yang menurunkan log-odds attrit sebesar 0.55. Artinya, semakin tinggi gaji bulanan karyawan, semakin kecil kemungkinan ia untuk keluar dari perusahaan.
- Sebaliknya, fitur NumCompaniesWorked yang bernilai 6 justru meningkatkan log-odds attrit sebesar 0.58. Ini menunjukkan bahwa karyawan yang pernah bekerja di banyak perusahaan sebelumnya memiliki kecenderungan attrit yang lebih tinggi.
- Selain itu, fitur StockOptionLevel yang bernilai 3 juga memberikan pengaruh signifikan dalam menurunkan risiko attrit sebesar 0.38, mengindikasikan bahwa kepemilikan saham perusahaan dapat meningkatkan retensi karyawan.



### Feature Importance
<a id="feature-importance"></a>
<img width="997" height="568" alt="image" src="https://github.com/user-attachments/assets/f7cd71b7-6877-43f8-83af-1af2d28a70b6" />

- Berdasarkan hasil SHAP analysis, OverTime dan StockOptionLevel merupakan dua fitur yang paling berpengaruh terhadap prediksi attrition model. Nilai SHAP yang tinggi menunjukkan bahwa kedua variabel ini memiliki kontribusi signifikan dalam menentukan apakah seorang karyawan akan keluar dari perusahaan.
- Fitur-fitur penting lainnya antara lain MonthlyIncome, Age, JobLevel, dan BusinessTravel. Variabel-variabel ini memberikan pengaruh substantial terhadap output model, dimana nilai-nilai tertentu dari fitur tersebut dapat secara signifikan meningkatkan atau menurunkan probabilitas attrition.
- Hasil ini konsisten dengan logika bisnis dimana faktor-faktor seperti beban kerja (overtime), kompensasi finansial, level jabatan, dan frekuensi perjalanan dinas memang secara intuitif mempengaruhi keputusan karyawan untuk bertahan atau meninggalkan perusahaan.

- Sekarang, melalui beeswarm plot, kita dapat mengamati hubungan antara fitur-fitur dan prediksi model.

<img width="990" height="497" alt="image" src="https://github.com/user-attachments/assets/d7e0a746-e596-4599-a692-5ff919089dee" />

- OverTime merupakan fitur paling penting. Karyawan yang sering lembur (nilai tinggi/merah) memiliki dampak positif terhadap risiko attrition, sementara yang jarang lembur (nilai rendah/biru) justru menurunkan risiko attrition.
- StockOptionLevel menunjukkan pola yang logis - karyawan dengan level kepemilikan saham rendah cenderung memiliki risiko attrition lebih tinggi, sedangkan yang memiliki level saham tinggi lebih cenderung bertahan di perusahaan.
- MonthlyIncome memiliki dampak negatif terhadap attrition, dimana gaji tinggi (nilai merah) justru menurunkan risiko karyawan keluar dari perusahaan.
- Age menunjukkan bahwa karyawan usia muda lebih berisiko attrit dibanding karyawan senior.
- BusinessTravel yang frequent meningkatkan risiko attrition, menunjukkan bahwa intensitas perjalanan dinas yang tinggi dapat mempengaruhi keputusan karyawan untuk bertahan.


## Financial Result
<a id="financial-result"></a>
**Estimasi Dampak Finansial Model Attrition terhadap Perusahaan**

Untuk menunjukkan nilai tambah dari analisis ini, akan disajikan performa model dalam bentuk estimasi keuntungan finansial bagi perusahaan. Analisis ini didasarkan pada confusion matrix dan data yang tersedia saat ini.

**Asumsi Dasar:** Karena tidak tersedia data spesifik mengenai biaya attrition aktual, digunakan asumsi berdasarkan penelitian industri bahwa biaya mengganti karyawan berkisar antara 1.5x hingga 2x gaji tahunan. Digunakan asumsi konservatif sebesar 1.5x gaji tahunan.

**Komponen Biaya dan Manfaat yang Diperhitungkan:**
- **Biaya Retensi untuk False Positive (FP):**
Karyawan yang salah diprediksi akan attrit namun sebenarnya bertahan. Perusahaan akan mengeluarkan biaya retensi yang tidak perlu.
Asumsi: Perusahaan memberikan program retensi berupa bonus, training, atau promosi dengan biaya sebesar 30% dari gaji bulanan.
- **Kehilangan Pendapatan dari False Negative (FN):**
Karyawan yang benar-benar attrit namun gagal terdeteksi oleh model. Perusahaan menanggung seluruh biaya attrition sebesar 1.5x gaji tahunan.
- **Penghematan Biaya dari True Positive (TP):**
Karyawan yang diprediksi attrit dan berhasil dipertahankan. Perusahaan menghemat biaya attrition sebesar 1.5x gaji tahunan, dikurangi biaya retensi sebesar 30% dari gaji bulanan.

**Langkah Selanjutnya:** Dilakukan perhitungan proyeksi keuntungan/kerugian berdasarkan nilai-nilai di atas menggunakan dataset aktual untuk hasil finansial, dengan mempertimbangkan jumlah karyawan pada setiap kategori confusion matrix (TP, FP, FN) dan gaji tahunan mereka.

Model menghasilkan estimasi hasil finansial sekitar [MASUKKAN ANGGKA DI SINI]. Jumlah sebenarnya akan bergantung pada kebijakan manajemen perusahaan saat mengimplementasikan strategi retensi untuk karyawan berdasarkan probabilitas attrition yang diprediksi.

Sebagai contoh, jika perusahaan ingin bersikap lebih konservatif dengan mengurangi pengeluaran yang terkait dengan false positive, perusahaan dapat menargetkan karyawan dengan probabilitas attrition yang lebih tinggi, sehingga memengaruhi potensi keuntungan.

Namun demikian, untuk tujuan estimasi dan sebagai dasar pengambilan keputusan, kita telah memastikan bahwa proyek ini sangat layak untuk dilakukan.

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
Berdasarkan estimasi awal, proyek ini memiliki potensi memberikan dampak finansial sebesar $862.500. Besarnya nilai manfaat aktual tentu akan sangat tergantung pada struktur biaya yang ditetapkan perusahaan serta sejauh mana strategi retensi berbasis model ini diimplementasikan oleh manajemen. Meskipun demikian, estimasi ini memberikan dasar yang kuat untuk pengambilan keputusan bisnis.

### Langkah Selanjutnya
<a id="langkah-selanjutnya"></a>
Konten langkah selanjutnya...
