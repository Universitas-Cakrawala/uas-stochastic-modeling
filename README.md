# Dokumentasi Analisis Prediksi Harga Properti Melbourne
**Mata Kuliah: Stochastic Modeling**  
**Topik: Supervised Learning untuk Prediksi Harga Properti**

---

## 📋 Daftar Isi
1. [Pendahuluan](#pendahuluan)
2. [Dataset dan Sumber Data](#dataset-dan-sumber-data)
3. [Metodologi](#metodologi)
4. [Hasil Analisis](#hasil-analisis)
5. [Evaluasi Model](#evaluasi-model)
6. [Kesimpulan](#kesimpulan)
7. [Saran dan Pengembangan](#saran-dan-pengembangan)

---

## 🎯 Pendahuluan

### Latar Belakang
Pasar properti, khususnya di Melbourne dan Sydney, menawarkan peluang yang menarik untuk analisis data dan prediksi harga. Prediksi harga properti menjadi sangat penting karena:
- Harga properti merupakan indikator kondisi pasar secara keseluruhan
- Membantu kesehatan ekonomi suatu negara
- Memberikan insight berharga bagi pembeli, penjual, dan investor

### Tujuan Penelitian
Mengembangkan model Machine Learning menggunakan **Supervised Learning** untuk memprediksi harga properti berdasarkan fitur-fitur yang relevan, sehingga dapat:
- Mengidentifikasi tren harga properti
- Memberikan wawasan berguna bagi stakeholder pasar properti
- Membantu pengambilan keputusan investasi

---

## 📊 Dataset dan Sumber Data

### Sumber Dataset
- **Nama**: Melbourne Housing Market Dataset
- **Sumber**: [Kaggle - Melbourne Housing Market](https://www.kaggle.com/datasets/shree1992/housedata)
- **File**: `data.csv`

### Deskripsi Variabel
Dataset berisi **18 kolom** dengan informasi lengkap tentang properti:

| No | Kolom | Deskripsi |
|----|-------|-----------|
| 1 | `date` | Tanggal rumah dijual |
| 2 | `price` | **Harga rumah (Target Variable)** |
| 3 | `bedrooms` | Jumlah kamar tidur |
| 4 | `bathrooms` | Jumlah kamar mandi |
| 5 | `sqft_living` | Luas bangunan |
| 6 | `sqft_lot` | Luas tanah |
| 7 | `floors` | Jumlah lantai |
| 8 | `waterfront` | Apakah berbatasan dengan air |
| 9 | `view` | Rating view dari rumah |
| 10 | `condition` | Kondisi rumah |
| 11 | `sqft_above` | Luas bangunan di atas tanah |
| 12 | `sqft_basement` | Luas basement |
| 13 | `yr_built` | Tahun dibangun |
| 14 | `yr_renovated` | Tahun direnovasi |
| 15 | `street` | Nama jalan |
| 16 | `city` | Kota |
| 17 | `statezip` | Kode pos |
| 18 | `country` | Negara |

### Tantangan Dataset
- **Kualitas data**: Potensi entri duplikat dan nilai yang hilang
- **Distribusi target**: Harga properti miring dengan outlier ekstrem
- **Multikolinearitas**: Fitur struktural yang saling berkorelasi
- **Feature engineering**: Perlu transformasi fitur waktu dan lokasi

---

## 🔍 Metodologi

### 1. Data Cleaning & Preprocessing

#### Pengecekan Kualitas Data
```python
# Cek duplikasi dan missing values
df.duplicated().sum()  # Hasil: 0 duplikasi
df.isnull().sum()     # Hasil: Tidak ada missing values
```

#### Penanganan Nilai Anomali
Menghapus data dengan nilai 0 yang tidak logis:
- **Price = 0**: 49 baris dihapus
- **Bedrooms = 0**: 2 baris dihapus  
- **Bathrooms = 0**: 2 baris dihapus

### 2. Exploratory Data Analysis (EDA)

#### Analisis Korelasi

![Correlation Heatmap](/assets/image_correlation_heatmap.png)

**Gambar 1: Heatmap Korelasi Variabel dengan Price**

Heatmap di atas menunjukkan kekuatan hubungan antara setiap variabel numerik dengan harga properti (`price`). Warna yang lebih gelap (biru) menunjukkan korelasi yang lebih rendah, sedangkan warna yang lebih terang (merah) menunjukkan korelasi yang lebih tinggi.

**Insight dari Correlation Analysis:**
- `sqft_living` memiliki korelasi tertinggi (0.43) - luas bangunan menjadi faktor utama
- `sqft_above` (0.37) dan `bathrooms` (0.33) juga berkorelasi kuat dengan harga
- `view` (0.23) dan `sqft_basement` (0.21) menunjukkan korelasi moderat
- Variabel seperti `yr_renovated` bahkan memiliki korelasi negatif (-0.029)

### 3. Outlier Detection & Removal

Menggunakan **IQR Method** untuk mendeteksi outlier pada variabel kunci:

```python
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)
```

**Variabel yang diproses:**
- `price` (target variable)
- `sqft_living`
- `sqft_above` 
- `sqft_basement`

### 4. Feature Selection

Berdasarkan analisis korelasi, dipilih **5 fitur utama**:
- `sqft_living` (korelasi: 0.43)
- `sqft_above` (korelasi: 0.37)
- `bathrooms` (korelasi: 0.33)
- `view` (korelasi: 0.23)
- `sqft_basement` (korelasi: 0.21)

### 5. Data Splitting
- **Training set**: 70% data
- **Testing set**: 30% data
- **Random state**: 42 (untuk reproducibility)

---

## 🤖 Model Development

### Supervised Learning Algorithms

Mengimplementasikan **3 algoritma regresi**:

1. **Linear Regression**
   - Model dasar untuk baseline performance
   - Asumsi hubungan linear antara fitur dan target

2. **Random Forest Regressor**
   - Ensemble method dengan multiple decision trees
   - Mengatasi overfitting dan menangani non-linearity

3. **Decision Tree Regressor**
   - Model berbasis pohon keputusan
   - Mudah diinterpretasi dan menangani interaksi fitur

---

## 📈 Hasil Analisis

### 1. Evaluasi Model Regresi

![Model Predictions Comparison](/assets/image_prediksi_aktual.png)

**Gambar 2: Perbandingan Prediksi vs Nilai Aktual untuk Ketiga Model**

Scatter plot di atas menunjukkan perbandingan antara harga aktual (sumbu x) dengan harga prediksi (sumbu y) untuk ketiga model. Garis diagonal merah menunjukkan prediksi yang sempurna (prediksi = aktual). Semakin dekat titik-titik dengan garis diagonal, semakin akurat prediksi model.

**Analisis Visual:**
- **Linear Regression**: Menunjukkan pola yang paling konsisten dan mendekati garis diagonal
- **Random Forest**: Prediksi cenderung lebih tersebar namun masih mengikuti pola umum
- **Decision Tree**: Menunjukkan pola yang paling tidak konsisten dengan banyak outlier

#### Metrik Evaluasi

| Model | MAE ($) | RMSE ($) | R² Score |
|-------|---------|----------|----------|
| **Linear Regression** | 127,432 | 159,156 | **0.4173** |
| **Random Forest** | 135,080 | 172,394 | 0.2937 |
| **Decision Tree** | 169,447 | 218,076 | -0.0665 |

![Performance Metrics Comparison](/assets/image_performance_metrics.png)

**Gambar 3: Perbandingan Metrik Performa Model**

Bar chart di atas memvisualisasikan tiga metrik evaluasi utama:
- **MAE (Mean Absolute Error)**: Rata-rata kesalahan absolut dalam dollar
- **RMSE (Root Mean Squared Error)**: Akar dari rata-rata kuadrat kesalahan
- **R² Score**: Proporsi variasi yang dapat dijelaskan oleh model (semakin tinggi semakin baik)

#### Key Findings:
- **Linear Regression** menunjukkan performa terbaik dengan R² = 0.4173
- Random Forest berada di posisi kedua meski lebih kompleks
- Decision Tree mengalami overfitting dengan R² negatif

### 2. Analisis Klasifikasi (Demonstrasi)

#### Kategorisasi Harga
- **Murah**: < $300,000
- **Sedang**: $300,000 - $600,000  
- **Mahal**: > $600,000

#### Classification Report

![Classification Report](/assets/image_classification_report.png)

**Gambar 5: Laporan Klasifikasi Detail**

Tabel di atas menunjukkan metrik evaluasi yang komprehensif untuk setiap kategori harga:

**Penjelasan Metrik:**
- **Precision**: Proporsi prediksi positif yang benar untuk setiap kategori
- **Recall**: Proporsi sampel positif aktual yang berhasil diprediksi dengan benar
- **F1-Score**: Harmonic mean dari precision dan recall
- **Support**: Jumlah sampel aktual untuk setiap kategori

**Analisis per Kategori:**
- **Murah (<$300k)**: Precision rendah (0.41) - banyak false positive
- **Sedang ($300k-$600k)**: Performa terbaik dengan precision 0.59 dan recall 0.68
- **Mahal (>$600k)**: Balanced performance dengan precision 0.58 dan recall 0.51

```
                   precision  recall  f1-score  support
Murah (<$300k)         0.41    0.32      0.36      269
Sedang ($300k-$600k)   0.59    0.68      0.63      656
Mahal (>$600k)         0.58    0.51      0.54      334

accuracy                                 0.56     1259
macro avg              0.53    0.50      0.51     1259
weighted avg           0.55    0.56      0.55     1259
```

#### Confusion Matrix

![Confusion Matrix](/assets/image_confusion_matrix.png)

**Gambar 4: Confusion Matrix untuk Klasifikasi Harga Properti**

Matrix konfusi di atas menunjukkan performa model klasifikasi dalam memprediksi kategori harga:
- **Diagonal utama** (85, 449, 170): Prediksi yang benar untuk setiap kategori
- **Off-diagonal**: Kesalahan klasifikasi antar kategori
- **Warna yang lebih gelap**: Menunjukkan jumlah prediksi yang lebih tinggi

**Analisis Matrix:**
- Model paling akurat dalam memprediksi kategori "Sedang" (449 benar dari 656 total)
- Kesulitan terbesar dalam membedakan kategori "Murah" dan "Sedang" (171 kesalahan)
- Kategori "Mahal" sering diprediksi sebagai "Sedang" (143 kesalahan)

**Insights:**
- Model klasifikasi mencapai akurasi 56%
- Performa terbaik pada kategori "Sedang" (precision: 0.59)
- Kesulitan membedakan kategori "Murah" dan "Mahal"

---

## 📊 Evaluasi Model

### Strengths
✅ **Linear Regression** memberikan hasil prediksi yang stabil  
✅ Model berhasil menangkap pola umum dalam data  
✅ Preprocessing yang komprehensif meningkatkan kualitas model  
✅ Feature selection berbasis korelasi efektif  

### Limitations
❌ R² Score masih relatif rendah (41.7%)  
❌ Model kesulitan menangani variabilitas harga yang tinggi  
❌ Outlier masih mempengaruhi performa model  
❌ Fitur lokasi belum dimanfaatkan optimal  

### Model Performance Analysis
- **Mean Absolute Error** sekitar $127K menunjukkan prediksi masih jauh dari nilai aktual
- **R² Score 0.4173** berarti model menjelaskan ~42% variasi dalam data
- Gap yang besar antara harga minimum dan maksimum menjadi tantangan utama

---

## 🎯 Kesimpulan

### Hasil Utama
1. **Linear Regression** terbukti menjadi model terbaik untuk dataset ini
2. Fitur `sqft_living`, `sqft_above`, dan `bathrooms` adalah prediktor terkuat
3. Preprocessing dan outlier removal berhasil meningkatkan kualitas data
4. Model mampu menangkap tren umum namun masih perlu improvement

### Kontribusi Penelitian
- Demonstrasi aplikasi supervised learning pada prediksi harga properti
- Perbandingan performa multiple algorithms
- Framework preprocessing yang dapat direplikasi
- Insights tentang faktor-faktor yang mempengaruhi harga properti

---

## 🚀 Saran dan Pengembangan

### Immediate Improvements
1. **Feature Engineering**
   - Membuat fitur `house_age` dari `yr_built`
   - Ekstraksi informasi dari alamat (kode pos, area)
   - Rasio luas basement terhadap total luas

2. **Advanced Modeling**
   - Hyperparameter tuning untuk Random Forest
   - Implementasi Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks untuk menangkap pola kompleks

3. **Data Enhancement**
   - Menambah data eksternal (ekonomi, demografi)
   - Informasi geografis (jarak ke CBD, sekolah, transportasi)
   - Data time series untuk tren temporal

### Research Extensions
- **Cross-validation** untuk validasi model yang lebih robust
- **Ensemble methods** untuk meningkatkan prediksi
- **Interpretability analysis** menggunakan SHAP atau LIME
- **Real-time prediction system** dengan data streaming

---

## 📚 Referensi

1. Dataset: [Kaggle - Melbourne Housing Market](https://www.kaggle.com/datasets/shree1992/housedata)
2. Scikit-learn Documentation
3. Pandas Documentation
4. Seaborn & Matplotlib Documentation

---

**Terima Kasih!**

*Presentasi ini mendemonstrasikan penerapan supervised learning dalam prediksi harga properti dengan pendekatan systematic dan data-driven approach.*
