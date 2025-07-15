# Deteksi Cacat Label Kemasan - Flask Web Application

Aplikasi web berbasis Flask untuk mendeteksi cacat pada label kemasan menggunakan machine learning dengan ekstraksi fitur gambar dan Random Forest classifier.

## 🚀 Fitur Utama

- **Upload Gambar**: Interface web yang user-friendly untuk mengunggah gambar label kemasan
- **Deteksi Otomatis**: Model ML yang dilatih untuk membedakan label "good" dan "defect"
- **Hasil Detail**: Menampilkan prediksi dengan tingkat kepercayaan (confidence score)
- **Training Interface**: Web interface untuk upload data training dan training model
- **Sample Data Generator**: Script untuk generate sample data training
- **API Endpoint**: REST API untuk integrasi dengan sistem lain
- **Real-time Training**: Training model langsung melalui web interface

## 📁 Struktur Proyek

```
defect-detection/
├── app.py                    # Aplikasi Flask utama
├── real_model.py            # Model ML dengan Random Forest
├── create_dummy_model.py    # Script untuk membuat dummy model
├── create_sample_data.py    # Script untuk generate sample data
├── add_samples.py           # Script untuk menambah sample data
├── requirements.txt         # Dependensi Python
├── models/                  # Folder untuk menyimpan model
│   └── defect_detection_model.pkl
├── static/                  # File statis
│   ├── training_data/       # Data training
│   │   ├── good/           # Sample gambar label baik
│   │   └── defect/         # Sample gambar label cacat
│   └── uploads/            # Gambar yang diupload untuk testing
└── templates/              # Template HTML
    ├── index.html          # Halaman utama
    ├── result.html         # Halaman hasil deteksi
    └── training.html       # Halaman training model
```

## 🛠️ Instalasi dan Setup

### 1. Clone atau Download Proyek

```bash
cd /home/admin/defect-detection
```

### 2. Buat dan Aktifkan Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Di Linux/Mac
# atau
# venv\Scripts\activate  # Di Windows
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

**Note**: Jika ada error dengan TensorFlow, jalankan:
```bash
pip install numpy pillow opencv-python-headless scikit-learn
```

### 4. Generate Sample Data Training

Buat sample data untuk training model:

```bash
python create_sample_data.py
```

Script ini akan membuat:
- 15 sample gambar "good" (label berkualitas baik)
- 15 sample gambar "defect" (label dengan cacat)
- Data disimpan di `static/training_data/good/` dan `static/training_data/defect/`

### 5. Training Model Awal

```bash
python real_model.py
```

Atau buat dummy model untuk testing:
```bash
python create_dummy_model.py
```

### 6. Jalankan Aplikasi

```bash
source venv/bin/activate  # Pastikan venv aktif
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

## 📖 Cara Penggunaan

### 🚀 Running Project

1. **Aktifkan Virtual Environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Jalankan Aplikasi**:
   ```bash
   python app.py
   ```

3. **Akses Aplikasi**:
   - Halaman utama: `http://localhost:5000`
   - Halaman training: `http://localhost:5000/training`

### 🧠 Training Model

#### Melalui Web Interface (Recommended):

1. **Buka Halaman Training**:
   ```
   http://localhost:5000/training
   ```

2. **Upload Data Training** (Opsional - sudah ada sample data):
   - Pilih kategori: "Good" atau "Defect"
   - Upload multiple files sekaligus
   - Sistem akan menampilkan jumlah data per kategori

3. **Start Training**:
   - Klik tombol "Start Training"
   - Monitor progress real-time
   - Model akan ter-update otomatis setelah selesai

#### Melalui Command Line:

```bash
# Training dengan data yang ada
python real_model.py

# Hasilnya akan tersimpan di models/defect_detection_model.pkl
```

### 📊 Generate Sample Data

#### Generate Sample Data Baru:

```bash
# Generate 15 good + 15 defect samples
python create_sample_data.py
```

#### Generate Sample Tambahan:

```bash
# Generate 5 defect samples tambahan
python add_samples.py
```

#### Customize Sample Generation:

Edit script `create_sample_data.py` untuk mengubah:
- Jumlah sample yang di-generate
- Jenis defect yang disimulasikan
- Ukuran dan resolusi gambar

```python
# Di dalam create_sample_data.py
def create_sample_dataset(good_count=20, defect_count=20):
    # Ubah parameter sesuai kebutuhan
```

### 🔍 Testing Deteksi

1. **Upload Gambar Test**:
   - Buka `http://localhost:5000`
   - Upload gambar label kemasan
   - Lihat hasil prediksi + confidence score

2. **Menggunakan API**:
   ```bash
   curl -X POST \
     http://localhost:5000/api/predict \
     -H 'Content-Type: multipart/form-data' \
     -F 'file=@/path/to/your/image.jpg'
   ```

## 🤖 Model Machine Learning

### Arsitektur Model

Model menggunakan **Random Forest Classifier** dengan ekstraksi fitur gambar:

#### Fitur yang Diekstraksi:
- **RGB Statistics**: Mean dan standard deviation untuk setiap channel
- **Brightness & Contrast**: Nilai kecerahan dan kontras rata-rata
- **Edge Detection**: Density dari edge horizontal dan vertikal
- **Histogram Features**: Distribusi intensitas pixel (5 bins pertama)
- **Total Features**: 17 fitur per gambar

#### Keunggulan:
- **Fast Training**: Training lebih cepat dibanding deep learning
- **Interpretable**: Bisa melihat feature importance
- **No GPU Required**: Berjalan di CPU biasa
- **Small Model Size**: File model kecil (~KB)

### Training Model dengan Data Sendiri

#### 1. Siapkan Dataset

Struktur folder yang dibutuhkan:
```
static/training_data/
├── good/          # Gambar label berkualitas baik
│   ├── good1.jpg
│   ├── good2.png
│   └── ...
└── defect/        # Gambar label dengan cacat
    ├── defect1.jpg
    ├── defect2.png
    └── ...
```

#### 2. Upload Data via Web Interface

1. Buka `http://localhost:5000/training`
2. Pilih kategori (Good/Defect)
3. Upload multiple files sekaligus
4. Sistem akan menghitung total data per kategori

#### 3. Training Requirements

- **Minimum**: 5 gambar per kategori
- **Recommended**: 20+ gambar per kategori untuk hasil optimal
- **Format**: PNG, JPG, JPEG, BMP, GIF
- **Size**: Maksimal 16MB per file

#### 4. Automatic Training

```bash
# Training otomatis dengan data yang ada
python real_model.py
```

Atau gunakan web interface untuk training real-time dengan progress monitoring.

## � API Endpoint

### Prediction API

```bash
curl -X POST \
  http://localhost:5000/api/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg'
```

**Response JSON:**
```json
{
  "filename": "20250715_143022_abc123.jpg",
  "prediction": "good",
  "confidence": 0.95,
  "status": "success",
  "timestamp": "2025-07-15T14:30:22.123456"
}
```

### Training Status API

```bash
# Get training progress
curl http://localhost:5000/training_status
```

**Response:**
```json
{
  "status": "running",
  "current_step": 2,
  "message": "Training model...",
  "results": {}
}
```

## �📊 Format Input dan Output

### Input
- **Format gambar**: PNG, JPG, JPEG, GIF, BMP
- **Ukuran maksimum**: 16MB
- **Resolusi**: Otomatis diresize ke 64x64 pixels untuk feature extraction

### Output
- **Prediksi**: "good" atau "defect"
- **Confidence**: Nilai 0.0 - 1.0
- **Status**: "success" atau error message

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production dengan Gunicorn
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Opsional)
Buat `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
```

## 🔍 Troubleshooting

### ❌ Error "Model tidak tersedia"
**Solusi:**
```bash
# Generate dummy model untuk testing
python create_dummy_model.py

# Atau training model dengan data real
python real_model.py
```

### ❌ Error "ModuleNotFoundError"
**Solusi:**
```bash
# Pastikan virtual environment aktif
source venv/bin/activate

# Install dependencies
pip install numpy pillow opencv-python-headless scikit-learn
```

### ❌ Error "No training data found"
**Solusi:**
```bash
# Generate sample data
python create_sample_data.py

# Atau upload data manual ke static/training_data/good/ dan static/training_data/defect/
```

### ❌ Error "File terlalu besar"
**Solusi:**
- Kompres gambar sebelum upload
- Atau edit `MAX_CONTENT_LENGTH` di `app.py`

### ❌ Virtual Environment Issues
**Solusi:**
```bash
# Buat ulang virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🚀 Quick Start Commands

```bash
# Setup lengkap dari awal
python3 -m venv venv
source venv/bin/activate
pip install numpy pillow opencv-python-headless scikit-learn
python create_sample_data.py
python real_model.py
python app.py
```

Lalu buka: `http://localhost:5000`

## 📈 Pengembangan Lebih Lanjut

### Fitur yang Bisa Ditambahkan
- **Batch Processing**: Upload multiple images sekaligus
- **History**: Menyimpan riwayat prediksi di database
- **User Authentication**: System login dan user management
- **Advanced Models**: CNN, YOLO untuk object detection
- **Data Augmentation**: Teknik augmentasi untuk data training
- **Model Versioning**: Tracking versi model yang berbeda

### Optimasi Performance
- **Caching**: Redis untuk cache hasil prediksi
- **Load Balancing**: Multiple worker processes
- **Async Processing**: Background job untuk training dan prediksi
- **GPU Support**: Implementasi model deep learning dengan GPU

### Sample Data Customization

Edit `create_sample_data.py` untuk membuat sample data yang lebih realistis:

```python
def create_sample_defect_image(width=224, height=224):
    # Tambah jenis defect baru:
    # - Scratches (goresan)
    # - Color bleeding (warna luntur) 
    # - Misalignment (posisi tidak tepat)
    # - Texture issues (masalah tekstur)
    pass
```

### Model Improvement

Untuk akurasi yang lebih baik:

1. **Tambah Feature Engineering**:
   ```python
   # Di real_model.py, tambah fitur baru:
   # - GLCM (texture features)
   # - LBP (local binary patterns)
   # - HOG (histogram of gradients)
   ```

2. **Hyperparameter Tuning**:
   ```python
   # Tuning Random Forest parameters
   model = RandomForestClassifier(
       n_estimators=200,
       max_depth=10,
       min_samples_split=5
   )
   ```

3. **Ensemble Methods**:
   ```python
   # Kombinasi multiple models
   from sklearn.ensemble import VotingClassifier
   ```

## 📝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Tim Pengembang

- **Developer**: AI Assistant 
- **ML Engineer**: Random Forest Implementation
- **Data Scientist**: Feature Engineering & Model Training
- **Backend Developer**: Flask Application & API
- **UI/UX**: Web Interface Design

## 📞 Kontak

- **Repository**: [defect-detection-ml](https://github.com/estotriramdani/defect-detection-ml)
- **Issues**: Gunakan GitHub Issues untuk bug reports dan feature requests
- **Documentation**: README.md (file ini)

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ Basic ML model dengan Random Forest
- ✅ Web interface untuk upload dan prediksi
- ✅ Training interface dengan real-time progress
- ✅ Sample data generator
- ✅ API endpoints untuk integrasi
- ✅ Feature extraction dari gambar

### Roadmap
- 🔄 Deep learning model implementation
- 🔄 Batch processing capabilities
- 🔄 Advanced visualization dashboard
- 🔄 Model performance monitoring

---

**Dibuat dengan ❤️ menggunakan Flask, scikit-learn, dan Machine Learning**

> 💡 **Tips**: Untuk hasil optimal, gunakan gambar dengan resolusi yang baik dan pastikan label kemasan terlihat jelas. Model ini cocok untuk deteksi cacat visual seperti noda, goresan, robekan, dan masalah kualitas cetak.
