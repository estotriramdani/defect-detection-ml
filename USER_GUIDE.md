# 📖 Panduan Penggunaan Aplikasi Deteksi Cacat Label

Selamat datang di **Deteksi Cacat Label Kemasan**! Panduan ini akan membantu Anda menggunakan aplikasi web untuk mendeteksi cacat pada label kemasan menggunakan teknologi AI/Machine Learning.

## 🎯 Apa yang Bisa Dilakukan Aplikasi Ini?

Aplikasi ini dapat:
- ✅ **Mendeteksi kualitas label** - Membedakan label "good" (baik) atau "defect" (cacat)
- ✅ **Deteksi gambar unknown** - Mengidentifikasi gambar yang tidak sesuai dengan pola training
- ✅ **Training model sendiri** - Melatih AI dengan data gambar Anda sendiri
- ✅ **Manajemen dataset** - Mengelola data training dengan mudah
- ✅ **API untuk integrasi** - Dapat diintegrasikan dengan sistem lain

---

## 🚀 Memulai Aplikasi

### 1. Jalankan Aplikasi

```bash
# Masuk ke folder project
cd /home/admin/defect-detection

# Aktifkan virtual environment
source venv/bin/activate

# Jalankan aplikasi
python app.py
```

### 2. Buka di Browser

Akses aplikasi di: **http://localhost:5001**

---

## 🔍 Cara Menggunakan - Deteksi Gambar

### Langkah 1: Upload Gambar

1. **Buka halaman utama**: http://localhost:5001
2. **Klik area upload** atau **drag & drop** gambar Anda
3. **Format yang didukung**: PNG, JPG, JPEG, GIF, BMP (max 16MB)

### Langkah 2: Lihat Hasil Deteksi

Setelah upload, Anda akan melihat hasil:

#### ✅ **Hasil "GOOD"**
```
Prediksi: GOOD
Confidence: 85.5%
Status: success (anomaly score: 0.234)
```
→ Label berkualitas baik, layak untuk digunakan

#### ❌ **Hasil "DEFECT"** 
```
Prediksi: DEFECT
Confidence: 92.1%
Status: success (anomaly score: 0.567)
```
→ Label memiliki cacat, perlu diperbaiki atau diganti

#### ❓ **Hasil "UNKNOWN"**
```
Prediksi: UNKNOWN
Confidence: 45.2%
Status: Gambar tidak sesuai dengan pola training data
```
→ Gambar tidak dikenali oleh sistem, mungkin bukan label kemasan

### Langkah 3: Interpretasi Hasil

| **Metric** | **Penjelasan** | **Range** |
|------------|----------------|-----------|
| **Prediction** | Hasil klasifikasi utama | good / defect / unknown |
| **Confidence** | Tingkat kepercayaan model | 0% - 100% |
| **Anomaly Score** | Seberapa "aneh" gambar ini | 0.0 - ∞ |
| **Status** | Penjelasan detail hasil | Text description |

**Tips Interpretasi:**
- **Confidence > 80%**: Hasil sangat reliable
- **Confidence 60-80%**: Hasil cukup reliable
- **Confidence < 60%**: Mungkin perlu review manual
- **Anomaly Score > 1.5**: Gambar kemungkinan unknown

---

## 🎓 Cara Training Model - Untuk Admin/User Advanced

### Mengapa Perlu Training?

- **Data lebih spesifik**: Model disesuaikan dengan jenis label Anda
- **Akurasi lebih tinggi**: Semakin banyak data, semakin akurat
- **Adaptasi produk baru**: Training ulang saat ada produk/label baru

### Langkah 1: Akses Halaman Training

Buka: **http://localhost:5001/training**

### Langkah 2: Upload Data Training

#### 📂 **Upload Gambar "GOOD"**

1. **Pilih tab "Upload Gambar GOOD"** (area hijau)
2. **Drag & drop** atau **klik "Pilih File"**
3. **Upload multiple files** sekaligus (Ctrl+A untuk select all)
4. **Klik "Upload ke Server"**

**Contoh gambar GOOD:**
- Label utuh tanpa cacat
- Warna cetak normal
- Posisi text/logo tepat
- Tidak ada noda atau goresan

#### 📂 **Upload Gambar "DEFECT"**

1. **Pilih tab "Upload Gambar DEFECT"** (area merah)
2. **Upload gambar label dengan cacat**
3. **Klik "Upload ke Server"**

**Contoh gambar DEFECT:**
- Label robek atau rusak
- Warna pudar/salah
- Text buram atau tidak jelas
- Noda, goresan, atau kotoran
- Posisi salah atau miring

### Langkah 3: Kelola Dataset

#### 📊 **Lihat Data Training**

Di bagian **"Manajemen Data Training"**:

- **Tab "Gambar Good"**: Melihat semua data good yang sudah diupload
- **Tab "Gambar Defect"**: Melihat semua data defect yang sudah diupload
- **Counter real-time**: Jumlah gambar per kategori

#### 🗑️ **Hapus Data (Jika Diperlukan)**

**Hapus gambar individual:**
1. Klik tombol **"Hapus"** di gambar yang ingin dihapus
2. Konfirmasi penghapusan

**Hapus per kategori:**
1. Klik **"Hapus Semua Good"** atau **"Hapus Semua Defect"**
2. Konfirmasi penghapusan

**Hapus semua data:**
1. Klik **"Hapus Semua Data Training"** di bagian bawah
2. Konfirmasi penghapusan

### Langkah 4: Mulai Training

#### ✅ **Requirements Minimum**
- **Minimal 5 gambar** per kategori (good dan defect)
- **Recommended: 20+ gambar** per kategori untuk hasil optimal

#### 🚀 **Proses Training**

1. **Klik "Mulai Training"**
2. **Monitor progress** real-time:
   - ⏳ Memuat Dataset
   - 🔧 Preprocessing Data
   - 🧠 Training Model
   - 💾 Menyimpan Model

3. **Lihat hasil training**:
   - Training Accuracy
   - Validation Accuracy
   - Loss Score
   - Total Epochs

#### ⚡ **Training Selesai**

- Model baru otomatis tersimpan
- Langsung bisa digunakan untuk deteksi
- Klik **"Coba Model Baru"** untuk testing

---

## 🔧 Pengaturan Lanjutan

### Menyesuaikan Sensitivity Detection

Edit file `app.py` untuk mengubah threshold:

```python
# Pengaturan deteksi unknown
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence (0.0 - 1.0)
ANOMALY_THRESHOLD = 1.5     # Maximum anomaly score
```

**Efek Perubahan:**

| **Setting** | **Confidence** | **Anomaly** | **Efek** |
|-------------|----------------|-------------|----------|
| **Ketat** | 0.8 | 1.0 | Lebih banyak "unknown" |
| **Normal** | 0.6 | 1.5 | Default (balanced) |
| **Longgar** | 0.4 | 2.0 | Lebih sedikit "unknown" |

### Training dengan Command Line

```bash
# Training manual via script
source venv/bin/activate
python real_model.py

# Generate sample data untuk testing
python create_sample_data.py

# Test unknown detection
python test_unknown_detection.py
```

---

## 🌐 Penggunaan API (Untuk Developer)

### Endpoint Prediksi

```bash
curl -X POST \
  http://localhost:5001/api/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/image.jpg'
```

### Response Format

```json
{
  "filename": "20250715_143022_label.png",
  "prediction": "good",
  "confidence": 0.892,
  "anomaly_score": 0.234,
  "status": "success (anomaly score: 0.234)",
  "timestamp": "2025-07-15T14:30:22.123456"
}
```

### Integration Example (Python)

```python
import requests

def detect_label_defect(image_path):
    url = "http://localhost:5001/api/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    
    return response.json()

# Usage
result = detect_label_defect("label.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

---

## ❓ FAQ - Pertanyaan Umum

### Q: Berapa jumlah minimum data training?
**A:** Minimal 5 gambar per kategori, tapi untuk hasil optimal gunakan 20+ gambar per kategori.

### Q: Format gambar apa saja yang didukung?
**A:** PNG, JPG, JPEG, GIF, BMP dengan ukuran maksimal 16MB per file.

### Q: Bagaimana jika hasil deteksi "unknown"?
**A:** Gambar kemungkinan bukan label kemasan atau sangat berbeda dari data training. Periksa apakah gambar sesuai dengan yang diharapkan.

### Q: Apakah model perlu di-retrain berkala?
**A:** Ya, retrain model ketika:
- Ada jenis label/produk baru
- Performa deteksi menurun
- Data training bertambah significantly

### Q: Bisakah menggunakan model untuk jenis produk lain?
**A:** Model dirancang khusus untuk label kemasan. Untuk produk lain, perlu training ulang dengan data yang sesuai.

### Q: Bagaimana cara backup model yang sudah dilatih?
**A:** Copy file `models/defect_detection_model.pkl` ke lokasi backup. Restore dengan copy kembali ke folder `models/`.

---

## 🚨 Troubleshooting

### Error: "Model tidak tersedia"
```bash
# Solusi 1: Generate dummy model untuk testing
python create_dummy_model.py

# Solusi 2: Training model dengan data yang ada
python real_model.py
```

### Error: "Tidak ada file yang dipilih"
- Pastikan format file gambar yang didukung
- Cek ukuran file tidak melebihi 16MB
- Coba upload satu file dulu untuk testing

### Error: Upload gagal
- Periksa koneksi internet
- Restart aplikasi: `Ctrl+C` lalu `python app.py`
- Cek space disk tersedia

### Model selalu predict "good"
```bash
# Retrain model dengan data yang lebih seimbang
python retrain_model.py
```

### Training gagal
- Pastikan minimal 5 gambar per kategori
- Cek format gambar valid
- Restart aplikasi jika macet

---

## 💡 Tips & Best Practices

### 📷 **Tips Foto untuk Training Data**

#### Gambar "GOOD":
- **Lighting**: Pencahayaan merata, tidak terlalu gelap/terang
- **Angle**: Foto straight, tidak miring
- **Focus**: Gambar sharp, tidak blur
- **Background**: Background kontras dengan label
- **Resolution**: Minimal 200x200 pixels

#### Gambar "DEFECT":
- **Variety**: Berbagai jenis cacat (robek, noda, pudar, dll)
- **Severity**: Mix cacat ringan dan berat
- **Real defects**: Gunakan contoh cacat asli, bukan simulasi
- **Clear defects**: Cacat terlihat jelas di foto

### 🎯 **Tips Penggunaan Sehari-hari**

1. **Start small**: Mulai dengan 10-15 gambar per kategori
2. **Test frequently**: Test model setelah setiap training
3. **Monitor accuracy**: Catat performa sebelum/sesudah retrain
4. **Update gradually**: Tambah data training secara bertahap
5. **Document changes**: Catat perubahan setting dan hasilnya

### 📊 **Monitoring Performance**

**Indikator model bagus:**
- Training accuracy > 90%
- Validation accuracy > 80%
- Consistent results pada gambar test
- Anomaly detection bekerja untuk gambar aneh

**Tanda perlu retrain:**
- Accuracy turun drastis
- Banyak false positive/negative
- Model tidak mengenali produk baru
- Unknown detection terlalu sensitif/tidak sensitif

---

## 📞 Bantuan & Support

### 🔗 **Resources**
- **Technical Documentation**: `README.md`
- **Source Code**: Folder project `/home/admin/defect-detection`
- **Model Files**: Folder `models/`
- **Training Data**: Folder `static/training_data/`

### 🐛 **Melaporkan Bug**
Jika menemukan bug atau error:
1. Catat error message lengkap
2. Screenshot halaman error
3. Langkah-langkah untuk reproduce
4. File gambar yang menyebabkan error (jika ada)

### 💬 **Permintaan Fitur Baru**
Ide fitur baru yang bisa ditambahkan:
- Batch processing multiple images
- Export hasil deteksi ke CSV/Excel
- User authentication & management
- Advanced model (CNN/YOLO)
- Mobile app integration
- Real-time camera detection

---

## 🎉 Selamat Menggunakan!

Anda sekarang siap menggunakan **Deteksi Cacat Label Kemasan**! 

**Workflow Normal:**
1. 🚀 **Start app**: `python app.py`
2. 📷 **Upload image**: Drag & drop di homepage
3. 👀 **Check result**: Lihat prediction + confidence
4. 🎓 **Improve model**: Training dengan data baru jika diperlukan

**Pro Tips:**
- Mulai dengan sample data yang sudah ada
- Test berbagai jenis gambar untuk memahami kemampuan model
- Training ulang ketika ada jenis produk/cacat baru
- Monitor performance secara berkala

---

> 💡 **Ingat**: Semakin banyak dan beragam data training yang Anda berikan, semakin akurat model dalam mendeteksi cacat label. Model AI belajar dari contoh yang Anda berikan!

**Good luck & happy detecting!** 🚀✨
