# Deteksi Cacat Label Kemasan - Flask Web Application

Aplikasi web berbasis Flask untuk mendeteksi cacat pada label kemasan menggunakan model CNN (Convolutional Neural Network) dengan TensorFlow.

## 🚀 Fitur Utama

- **Upload Gambar**: Interface web yang user-friendly untuk mengunggah gambar label kemasan
- **Deteksi Otomatis**: Model CNN yang dilatih untuk membedakan label "good" dan "defect"
- **Hasil Detail**: Menampilkan prediksi dengan tingkat kepercayaan (confidence score)
- **API Endpoint**: REST API untuk integrasi dengan sistem lain
- **Responsive Design**: Tampilan yang optimal di desktop dan mobile

## 📁 Struktur Proyek

```
defect-detection/
├── app.py              # Aplikasi Flask utama
├── config.py           # Konfigurasi aplikasi
├── utils.py            # Fungsi-fungsi utility
├── create_model.py     # Script untuk membuat model CNN
├── requirements.txt    # Dependensi Python
├── models/            # Folder untuk menyimpan model
│   └── defect_detection_model.h5
├── static/            # File statis (CSS, JS, images)
│   └── uploads/       # Folder untuk gambar yang diupload
└── templates/         # Template HTML
    ├── index.html     # Halaman utama
    └── result.html    # Halaman hasil deteksi
```

## 🛠️ Instalasi dan Setup

### 1. Clone atau Download Proyek

```bash
cd /home/admin/defect-detection
```

### 2. Buat Virtual Environment (Opsional tapi Disarankan)

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

### 4. Buat Model CNN (Dummy untuk Testing)

```bash
python create_model.py
```

**Catatan**: Script ini akan membuat model dummy untuk demonstrasi. Untuk penggunaan production, Anda perlu melatih model dengan dataset yang sesuai.

### 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di: `http://localhost:5000`

## 📖 Cara Penggunaan

### Web Interface

1. Buka browser dan akses `http://localhost:5000`
2. Upload gambar label kemasan melalui drag & drop atau tombol "Pilih File"
3. Klik "Analisis Gambar" untuk memulai deteksi
4. Lihat hasil prediksi beserta tingkat kepercayaan

### API Endpoint

Untuk integrasi dengan aplikasi lain, gunakan endpoint API:

```bash
curl -X POST \
  http://localhost:5000/api/predict \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg'
```

Response JSON:
```json
{
  "filename": "20250715_143022_abc123.jpg",
  "prediction": "good",
  "confidence": 0.95,
  "status": "success",
  "timestamp": "2025-07-15T14:30:22.123456"
}
```

## 🔧 Konfigurasi

### Environment Variables

Untuk production, set environment variables berikut:

```bash
export SECRET_KEY="your-secret-key-here"
export FLASK_ENV="production"
```

### Kustomisasi Model

Edit file `config.py` untuk mengubah:
- Path model CNN
- Ukuran maksimum file upload
- Format file yang diizinkan

## 🎯 Model CNN

### Arsitektur Model

Model menggunakan arsitektur CNN dengan layers:
- 4 Convolutional layers dengan MaxPooling
- Dropout layers untuk regularization
- Dense layers untuk klasifikasi final
- Output: 2 kelas (good/defect)

### Training Model dengan Data Sendiri

Untuk melatih model dengan dataset Anda:

1. **Siapkan Dataset**:
   ```
   dataset/
   ├── train/
   │   ├── good/
   │   └── defect/
   └── validation/
       ├── good/
       └── defect/
   ```

2. **Modifikasi `create_model.py`** untuk menambahkan:
   - Data loading dan preprocessing
   - Training loop
   - Model evaluation
   - Model saving

3. **Jalankan Training**:
   ```bash
   python create_model.py
   ```

## 📊 Format Input dan Output

### Input
- **Format gambar**: PNG, JPG, JPEG, GIF, BMP
- **Ukuran maksimum**: 16MB
- **Resolusi**: Otomatis diresize ke 224x224 pixels

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

### Error "Model tidak tersedia"
- Pastikan file `models/defect_detection_model.h5` ada
- Jalankan `python create_model.py` untuk membuat model dummy

### Error "File terlalu besar"
- Ubah `MAX_CONTENT_LENGTH` di `config.py`
- Atau compress gambar sebelum upload

### Error Dependencies
- Pastikan semua package di `requirements.txt` terinstall
- Coba update pip: `pip install --upgrade pip`

## 📈 Pengembangan Lebih Lanjut

### Fitur yang Bisa Ditambahkan
- **Batch Processing**: Upload multiple images
- **History**: Menyimpan riwayat prediksi
- **User Authentication**: Login system
- **Database Integration**: Menyimpan hasil ke database
- **Real-time Processing**: WebSocket untuk real-time updates
- **Model Monitoring**: Tracking performa model

### Optimasi Performance
- **Caching**: Redis untuk cache hasil prediksi
- **Load Balancing**: Multiple worker processes
- **GPU Acceleration**: CUDA support untuk inferensi lebih cepat

## 📝 Contributing

1. Fork repository
2. Buat feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Tim Pengembang

- **Developer**: [Your Name]
- **AI/ML Engineer**: [Your Name]
- **UI/UX Designer**: [Your Name]

## 📞 Kontak

- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/defect-detection](https://github.com/yourusername/defect-detection)

---

**Dibuat dengan ❤️ menggunakan Flask dan TensorFlow**
