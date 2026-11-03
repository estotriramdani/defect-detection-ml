# Deteksi Cacat Label Kemasan

Aplikasi web Flask untuk mengklasifikasikan kualitas label kemasan menjadi **good**, **defect**, atau **unknown**. Model menggunakan **Random Forest** (scikit-learn) dengan ekstraksi fitur gambar dan *anomaly detection* berbasis jarak fitur.

> Panduan penggunaan untuk end-user: [`USER_GUIDE.md`](USER_GUIDE.md)

## Fitur

- Upload gambar (drag & drop) dan prediksi real-time.
- Klasifikasi tiga kelas: `good`, `defect`, `unknown`.
- Confidence score + anomaly score (dual threshold).
- Training model langsung dari web, dengan progress monitoring.
- Manajemen dataset training: lihat, hapus per gambar, hapus per kategori.
- REST API untuk integrasi (`POST /api/predict`).

## Teknologi

| Komponen | Keterangan |
|---|---|
| Web | Flask 3, Jinja2, Bootstrap 5 (CDN) |
| ML | scikit-learn `RandomForestClassifier` |
| Gambar | Pillow, NumPy |
| Deployment | PM2 (`ecosystem.config.js`) |

Model **bukan** deep learning/CNN: gambar di-resize ke 64×64 lalu diubah menjadi 15 fitur numerik (mean & std RGB, brightness, contrast, edge density, 5 bin histogram). Kelebihan: ringan, cepat, tanpa GPU.

## Struktur Proyek

```
defect-detection-ml/
├── app.py                     # Application factory & entry point
├── config.py                  # Konfigurasi (environment variable)
├── auth.py                    # HTTP Basic Auth untuk area admin
├── utils.py                   # Helper file & dataset training
├── blueprints/
│   ├── main.py                # Halaman utama & upload prediksi
│   ├── api.py                 # REST API
│   └── training.py            # Training & manajemen dataset (admin)
├── services/
│   ├── model_service.py       # State model & logika prediksi
│   └── training_service.py    # Background training + lock
├── real_model.py              # SimpleDefectDetector: fitur & training
├── generate_from_base.py      # Generator dataset dari gambar label asli (disarankan)
├── create_sample_data.py      # Generator dataset sintetis (alternatif)
├── retrain_model.py           # Retrain model dari CLI
├── test_unknown_detection.py  # Cek cepat perilaku deteksi unknown
├── requirements.txt
├── .env.example               # Contoh konfigurasi environment
├── ecosystem.config.js        # Konfigurasi PM2
├── README.md
├── USER_GUIDE.md
├── models/                    # model .pkl (dibuat saat training, tidak di-commit)
├── static/
│   ├── base-data/             # Contoh gambar untuk testing
│   ├── training_data/         # Dataset training (good/ & defect/)
│   └── uploads/               # Gambar hasil upload
└── templates/
    ├── index.html
    ├── result.html
    └── training.html
```

Folder `models/`, `static/training_data/good/`, `static/training_data/defect/`, dan `static/uploads/` dibuat otomatis saat aplikasi dijalankan.

## Instalasi

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Menjalankan Aplikasi

```bash
# 1. Buat dataset contoh dari label asli (26 good + 25 defect)
python generate_from_base.py

# 2. Latih model awal
python retrain_model.py

# 3. Jalankan aplikasi
python app.py
```

Jika tidak memiliki gambar label asli, gunakan generator sintetis:

```bash
python create_sample_data.py   # 15 good + 15 defect
```

Akses **http://localhost:5001** (halaman training: `/training`). Port dapat diubah lewat environment variable `PORT`.

Model dimuat saat startup dari `models/defect_detection_model.pkl`. Jika file belum ada, aplikasi tetap jalan tetapi prediksi mengembalikan error "Model tidak tersedia" sampai model dilatih.

## Konfigurasi

Konfigurasi dibaca dari environment variable. Untuk development, salin `.env.example` menjadi `.env` (dimuat otomatis oleh `python-dotenv`).

| Variable | Default | Keterangan |
|---|---|---|
| `SECRET_KEY` | dev key | Wajib diganti di production |
| `FLASK_DEBUG` | `0` | Set `1` untuk auto-reload |
| `HOST` / `PORT` | `0.0.0.0` / `5001` | Alamat server |
| `CONFIDENCE_THRESHOLD` | `0.6` | Ambang minimum confidence |
| `ANOMALY_THRESHOLD` | `1.5` | Ambang maksimum anomaly score |
| `TRAIN_MIN_PER_CLASS` | `5` | Minimum gambar per kategori untuk training |
| `ADMIN_USERNAME` | `admin` | Username area admin |
| `ADMIN_PASSWORD` | *(kosong)* | Jika kosong, autentikasi admin **nonaktif** |
| `MODEL_PATH` | `models/...pkl` | Lokasi file model |
| `MAX_CONTENT_LENGTH` | `16777216` | Batas ukuran upload (byte) |

> Untuk production, selalu set `SECRET_KEY` dan `ADMIN_PASSWORD`, serta `FLASK_DEBUG=0`.

## Autentikasi Admin

Halaman training dan seluruh endpoint manajemen dataset dilindungi **HTTP Basic Auth** ketika `ADMIN_PASSWORD` di-set:

```bash
export ADMIN_USERNAME=admin
export ADMIN_PASSWORD=rahasia
python app.py
```

Browser akan menampilkan dialog login saat membuka `/training`. Endpoint API prediksi (`/api/predict`) tetap publik.

## Training

### Lewat Web

1. Buka `http://localhost:5001/training`.
2. Upload gambar ke kategori **Good** dan **Defect** (minimal 5 per kategori, disarankan 20+).
3. Klik **Mulai Training** dan tunggu hingga selesai.
4. Model baru langsung dipakai tanpa restart.

### Lewat CLI

```bash
python retrain_model.py        # training + simpan model
python generate_from_base.py   # regenerate dataset dari static/base-data/label_fibemini.png
python create_sample_data.py   # alternatif: dataset sintetis
```

Dataset manual mengikuti struktur:

```
static/training_data/
├── good/     # label berkualitas baik
└── defect/   # label cacat
```

Format yang didukung: PNG, JPG, JPEG, BMP, GIF (maks. 16 MB per file).

## Deteksi Unknown

Prediksi dianggap `unknown` jika salah satu kondisi terpenuhi:

| Parameter | Default | Lokasi |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | 0.6 | environment variable |
| `ANOMALY_THRESHOLD` | 1.5 | environment variable |

- Turunkan threshold → lebih sensitif (lebih banyak `unknown`).
- Naikkan threshold → lebih longgar.

Anomaly score dihitung dari rata-rata jarak Euclidean fitur gambar ke seluruh sampel training, dinormalisasi dengan persentil ke-90 jarak antar-sampel, dan dibatasi maksimal 3.0.

## API

### `POST /api/predict`

```bash
curl -X POST http://localhost:5001/api/predict \
  -F "file=@path/to/image.jpg"
```

Contoh respons:

```json
{
  "filename": "20251103_101500_label.png",
  "prediction": "good",
  "confidence": 0.892,
  "anomaly_score": 0.234,
  "status": "success (anomaly score: 0.234)",
  "timestamp": "2025-11-03T10:15:00.123456"
}
```

Nilai `prediction`: `good`, `defect`, `unknown`, atau `Error`. Kode HTTP 400 untuk file tidak ada/tidak didukung.

### Endpoint lain

Endpoint bertanda 🔒 memerlukan Basic Auth jika `ADMIN_PASSWORD` di-set.

| Method | Path | Fungsi | Auth |
|---|---|---|---|
| GET | `/` | Halaman upload | – |
| POST | `/upload` | Upload + tampilkan halaman hasil | – |
| POST | `/api/predict` | Prediksi (JSON) | – |
| GET | `/training` | Halaman training | 🔒 |
| POST | `/upload_training_data` | Upload dataset (field `files`, `category`) | 🔒 |
| POST | `/start_training` | Mulai training background | 🔒 |
| GET | `/training_status` | Status training (JSON) | 🔒 |
| GET | `/get_training_data` | Daftar gambar training (JSON) | 🔒 |
| POST | `/delete_training_image` | Hapus satu gambar (`category`, `filename`) | 🔒 |
| POST | `/clear_training_data` | Hapus per kategori / semua (`category`) | 🔒 |

## Deployment (PM2)

```bash
pm2 start ecosystem.config.js
pm2 logs defect-detection-ml
```

`ecosystem.config.js` mengasumsikan interpreter `venv/bin/python` (Linux). Untuk Windows, sesuaikan ke `venv\\Scripts\\python.exe`.

## Troubleshooting

| Gejala | Solusi |
|---|---|
| `Model tidak tersedia` | Jalankan `python retrain_model.py` |
| `ModuleNotFoundError` | Aktifkan venv lalu `pip install -r requirements.txt` |
| `No training data found` | Jalankan `python generate_from_base.py` atau upload data |
| Semua gambar jadi `good` | Retrain dengan dataset lebih seimbang/beragam |
| File terlalu besar | Kompres gambar atau ubah `MAX_CONTENT_LENGTH` |

## Catatan Pengembangan

- Threshold, port, dan kredensial admin dikonfigurasi lewat environment variable.
- `FLASK_DEBUG` default `0`; jangan aktifkan debug di production.
- Endpoint admin hanya dilindungi Basic Auth sederhana; untuk deployment publik gunakan HTTPS/reverse proxy.

## Lisensi

MIT.
