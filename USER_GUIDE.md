# Panduan Penggunaan — Deteksi Cacat Label Kemasan

Panduan ini ditujukan untuk pengguna aplikasi (operator/QA). Untuk informasi teknis, lihat [`README.md`](README.md).

## Kemampuan Aplikasi

- Menilai kualitas label kemasan: **GOOD** (baik), **DEFECT** (cacat), atau **UNKNOWN** (tidak dikenali).
- Melatih ulang model dengan dataset milik Anda sendiri.
- Mengelola dataset training (lihat, hapus, reset).
- Menyediakan API untuk integrasi dengan sistem lain.

## Menjalankan Aplikasi

```bash
# Aktifkan virtual environment
source venv/bin/activate        # Linux/macOS
.\venv\Scripts\Activate.ps1    # Windows PowerShell

# Jalankan
python app.py
```

Buka browser ke **http://localhost:5001**.

Jika administrator mengaktifkan autentikasi (`ADMIN_PASSWORD`), halaman training akan meminta username/password sebelum dapat diakses.

Jika model belum tersedia, jalankan sekali:

```bash
python generate_from_base.py
python retrain_model.py
```

Atau gunakan dataset sintetis bila tidak ada gambar label asli:

```bash
python create_sample_data.py
```

## Deteksi Gambar

1. Buka **http://localhost:5001**.
2. Drag & drop gambar ke area upload, atau klik **Pilih File**.
3. Format: PNG, JPG, JPEG, GIF, BMP — maksimal 16 MB.
4. Tunggu hasil prediksi.

### Membaca Hasil

| Hasil | Arti | Tindakan |
|---|---|---|
| **GOOD** | Label berkualitas baik | Lanjut ke proses berikutnya |
| **DEFECT** | Label cacat | Inspeksi manual / perbaiki |
| **UNKNOWN** | Gambar tidak sesuai pola training | Periksa ulang gambar atau tambah data training |

Setiap hasil menampilkan **confidence** (0–100%). Semakin tinggi, semakin yakin model.

Pedoman umum:

- Confidence ≥ 80% → hasil cukup dapat diandalkan.
- Confidence < 60% → model ragu; lakukan verifikasi manual.
- Anomaly score tinggi (di atas 1.5) → gambar dianggap di luar pola training.

## Training Model

### Kapan perlu training ulang?

- Ada jenis label/produk baru.
- Hasil deteksi mulai banyak yang salah.
- Dataset training bertambah banyak.

### Langkah-langkah

1. Buka **http://localhost:5001/training**.
2. Pada bagian upload, pilih kategori **Good** atau **Defect**, lalu unggah beberapa gambar sekaligus.
3. Pastikan jumlah minimal **5 gambar per kategori** (disarankan 20+).
4. Klik **Mulai Training**.
5. Tunggu hingga progress selesai; model langsung aktif tanpa restart.

### Kriteria gambar training

**Good:**
- Label utuh, tidak ada cacat.
- Warna cetak normal, teks/logo tajam.
- Pencahayaan merata, gambar tidak blur.

**Defect:**
- Label robek, tergores, atau bernoda.
- Warna pudar/salah, cetakan buram.
- Posisi teks/logo tidak tepat.
- Gunakan variasi jenis dan tingkat keparahan cacat.

### Mengelola Dataset

Di halaman training, bagian **Manajemen Data Training**:

- **Lihat** thumbnail semua gambar per kategori.
- **Hapus** satu gambar dengan tombol hapus.
- **Hapus Semua Good / Defect** untuk mengosongkan satu kategori.
- **Hapus Semua Data Training** untuk reset total.

Setelah mengubah dataset, lakukan training ulang agar model menyesuaikan.

### Hasil Training

Setelah selesai, aplikasi menampilkan:

- **Training accuracy** — akurasi pada data latih.
- **Validation accuracy** — akurasi pada 20% data yang disisihkan.
- Jumlah sampel training dan test.

Training accuracy yang sangat tinggi (mis. 100%) dengan validation accuracy lebih rendah adalah hal normal pada dataset kecil.

## Pengaturan Sensitivitas

Threshold deteksi `unknown` diatur melalui environment variable (atau file `.env`):

```bash
CONFIDENCE_THRESHOLD=0.6   # minimum confidence
ANOMALY_THRESHOLD=1.5      # maksimum anomaly score
```

| Mode | Confidence | Anomaly | Efek |
|---|---|---|---|
| Ketat | 0.8 | 1.0 | Lebih banyak hasil UNKNOWN |
| Normal | 0.6 | 1.5 | Seimbang (default) |
| Longgar | 0.4 | 2.0 | Lebih sedikit hasil UNKNOWN |

Setelah mengubah nilai, restart aplikasi.

## API (untuk Developer)

```bash
curl -X POST http://localhost:5001/api/predict \
  -F "file=@gambar.jpg"
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

Contoh integrasi Python:

```python
import requests

def deteksi_label(path):
    with open(path, "rb") as f:
        response = requests.post(
            "http://localhost:5001/api/predict",
            files={"file": f},
        )
    return response.json()

hasil = deteksi_label("label.jpg")
print(hasil["prediction"], hasil["confidence"])
```

## FAQ

**Berapa data minimum?**
5 gambar per kategori. Untuk hasil stabil, gunakan 20+ per kategori.

**Kenapa hasilnya UNKNOWN?**
Gambar terlalu berbeda dari data training (mis. bukan label, sudut/ukuran tidak biasa, pencahayaan ekstrem). Tambahkan contoh serupa ke dataset lalu retrain.

**Apakah model harus di-retrain rutin?**
Tidak wajib. Retrain saat ada produk baru atau performa menurun.

**Di mana model disimpan?**
`models/defect_detection_model.pkl`. Backup file ini untuk menyimpan model.

**Bisakah dipakai untuk produk lain?**
Bisa, tetapi perlu dataset dan training ulang yang sesuai.

## Troubleshooting

| Masalah | Solusi |
|---|---|
| "Model tidak tersedia" | Jalankan `python retrain_model.py` |
| "Tidak ada file yang dipilih" | Pastikan file dipilih dan formatnya didukung |
| Upload gagal | Cek ukuran file (maks. 16 MB), lalu restart aplikasi |
| Semua hasil GOOD | Tambah variasi data defect, lalu retrain |
| Training gagal | Pastikan minimal 5 gambar per kategori dan file gambar valid |

## Tips

1. Mulai dengan 10–15 gambar per kategori.
2. Uji model dengan gambar baru setelah setiap training.
3. Catat perubahan dataset dan hasil akurasi.
4. Tambah data secara bertahap, jangan langsung ratusan gambar serupa.
5. Gunakan foto asli, bukan gambar sintetis, untuk hasil produksi.
