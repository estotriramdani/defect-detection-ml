"""
Utility functions untuk aplikasi deteksi cacat label
Berisi fungsi-fungsi helper yang dapat digunakan di berbagai bagian aplikasi
"""

import os
import uuid
from datetime import datetime
from PIL import Image, ImageOps
import numpy as np
import cv2

def generate_unique_filename(original_filename):
    """
    Membuat nama file yang unik dengan timestamp dan UUID
    
    Args:
        original_filename (str): Nama file asli
        
    Returns:
        str: Nama file yang unik
    """
    # Ambil ekstensi file
    file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else ''
    
    # Buat timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Buat UUID pendek
    unique_id = str(uuid.uuid4())[:8]
    
    # Gabungkan untuk membuat nama file unik
    unique_filename = f"{timestamp}_{unique_id}.{file_extension}"
    
    return unique_filename

def validate_image_file(file):
    """
    Validasi file gambar yang diupload
    
    Args:
        file: File object dari request
        
    Returns:
        dict: Dictionary berisi status validasi dan pesan error jika ada
    """
    if not file:
        return {'valid': False, 'error': 'Tidak ada file yang dipilih'}
    
    if file.filename == '':
        return {'valid': False, 'error': 'Nama file kosong'}
    
    # Cek ekstensi file
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return {'valid': False, 'error': 'Format file tidak didukung'}
    
    # Cek ukuran file (max 16MB)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset pointer
    
    if file_size > 16 * 1024 * 1024:
        return {'valid': False, 'error': 'File terlalu besar (maksimal 16MB)'}
    
    try:
        # Cek apakah file benar-benar gambar
        Image.open(file).verify()
        file.seek(0)  # Reset pointer setelah verify
        return {'valid': True, 'error': None}
    except Exception:
        return {'valid': False, 'error': 'File bukan gambar yang valid'}

def preprocess_image_advanced(image_path, target_size=(224, 224)):
    """
    Preprocessing gambar tingkat lanjut dengan berbagai teknik
    
    Args:
        image_path (str): Path ke file gambar
        target_size (tuple): Ukuran target untuk resize
        
    Returns:
        numpy.ndarray: Array gambar yang telah diproses
    """
    try:
        # Baca gambar menggunakan PIL
        image = Image.open(image_path)
        
        # Konversi ke RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Auto-orient berdasarkan EXIF data
        image = ImageOps.exif_transpose(image)
        
        # Resize dengan mempertahankan aspect ratio
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Buat background putih dan paste gambar di tengah
        background = Image.new('RGB', target_size, (255, 255, 255))
        
        # Hitung posisi untuk center image
        x = (target_size[0] - image.width) // 2
        y = (target_size[1] - image.height) // 2
        
        background.paste(image, (x, y))
        
        # Konversi ke numpy array
        image_array = np.array(background)
        
        # Normalisasi
        image_array = image_array.astype('float32') / 255.0
        
        # Tambahkan batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error dalam preprocessing: {e}")
        return None

def enhance_image_quality(image_path):
    """
    Meningkatkan kualitas gambar sebelum prediksi
    
    Args:
        image_path (str): Path ke file gambar
        
    Returns:
        str: Path ke gambar yang telah ditingkatkan kualitasnya
    """
    try:
        # Baca gambar dengan OpenCV
        image = cv2.imread(image_path)
        
        # Konversi BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Gaussian blur untuk mengurangi noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Tingkatkan kontras menggunakan CLAHE
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Simpan gambar yang telah ditingkatkan
        enhanced_path = image_path.replace('.', '_enhanced.')
        cv2.imwrite(enhanced_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        return enhanced_path
        
    except Exception as e:
        print(f"Error dalam enhancement: {e}")
        return image_path

def calculate_prediction_confidence(prediction_array):
    """
    Menghitung tingkat kepercayaan prediksi dengan lebih detail
    
    Args:
        prediction_array (numpy.ndarray): Array hasil prediksi model
        
    Returns:
        dict: Dictionary berisi informasi confidence yang detail
    """
    try:
        # Ambil probabilitas untuk setiap kelas
        probabilities = prediction_array[0]
        
        # Hitung confidence sebagai probabilitas maksimum
        max_confidence = float(np.max(probabilities))
        
        # Hitung entropy untuk mengukur uncertainty
        # Entropy rendah = confidence tinggi
        epsilon = 1e-10  # Untuk menghindari log(0)
        entropy = -np.sum(probabilities * np.log(probabilities + epsilon))
        
        # Normalisasi entropy (untuk 2 kelas, max entropy = ln(2))
        normalized_entropy = entropy / np.log(2)
        
        # Hitung margin (selisih antara top-2 predictions)
        sorted_probs = np.sort(probabilities)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])
        
        return {
            'confidence': max_confidence,
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'margin': margin,
            'all_probabilities': probabilities.tolist()
        }
        
    except Exception as e:
        print(f"Error dalam calculate confidence: {e}")
        return {
            'confidence': 0.0,
            'entropy': 1.0,
            'normalized_entropy': 1.0,
            'margin': 0.0,
            'all_probabilities': [0.5, 0.5]
        }

def create_prediction_report(filename, prediction_result, confidence_details):
    """
    Membuat laporan detail dari hasil prediksi
    
    Args:
        filename (str): Nama file gambar
        prediction_result (str): Hasil prediksi ('good' atau 'defect')
        confidence_details (dict): Detail confidence dari calculate_prediction_confidence
        
    Returns:
        dict: Laporan lengkap hasil prediksi
    """
    timestamp = datetime.now().isoformat()
    
    # Tentukan level confidence berdasarkan nilai
    confidence_level = "Rendah"
    if confidence_details['confidence'] >= 0.9:
        confidence_level = "Sangat Tinggi"
    elif confidence_details['confidence'] >= 0.8:
        confidence_level = "Tinggi"
    elif confidence_details['confidence'] >= 0.7:
        confidence_level = "Sedang"
    
    # Rekomendasi berdasarkan hasil
    recommendation = ""
    if prediction_result == 'good' and confidence_details['confidence'] >= 0.8:
        recommendation = "Produk dapat melanjutkan ke tahap berikutnya"
    elif prediction_result == 'defect' and confidence_details['confidence'] >= 0.8:
        recommendation = "Produk perlu inspeksi manual dan tindakan korektif"
    else:
        recommendation = "Disarankan untuk melakukan verifikasi manual"
    
    report = {
        'filename': filename,
        'timestamp': timestamp,
        'prediction': prediction_result,
        'confidence': confidence_details['confidence'],
        'confidence_level': confidence_level,
        'entropy': confidence_details['entropy'],
        'margin': confidence_details['margin'],
        'recommendation': recommendation,
        'technical_details': {
            'all_probabilities': confidence_details['all_probabilities'],
            'normalized_entropy': confidence_details['normalized_entropy']
        }
    }
    
    return report

def cleanup_old_uploads(upload_folder, max_age_hours=24):
    """
    Membersihkan file upload yang sudah lama
    
    Args:
        upload_folder (str): Path ke folder upload
        max_age_hours (int): Maksimal umur file dalam jam
    """
    try:
        current_time = datetime.now()
        
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            
            # Cek apakah ini file (bukan folder)
            if os.path.isfile(file_path):
                # Ambil waktu modifikasi file
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                # Hitung selisih waktu
                time_diff = current_time - file_time
                
                # Hapus file jika sudah terlalu lama
                if time_diff.total_seconds() > max_age_hours * 3600:
                    os.remove(file_path)
                    print(f"Menghapus file lama: {filename}")
                    
    except Exception as e:
        print(f"Error dalam cleanup: {e}")
