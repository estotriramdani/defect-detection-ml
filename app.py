"""
Flask Web Application untuk Deteksi Cacat Label Kemasan
Menggunakan model CNN TensorFlow untuk klasifikasi gambar
Author: Developer
Date: July 15, 2025
"""

import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import pickle
from PIL import Image
from datetime import datetime
import threading
import json

# Import model class
from real_model import SimpleDefectDetector

# Konfigurasi aplikasi Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TRAINING_DATA_FOLDER'] = 'static/training_data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables untuk training
training_status = {
    'status': 'idle',  # idle, running, completed, error
    'current_step': 0,
    'epoch': 0,
    'total_epochs': 0,
    'message': '',
    'results': {}
}

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Path model yang telah dilatih
MODEL_PATH = 'models/defect_detection_model.pkl'

# Memuat model saat aplikasi dimulai
try:
    model = SimpleDefectDetector.load(MODEL_PATH)
    print("Model berhasil dimuat")
except Exception as e:
    print(f"Error memuat model: {e}")
    model = None

def allowed_file(filename):
    """
    Memeriksa apakah file memiliki ekstensi yang diizinkan
    
    Args:
        filename (str): Nama file yang akan diperiksa
        
    Returns:
        bool: True jika ekstensi file diizinkan, False jika tidak
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocessing gambar untuk input model
    
    Args:
        image_path (str): Path ke file gambar
        target_size (tuple): Ukuran target untuk resize gambar
        
    Returns:
        numpy.ndarray: Array gambar yang telah diproses
    """
    try:
        # Membaca gambar menggunakan PIL
        image = Image.open(image_path)
        
        # Konversi ke RGB jika diperlukan
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize gambar ke ukuran yang sama dengan training (64x64)
        image = image.resize(target_size)
        
        # Konversi ke array numpy
        image_array = np.array(image)
        
        # Normalisasi pixel values ke range [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Tambahkan batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        print(f"Error dalam preprocessing gambar: {e}")
        return None

def predict_defect(image_path):
    """
    Melakukan prediksi cacat pada gambar menggunakan model
    
    Args:
        image_path (str): Path ke file gambar
        
    Returns:
        dict: Dictionary berisi hasil prediksi dengan format:
              {'prediction': str, 'confidence': float, 'status': str}
    """
    if model is None:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'status': 'Model tidak tersedia'
        }
    
    try:
        # Preprocessing gambar
        processed_image = preprocess_image(image_path)
        
        if processed_image is None:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'status': 'Gagal memproses gambar'
            }
        
        # Melakukan prediksi dengan anomaly detection
        prediction_result = model.predict(processed_image)
        
        # Check if model returns tuple (new format) or array (old format)
        if isinstance(prediction_result, tuple):
            prediction, anomaly_score = prediction_result
        else:
            prediction = prediction_result
            anomaly_score = 0.0  # Default for old models
        
        # Menginterpretasi hasil prediksi
        # Output model adalah [probability_good, probability_defect]
        confidence = float(np.max(prediction[0]))
        predicted_class = np.argmax(prediction[0])
        
        # Threshold untuk mendeteksi gambar yang tidak dikenal
        # Kombinasi confidence threshold dan anomaly score
        CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence
        ANOMALY_THRESHOLD = 1.5     # Maximum acceptable anomaly score
        
        if confidence < CONFIDENCE_THRESHOLD or anomaly_score > ANOMALY_THRESHOLD:
            predicted_label = 'unknown'
            status_message = f'Gambar tidak sesuai dengan pola training data (confidence: {confidence:.3f}, anomaly: {anomaly_score:.3f})'
        else:
            # Mapping class index ke label
            predicted_label = model.classes[predicted_class]
            status_message = f'success (anomaly score: {anomaly_score:.3f})'
        
        return {
            'prediction': predicted_label,
            'confidence': confidence,
            'status': status_message,
            'anomaly_score': anomaly_score
        }
    
    except Exception as e:
        print(f"Error dalam prediksi: {e}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'status': f'Error: {str(e)}'
        }

def create_upload_folder():
    """
    Membuat folder upload dan training jika belum ada
    """
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Membuat folder training data
    training_folder = app.config['TRAINING_DATA_FOLDER']
    if not os.path.exists(training_folder):
        os.makedirs(training_folder)
        os.makedirs(os.path.join(training_folder, 'good'))
        os.makedirs(os.path.join(training_folder, 'defect'))

def count_training_images():
    """
    Menghitung jumlah gambar training untuk setiap kategori
    
    Returns:
        dict: Dictionary berisi jumlah gambar good dan defect
    """
    good_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'good')
    defect_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'defect')
    
    good_count = len([f for f in os.listdir(good_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]) if os.path.exists(good_folder) else 0
    defect_count = len([f for f in os.listdir(defect_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]) if os.path.exists(defect_folder) else 0
    
    return {'good_count': good_count, 'defect_count': defect_count}

def get_training_images_list():
    """
    Mendapatkan list semua gambar training untuk setiap kategori
    
    Returns:
        dict: Dictionary berisi list gambar good dan defect dengan informasi file
    """
    good_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'good')
    defect_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'defect')
    
    def get_image_info(folder, category):
        images = []
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filepath = os.path.join(folder, filename)
                    stat = os.stat(filepath)
                    images.append({
                        'filename': filename,
                        'category': category,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                        'path': f'static/training_data/{category}/{filename}'
                    })
        return sorted(images, key=lambda x: x['created'], reverse=True)
    
    good_images = get_image_info(good_folder, 'good')
    defect_images = get_image_info(defect_folder, 'defect')
    
    return {
        'good_images': good_images,
        'defect_images': defect_images,
        'good_count': len(good_images),
        'defect_count': len(defect_images)
    }

@app.route('/')
def index():
    """
    Route untuk halaman utama aplikasi
    
    Returns:
        str: Rendered template halaman utama
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Route untuk mengunggah dan memproses file gambar
    
    Returns:
        str: Redirect ke halaman hasil atau halaman utama jika error
    """
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Membuat nama file yang aman
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        
        # Menyimpan file
        create_upload_folder()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Melakukan prediksi
        result = predict_defect(filepath)
        
        return render_template('result.html', 
                             filename=filename,
                             prediction=result['prediction'],
                             confidence=result['confidence'],
                             status=result['status'])
    else:
        flash('File tidak valid. Gunakan format: PNG, JPG, JPEG, GIF, atau BMP')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint untuk prediksi (untuk integrasi dengan aplikasi lain)
    
    Returns:
        json: Response dalam format JSON
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        
        create_upload_folder()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = predict_defect(filepath)
        
        return jsonify({
            'filename': filename,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'status': result['status'],
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Format file tidak didukung'}), 400

@app.errorhandler(413)
def too_large(e):
    """
    Handler untuk error file terlalu besar
    """
    flash('File terlalu besar. Maksimal 16MB')
    return redirect(url_for('index'))

@app.route('/training')
def training():
    """
    Route untuk halaman training model
    
    Returns:
        str: Rendered template halaman training
    """
    training_data = get_training_images_list()
    return render_template('training.html', 
                         good_count=training_data['good_count'],
                         defect_count=training_data['defect_count'],
                         good_images=training_data['good_images'],
                         defect_images=training_data['defect_images'])

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    """
    Route untuk upload gambar training data
    
    Returns:
        json: Response dengan status upload
    """
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'message': 'Tidak ada file yang dikirim'})
        
        files = request.files.getlist('files')
        category = request.form.get('category')  # 'good' atau 'defect'
        
        if category not in ['good', 'defect']:
            return jsonify({'success': False, 'message': 'Kategori tidak valid'})
        
        if not files or files[0].filename == '':
            return jsonify({'success': False, 'message': 'Tidak ada file yang dipilih'})
        
        # Membuat folder jika belum ada
        create_upload_folder()
        target_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], category)
        
        uploaded_count = 0
        for file in files:
            if file and allowed_file(file.filename):
                # Membuat nama file yang aman
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                filename = f"{timestamp}{uploaded_count}_{filename}"
                
                # Menyimpan file
                filepath = os.path.join(target_folder, filename)
                file.save(filepath)
                uploaded_count += 1
        
        # Hitung ulang gambar training
        counts = count_training_images()
        
        return jsonify({
            'success': True, 
            'message': f'{uploaded_count} file berhasil diupload',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/start_training', methods=['POST'])
def start_training():
    """
    Route untuk memulai training model
    
    Returns:
        json: Response dengan status training
    """
    global training_status
    
    try:
        # Cek apakah training sedang berjalan
        if training_status['status'] == 'running':
            return jsonify({'success': False, 'message': 'Training sedang berjalan'})
        
        # Cek jumlah data training
        counts = count_training_images()
        if counts['good_count'] < 5 or counts['defect_count'] < 5:
            return jsonify({'success': False, 'message': 'Minimal 5 gambar per kategori diperlukan'})
        
        # Reset status training
        training_status = {
            'status': 'running',
            'current_step': 0,
            'epoch': 0,
            'total_epochs': 10,
            'message': 'Memulai training...',
            'results': {}
        }
        
        # Mulai training di thread terpisah
        training_thread = threading.Thread(target=train_model_background)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({'success': True, 'message': 'Training dimulai'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/training_status')
def get_training_status():
    """
    Route untuk mendapatkan status training
    
    Returns:
        json: Status training saat ini
    """
    return jsonify(training_status)

@app.route('/get_training_data')
def get_training_data():
    """
    Route untuk mendapatkan list training data dalam format JSON
    
    Returns:
        json: List gambar training untuk kedua kategori
    """
    try:
        training_data = get_training_images_list()
        return jsonify({
            'success': True,
            'data': training_data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })

@app.route('/delete_training_image', methods=['POST'])
def delete_training_image():
    """
    Route untuk menghapus gambar training tertentu
    
    Returns:
        json: Response dengan status penghapusan
    """
    try:
        data = request.get_json()
        category = data.get('category')
        filename = data.get('filename')
        
        if category not in ['good', 'defect']:
            return jsonify({'success': False, 'message': 'Kategori tidak valid'})
        
        if not filename:
            return jsonify({'success': False, 'message': 'Nama file tidak valid'})
        
        # Konstruksi path file
        file_path = os.path.join(app.config['TRAINING_DATA_FOLDER'], category, filename)
        
        # Cek apakah file ada
        if not os.path.exists(file_path):
            return jsonify({'success': False, 'message': 'File tidak ditemukan'})
        
        # Hapus file
        os.remove(file_path)
        
        # Update counter
        counts = count_training_images()
        
        return jsonify({
            'success': True,
            'message': 'Gambar berhasil dihapus',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@app.route('/clear_training_data', methods=['POST'])
def clear_training_data():
    """
    Route untuk menghapus semua data training
    
    Returns:
        json: Response dengan status penghapusan
    """
    try:
        data = request.get_json()
        category = data.get('category')  # 'good', 'defect', atau 'all'
        
        if category not in ['good', 'defect', 'all']:
            return jsonify({'success': False, 'message': 'Kategori tidak valid'})
        
        deleted_count = 0
        
        if category == 'all' or category == 'good':
            good_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'good')
            if os.path.exists(good_folder):
                for filename in os.listdir(good_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        os.remove(os.path.join(good_folder, filename))
                        deleted_count += 1
        
        if category == 'all' or category == 'defect':
            defect_folder = os.path.join(app.config['TRAINING_DATA_FOLDER'], 'defect')
            if os.path.exists(defect_folder):
                for filename in os.listdir(defect_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        os.remove(os.path.join(defect_folder, filename))
                        deleted_count += 1
        
        # Update counter
        counts = count_training_images()
        
        return jsonify({
            'success': True,
            'message': f'{deleted_count} gambar berhasil dihapus',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

def train_model_background():
    """
    Fungsi untuk training model di background
    """
    global training_status
    
    try:
        # Step 1: Load dataset
        training_status['current_step'] = 0
        training_status['message'] = 'Memuat dataset...'
        
        import time
        from real_model import SimpleDefectDetector
        
        time.sleep(1)
        
        # Step 2: Preprocessing
        training_status['current_step'] = 1
        training_status['message'] = 'Preprocessing data...'
        time.sleep(1)
        
        # Step 3: Training
        training_status['current_step'] = 2
        training_status['message'] = 'Training model...'
        
        # Training model sesungguhnya
        detector = SimpleDefectDetector()
        results = detector.train(app.config['TRAINING_DATA_FOLDER'])
        
        # Step 4: Saving model
        training_status['current_step'] = 3
        training_status['message'] = 'Menyimpan model...'
        time.sleep(1)
        
        # Simpan model
        detector.save(MODEL_PATH)
        
        # Update global model
        global model
        model = detector
        
        # Training selesai
        training_status['status'] = 'completed'
        training_status['current_step'] = 4
        training_status['message'] = 'Training selesai!'
        training_status['results'] = results
        
    except Exception as e:
        training_status['status'] = 'error'
        training_status['message'] = f'Error during training: {str(e)}'

def create_trained_model():
    """
    Membuat model yang sudah 'dilatih' dengan data training
    """
    global model
    try:
        counts = count_training_images()
        
        # Buat model baru
        new_model = SimpleDefectDetector()
        new_model.training_data_count = counts
        new_model.trained_at = datetime.now()
        
        # Simpan model
        model_path = "models/defect_detection_model.pkl"
        new_model.save(model_path)
        
        # Update global model
        model = new_model
        
        print(f"Model berhasil diperbarui dengan {counts['good_count']} gambar good dan {counts['defect_count']} gambar defect")
        
    except Exception as e:
        print(f"Error creating trained model: {e}")

@app.errorhandler(413)
def too_large(e):
    """
    Handler untuk error file terlalu besar
    """
    flash('File terlalu besar. Maksimal 16MB')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Membuat folder yang diperlukan
    create_upload_folder()
    
    # Menjalankan aplikasi
    app.run(debug=True, host='0.0.0.0', port=5001)
