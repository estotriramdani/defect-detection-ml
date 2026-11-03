"""
State model dan logika prediksi.

Model disimpan sebagai singleton agar dapat diganti saat training selesai
tanpa restart aplikasi.
"""

import os
import threading

import numpy as np
from PIL import Image

from real_model import SimpleDefectDetector

_model = None
_model_lock = threading.RLock()


def load_model(model_path):
    """Muat model dari disk. Mengembalikan None jika belum tersedia/invalid."""
    global _model

    if not os.path.exists(model_path):
        print(f"Model belum tersedia: {model_path}")
        with _model_lock:
            _model = None
        return None

    detector = SimpleDefectDetector.load(model_path)
    if not getattr(detector, 'trained', False):
        print(f"Model tidak valid atau belum dilatih: {model_path}")
        detector = None

    with _model_lock:
        _model = detector
    return detector


def get_model():
    with _model_lock:
        return _model


def set_model(detector):
    global _model
    with _model_lock:
        _model = detector


def preprocess_image(image_path, target_size=(64, 64)):
    """
    Preprocessing gambar untuk input model.

    Args:
        image_path (str): Path ke file gambar
        target_size (tuple): Ukuran target untuk resize gambar

    Returns:
        numpy.ndarray | None: Array gambar (1, H, W, 3) atau None jika gagal
    """
    try:
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size)
        image_array = np.array(image).astype('float32') / 255.0
        return np.expand_dims(image_array, axis=0)

    except Exception as exc:
        print(f"Error dalam preprocessing gambar: {exc}")
        return None


def predict_defect(image_path, confidence_threshold, anomaly_threshold):
    """
    Prediksi cacat pada gambar.

    Returns:
        dict: {'prediction', 'confidence', 'anomaly_score', 'status'}
    """
    model = get_model()
    if model is None:
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'status': 'Model tidak tersedia',
        }

    try:
        processed_image = preprocess_image(image_path)
        if processed_image is None:
            return {
                'prediction': 'Error',
                'confidence': 0.0,
                'anomaly_score': 0.0,
                'status': 'Gagal memproses gambar',
            }

        prediction_result = model.predict(processed_image)

        # Kompatibilitas dengan model lama yang hanya mengembalikan probabilitas.
        if isinstance(prediction_result, tuple):
            prediction, anomaly_score = prediction_result
        else:
            prediction = prediction_result
            anomaly_score = 0.0

        confidence = float(np.max(prediction[0]))
        predicted_class = int(np.argmax(prediction[0]))

        if confidence < confidence_threshold or anomaly_score > anomaly_threshold:
            predicted_label = 'unknown'
            status_message = (
                'Gambar tidak sesuai dengan pola training data '
                f'(confidence: {confidence:.3f}, anomaly: {anomaly_score:.3f})'
            )
        else:
            predicted_label = model.classes[predicted_class]
            status_message = f'success (anomaly score: {anomaly_score:.3f})'

        return {
            'prediction': predicted_label,
            'confidence': confidence,
            'anomaly_score': float(anomaly_score),
            'status': status_message,
        }

    except Exception as exc:
        print(f"Error dalam prediksi: {exc}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'anomaly_score': 0.0,
            'status': f'Error: {exc}',
        }
