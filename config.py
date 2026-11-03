"""
Konfigurasi aplikasi.

Semua nilai dapat diubah melalui environment variable (atau file `.env`).
Lihat `.env.example` untuk daftar lengkapnya.
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Muat .env jika python-dotenv tersedia (opsional).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(BASE_DIR, '.env'))
except ImportError:  # pragma: no cover - dotenv bersifat opsional
    pass


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class Config:
    """Konfigurasi dasar aplikasi."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = _env_bool('FLASK_DEBUG', False)
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = _env_int('PORT', 5001)

    # Path (absolut agar tidak bergantung pada working directory)
    MODEL_PATH = os.environ.get('MODEL_PATH') or os.path.join(
        BASE_DIR, 'models', 'defect_detection_model.pkl'
    )
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or os.path.join(
        BASE_DIR, 'static', 'uploads'
    )
    TRAINING_DATA_FOLDER = os.environ.get('TRAINING_DATA_FOLDER') or os.path.join(
        BASE_DIR, 'static', 'training_data'
    )

    # Upload
    MAX_CONTENT_LENGTH = _env_int('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)  # 16MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

    # Ambang deteksi unknown
    CONFIDENCE_THRESHOLD = _env_float('CONFIDENCE_THRESHOLD', 0.6)
    ANOMALY_THRESHOLD = _env_float('ANOMALY_THRESHOLD', 1.5)

    # Training
    TRAIN_MIN_PER_CLASS = _env_int('TRAIN_MIN_PER_CLASS', 5)

    # Autentikasi endpoint admin (training & manajemen dataset).
    # Jika ADMIN_PASSWORD kosong, autentikasi dinonaktifkan (mode development).
    ADMIN_USERNAME = os.environ.get('ADMIN_USERNAME', 'admin')
    ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')
