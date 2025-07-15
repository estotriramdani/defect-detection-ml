# File untuk menyimpan konfigurasi aplikasi
# Configuration settings untuk Flask app

import os
from datetime import timedelta

class Config:
    """
    Konfigurasi dasar untuk aplikasi Flask
    """
    # Secret key untuk session dan CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Konfigurasi upload file
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Ekstensi file yang diizinkan
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # Konfigurasi model
    MODEL_PATH = 'models/defect_detection_model.h5'
    
    # Konfigurasi session
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)
    
    # Konfigurasi logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'

class DevelopmentConfig(Config):
    """
    Konfigurasi untuk environment development
    """
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """
    Konfigurasi untuk environment production
    """
    DEBUG = False
    TESTING = False
    
    # Secret key harus diset melalui environment variable
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable harus diset untuk production")

class TestingConfig(Config):
    """
    Konfigurasi untuk testing
    """
    DEBUG = True
    TESTING = True
    WTF_CSRF_ENABLED = False

# Dictionary untuk memudahkan pemilihan konfigurasi
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
