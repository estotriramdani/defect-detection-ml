"""
Script untuk membuat model CNN sederhana untuk deteksi cacat label
Ini adalah contoh implementasi model yang dapat Anda latih dengan data Anda sendiri
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def create_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Membuat arsitektur model CNN untuk klasifikasi binary (good/defect)
    
    Args:
        input_shape (tuple): Ukuran input gambar (height, width, channels)
        num_classes (int): Jumlah kelas (2 untuk binary classification)
        
    Returns:
        tensorflow.keras.Model: Model CNN yang belum dilatih
    """
    
    model = models.Sequential([
        # Convolutional Base
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def compile_model(model):
    """
    Mengkompilasi model dengan optimizer, loss function, dan metrics
    
    Args:
        model: Model Keras yang akan dikompilasi
    """
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_dummy_model():
    """
    Membuat model dummy untuk testing (tanpa training data)
    Model ini akan memberikan prediksi random untuk keperluan demonstrasi
    """
    
    # Membuat model
    model = create_cnn_model()
    model = compile_model(model)
    
    # Menginisialisasi weights dengan random values
    dummy_input = np.random.random((1, 224, 224, 3))
    _ = model.predict(dummy_input)
    
    print("Model Summary:")
    model.summary()
    
    return model

def save_model(model, model_path):
    """
    Menyimpan model ke file
    
    Args:
        model: Model Keras yang akan disimpan
        model_path (str): Path untuk menyimpan model
    """
    model.save(model_path)
    print(f"Model berhasil disimpan ke: {model_path}")

if __name__ == "__main__":
    # Membuat folder models jika belum ada
    os.makedirs("models", exist_ok=True)
    
    # Membuat dan menyimpan dummy model
    print("Membuat model CNN untuk deteksi cacat...")
    model = create_dummy_model()
    
    # Menyimpan model
    model_path = "models/defect_detection_model.h5"
    save_model(model, model_path)
    
    print("\nCATATAN:")
    print("Ini adalah model dummy untuk demonstrasi.")
    print("Untuk aplikasi production, Anda perlu:")
    print("1. Mengumpulkan dataset gambar label 'good' dan 'defect'")
    print("2. Melakukan preprocessing data")
    print("3. Melatih model dengan data training yang sesuai")
    print("4. Melakukan evaluasi dan validasi model")
    print("5. Menyimpan model yang telah dilatih")
