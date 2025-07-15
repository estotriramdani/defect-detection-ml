"""
Model dummy sederhana untuk demonstrasi aplikasi deteksi cacat
Menggunakan random prediction untuk testing tanpa TensorFlow
"""

import numpy as np
import os
import pickle
from datetime import datetime

class DummyModel:
    """
    Model dummy untuk demonstrasi aplikasi deteksi cacat
    Menghasilkan prediksi random untuk testing
    """
    
    def __init__(self):
        self.classes = ['good', 'defect']
        self.model_name = "Dummy CNN Model"
        self.created_at = datetime.now()
    
    def predict(self, image_array):
        """
        Simulasi prediksi model CNN
        
        Args:
            image_array: Array gambar (tidak digunakan dalam dummy model)
            
        Returns:
            numpy.ndarray: Array probabilitas untuk setiap kelas
        """
        # Simulasi prediksi dengan probabilitas random
        # Bias sedikit ke arah 'good' untuk demonstrasi yang lebih realistis
        good_prob = np.random.uniform(0.3, 0.9)
        defect_prob = 1.0 - good_prob
        
        return np.array([[good_prob, defect_prob]])
    
    def save(self, filepath):
        """
        Simpan model dummy ke file
        
        Args:
            filepath (str): Path untuk menyimpan model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'classes': self.classes,
            'model_name': self.model_name,
            'created_at': self.created_at,
            'type': 'dummy'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Dummy model berhasil disimpan ke: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model dummy dari file
        
        Args:
            filepath (str): Path ke file model
            
        Returns:
            DummyModel: Instance model dummy
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            model = cls()
            model.classes = model_data.get('classes', ['good', 'defect'])
            model.model_name = model_data.get('model_name', 'Dummy CNN Model')
            model.created_at = model_data.get('created_at', datetime.now())
            
            print(f"Dummy model berhasil dimuat dari: {filepath}")
            return model
            
        except Exception as e:
            print(f"Error loading dummy model: {e}")
            return cls()  # Return new dummy model if loading fails

def create_dummy_model():
    """
    Membuat dan menyimpan dummy model
    """
    print("=== Membuat Dummy Model untuk Deteksi Cacat ===")
    
    model = DummyModel()
    model_path = "models/defect_detection_model.pkl"
    
    print(f"Model: {model.model_name}")
    print(f"Classes: {model.classes}")
    print(f"Created: {model.created_at}")
    
    # Test prediksi
    print("\nTesting prediksi dummy:")
    dummy_image = np.random.random((1, 224, 224, 3))
    prediction = model.predict(dummy_image)
    
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    predicted_label = model.classes[predicted_class]
    
    print(f"  - Prediksi: {predicted_label}")
    print(f"  - Confidence: {confidence:.4f}")
    print(f"  - Raw output: {prediction[0]}")
    
    # Simpan model
    model.save(model_path)
    
    print(f"\n✅ Dummy model berhasil dibuat dan disimpan!")
    print(f"📁 Path: {os.path.abspath(model_path)}")
    print("\n⚠️  CATATAN:")
    print("   - Ini adalah model dummy untuk demonstrasi")
    print("   - Model menghasilkan prediksi random")
    print("   - Untuk produksi, gunakan model CNN yang sudah dilatih")

if __name__ == "__main__":
    create_dummy_model()
