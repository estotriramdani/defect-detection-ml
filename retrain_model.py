#!/usr/bin/env python3
"""
Script untuk melatih ulang model dengan fitur anomaly detection
"""

import os
import sys

from real_model import SimpleDefectDetector

# Pastikan output UTF-8 agar emoji tidak error di konsol Windows (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def retrain_model():
    print("🚀 Memulai pelatihan ulang model dengan fitur anomaly detection...")
    
    # Create model instance
    detector = SimpleDefectDetector()
    
    # Check if training data exists
    training_folder = "static/training_data"
    if not os.path.exists(training_folder):
        print("❌ Folder training data tidak ditemukan!")
        return
    
    good_folder = os.path.join(training_folder, 'good')
    defect_folder = os.path.join(training_folder, 'defect')
    
    if not os.path.exists(good_folder) or not os.path.exists(defect_folder):
        print("❌ Folder 'good' atau 'defect' tidak ditemukan!")
        return
    
    good_count = len([f for f in os.listdir(good_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    defect_count = len([f for f in os.listdir(defect_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
    
    print(f"📊 Data training tersedia:")
    print(f"   - Good: {good_count} gambar")
    print(f"   - Defect: {defect_count} gambar")
    
    if good_count == 0 or defect_count == 0:
        print("❌ Diperlukan data training untuk kedua kategori (good dan defect)!")
        return
    
    try:
        # Train the model
        results = detector.train(training_folder)
        
        # Save the model
        model_path = "models/defect_detection_model.pkl"
        detector.save(model_path)
        
        print("\n✅ Model berhasil dilatih ulang!")
        print(f"   - Training accuracy: {results['train_accuracy']:.3f}")
        print(f"   - Validation accuracy: {results['val_accuracy']:.3f}")
        print(f"   - Model disimpan di: {model_path}")
        print("\n🔄 Restart aplikasi untuk menggunakan model yang baru.")
        
    except Exception as e:
        print(f"❌ Error saat training: {e}")

if __name__ == "__main__":
    retrain_model()
