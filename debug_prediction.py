"""
Script debug untuk testing prediksi model
"""

import sys
import os
sys.path.append('.')

from real_model import SimpleDefectDetector
from PIL import Image
import numpy as np

def test_prediction_debug():
    print("🔍 Debug Mode: Testing Model Prediction")
    
    # Load model
    try:
        model = SimpleDefectDetector.load("models/defect_detection_model.pkl")
        print("✅ Model loaded successfully")
        print(f"   - Trained: {model.trained}")
        print(f"   - Training data: {model.training_data_count}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test dengan base image asli
    base_image_path = "static/base-data/label_fibemini.png"
    print(f"\n📸 Testing with base image: {base_image_path}")
    
    try:
        # Load dan preprocessing seperti di app.py
        image = Image.open(base_image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke ukuran yang dibutuhkan model
        image = image.resize((224, 224))
        
        # Konversi ke array numpy
        image_array = np.array(image)
        
        # Normalisasi pixel values ke range [0, 1]
        image_array = image_array.astype('float32') / 255.0
        
        # Tambahkan batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        print(f"   - Image shape: {image_array.shape}")
        print(f"   - Image range: [{image_array.min():.3f}, {image_array.max():.3f}]")
        
        # Prediksi
        prediction = model.predict(image_array)
        
        print(f"\n🎯 Prediction Results:")
        print(f"   - Raw output: {prediction[0]}")
        print(f"   - Probabilities: [good={prediction[0][0]:.4f}, defect={prediction[0][1]:.4f}]")
        
        # Interpretasi hasil
        predicted_class = np.argmax(prediction[0])
        confidence = float(np.max(prediction[0]))
        predicted_label = model.classes[predicted_class]
        
        print(f"   - Predicted class: {predicted_label}")
        print(f"   - Confidence: {confidence:.4f}")
        
        if predicted_label == "good":
            print("✅ CORRECT: Base image predicted as GOOD")
        else:
            print("❌ ERROR: Base image predicted as DEFECT!")
            print("   This indicates a problem with the model or feature extraction")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

def test_feature_extraction():
    print("\n🔧 Debug Mode: Testing Feature Extraction")
    
    model = SimpleDefectDetector()
    base_image_path = "static/base-data/label_fibemini.png"
    
    try:
        # Extract features from base image
        features = model.extract_features(base_image_path)
        print(f"✅ Features extracted successfully")
        print(f"   - Feature count: {len(features)}")
        print(f"   - Feature values: {features}")
        print(f"   - Feature range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Test dengan gambar good dari training
        good_sample_path = "static/training_data/good"
        good_files = [f for f in os.listdir(good_sample_path) if f.endswith('.png')]
        
        if good_files:
            good_file = os.path.join(good_sample_path, good_files[0])
            good_features = model.extract_features(good_file)
            print(f"\n📊 Comparison with training good sample:")
            print(f"   - Base image features: {features[:5]}...")  # First 5 features
            print(f"   - Good sample features: {good_features[:5]}...")
            
            # Calculate similarity
            diff = np.abs(features - good_features)
            similarity = 1 - (np.mean(diff) / np.mean(np.abs(features)))
            print(f"   - Feature similarity: {similarity:.4f}")
            
    except Exception as e:
        print(f"❌ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prediction_debug()
    test_feature_extraction()
