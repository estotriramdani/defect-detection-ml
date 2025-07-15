"""
Debug prediksi dengan format yang sama persis seperti di app.py
"""

import os
import numpy as np
from PIL import Image
from real_model import SimpleDefectDetector

def preprocess_image_like_app(image_path, target_size=(224, 224)):
    """
    Preprocessing gambar persis seperti di app.py
    """
    try:
        # Membaca gambar menggunakan PIL
        image = Image.open(image_path)
        
        # Konversi ke RGB jika diperlukan
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize gambar ke ukuran yang dibutuhkan model
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

def debug_prediction_app_style():
    print("🔍 Debug Prediction dengan App.py Style")
    
    # Load model
    model = SimpleDefectDetector.load("models/defect_detection_model.pkl")
    
    # Test dengan base image
    image_path = "static/base-data/label_fibemini.png"
    
    print(f"📸 Testing: {image_path}")
    
    # Preprocessing seperti di app.py
    processed_image = preprocess_image_like_app(image_path)
    
    if processed_image is not None:
        print(f"   - Processed shape: {processed_image.shape}")
        print(f"   - Processed range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # Prediksi dengan model
        prediction = model.predict(processed_image)
        
        print(f"   - Raw prediction: {prediction[0]}")
        
        # Interpretasi hasil seperti di app.py
        confidence = float(np.max(prediction[0]))
        predicted_class = np.argmax(prediction[0])
        predicted_label = model.classes[predicted_class]
        
        print(f"   - Predicted class: {predicted_label}")
        print(f"   - Confidence: {confidence:.4f}")
        
        if predicted_label == "good":
            print("✅ CORRECT: Predicted as GOOD")
        else:
            print("❌ ERROR: Predicted as DEFECT")
            
        # Debug: Mari kita cek yang terjadi dalam predict function
        print(f"\n🔧 Debug internal predict process:")
        
        # Step 1: Convert normalized array back to 0-255 range
        if processed_image.max() <= 1.0:
            img_uint8 = (processed_image[0] * 255).astype(np.uint8)
        else:
            img_uint8 = processed_image[0].astype(np.uint8)
        
        print(f"   - Converted back shape: {img_uint8.shape}")
        print(f"   - Converted back range: [{img_uint8.min()}, {img_uint8.max()}]")
        
        # Step 2: Convert to PIL Image for feature extraction
        img = Image.fromarray(img_uint8)
        print(f"   - PIL Image size: {img.size}")
        
        # Step 3: Resize to 64x64 (like in updated predict function)
        img = img.resize((64, 64))
        print(f"   - After resize: {img.size}")
        
        # Step 4: Save temp and extract features
        temp_path = "debug_temp.png"
        img.save(temp_path)
        
        features = model.extract_features(temp_path)
        print(f"   - Features extracted: {features[:5]}...")
        
        # Step 5: Predict with features directly
        features_reshaped = features.reshape(1, -1)
        proba = model.model.predict_proba(features_reshaped)[0]
        direct_predicted = np.argmax(proba)
        
        print(f"   - Direct prediction: {proba}")
        print(f"   - Direct class: {model.classes[direct_predicted]}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    debug_prediction_app_style()
