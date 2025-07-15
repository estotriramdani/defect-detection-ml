"""
Final test dengan preprocessing yang sudah diperbaiki
"""

import os
import numpy as np
from PIL import Image
from real_model import SimpleDefectDetector

def preprocess_image_fixed(image_path, target_size=(64, 64)):
    """
    Preprocessing gambar yang sudah diperbaiki
    """
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image)
        image_array = image_array.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error: {e}")
        return None

def final_test():
    print("🎯 Final Test - Fixed Preprocessing")
    
    model = SimpleDefectDetector.load("models/defect_detection_model.pkl")
    
    # Test dengan base image
    image_path = "static/base-data/label_fibemini.png"
    processed_image = preprocess_image_fixed(image_path)
    
    if processed_image is not None:
        prediction = model.predict(processed_image)
        confidence = float(np.max(prediction[0]))
        predicted_class = np.argmax(prediction[0])
        predicted_label = model.classes[predicted_class]
        
        print(f"📸 Base Image Test:")
        print(f"   - Prediction: {predicted_label}")
        print(f"   - Confidence: {confidence:.4f}")
        print(f"   - Raw output: {prediction[0]}")
        
        if predicted_label == "good":
            print("✅ SUCCESS: Base image correctly predicted as GOOD!")
        else:
            print("❌ FAILED: Still predicting as DEFECT")
            
    # Test beberapa sample good dan defect
    print(f"\n📊 Testing Training Samples:")
    
    # Test good samples
    good_folder = "static/training_data/good"
    good_files = [f for f in os.listdir(good_folder) if f.endswith('.png')][:3]
    
    for good_file in good_files:
        good_path = os.path.join(good_folder, good_file)
        processed = preprocess_image_fixed(good_path)
        if processed is not None:
            pred = model.predict(processed)
            pred_class = model.classes[np.argmax(pred[0])]
            conf = float(np.max(pred[0]))
            print(f"   Good sample: {pred_class} ({conf:.3f})")
    
    # Test defect samples
    defect_folder = "static/training_data/defect"
    defect_files = [f for f in os.listdir(defect_folder) if f.endswith('.png')][:3]
    
    for defect_file in defect_files:
        defect_path = os.path.join(defect_folder, defect_file)
        processed = preprocess_image_fixed(defect_path)
        if processed is not None:
            pred = model.predict(processed)
            pred_class = model.classes[np.argmax(pred[0])]
            conf = float(np.max(pred[0]))
            print(f"   Defect sample: {pred_class} ({conf:.3f})")

if __name__ == "__main__":
    final_test()
