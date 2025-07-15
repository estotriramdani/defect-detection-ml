"""
Script untuk test prediksi langsung dengan model
"""

from real_model import SimpleDefectDetector
import numpy as np

def test_direct_prediction():
    print("🔍 Direct Prediction Test")
    
    # Load model
    model = SimpleDefectDetector.load("models/defect_detection_model.pkl")
    
    # Test dengan base image
    base_image_path = "static/base-data/label_fibemini.png"
    
    # Extract features langsung
    features = model.extract_features(base_image_path)
    print(f"Base image features: {features}")
    
    # Predict langsung dengan features
    features_reshaped = features.reshape(1, -1)
    proba = model.model.predict_proba(features_reshaped)[0]
    predicted_class = np.argmax(proba)
    
    print(f"Direct prediction probabilities: {proba}")
    print(f"Predicted class: {model.classes[predicted_class]}")
    print(f"Confidence: {np.max(proba):.4f}")
    
    # Test dengan sample good dari training
    import os
    good_folder = "static/training_data/good"
    good_files = [f for f in os.listdir(good_folder) if f.endswith('.png')]
    
    if good_files:
        # Test beberapa file good
        for i, good_file in enumerate(good_files[:3]):  # Test 3 file pertama
            good_path = os.path.join(good_folder, good_file)
            good_features = model.extract_features(good_path)
            good_features_reshaped = good_features.reshape(1, -1)
            good_proba = model.model.predict_proba(good_features_reshaped)[0]
            good_predicted = np.argmax(good_proba)
            
            print(f"\nGood sample {i+1} ({good_file}):")
            print(f"  Features: {good_features[:5]}...")  # First 5 features
            print(f"  Probabilities: {good_proba}")
            print(f"  Predicted: {model.classes[good_predicted]}")
    
    # Test dengan sample defect dari training
    defect_folder = "static/training_data/defect"
    defect_files = [f for f in os.listdir(defect_folder) if f.endswith('.png')]
    
    if defect_files:
        # Test beberapa file defect
        for i, defect_file in enumerate(defect_files[:3]):  # Test 3 file pertama
            defect_path = os.path.join(defect_folder, defect_file)
            defect_features = model.extract_features(defect_path)
            defect_features_reshaped = defect_features.reshape(1, -1)
            defect_proba = model.model.predict_proba(defect_features_reshaped)[0]
            defect_predicted = np.argmax(defect_proba)
            
            print(f"\nDefect sample {i+1} ({defect_file}):")
            print(f"  Features: {defect_features[:5]}...")  # First 5 features
            print(f"  Probabilities: {defect_proba}")
            print(f"  Predicted: {model.classes[defect_predicted]}")

if __name__ == "__main__":
    test_direct_prediction()
