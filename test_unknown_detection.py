#!/usr/bin/env python3
"""
Script untuk menguji deteksi "unknown" pada gambar yang tidak sesuai training data
"""

import os
import sys
sys.path.append('.')

from app import predict_defect
from real_model import SimpleDefectDetector

def test_unknown_detection():
    print("🧪 Testing Unknown Detection...")
    
    # Test images
    test_images = [
        "static/base-data/sure-defect.png",
        "static/base-data/label_fibemini.png"
    ]
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            continue
            
        print(f"\n📸 Testing: {image_path}")
        
        # Predict using the app function
        result = predict_defect(image_path)
        
        print(f"   🔍 Prediction: {result['prediction']}")
        print(f"   📊 Confidence: {result['confidence']:.3f}")
        if 'anomaly_score' in result:
            print(f"   ⚠️  Anomaly Score: {result['anomaly_score']:.3f}")
        print(f"   📝 Status: {result['status']}")
        
        # Expected behavior for sure-defect.png: should be "unknown"
        if "sure-defect" in image_path:
            if result['prediction'] == 'unknown':
                print("   ✅ PASS: Correctly detected as unknown!")
            else:
                print("   ❌ FAIL: Should be detected as unknown!")
        
        print("-" * 50)

if __name__ == "__main__":
    test_unknown_detection()
