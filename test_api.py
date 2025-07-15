"""
Test prediksi melalui API Flask
"""

import requests
import json

def test_api_prediction():
    print("🌐 Testing Flask API Prediction")
    
    # Test dengan base image
    files = {'file': open('static/base-data/label_fibemini.png', 'rb')}
    
    try:
        response = requests.post('http://localhost:5000/api/predict', files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response successful:")
            print(f"   - Filename: {result['filename']}")
            print(f"   - Prediction: {result['prediction']}")
            print(f"   - Confidence: {result['confidence']:.4f}")
            print(f"   - Status: {result['status']}")
            
            if result['prediction'] == 'good':
                print("✅ CORRECT: Base image predicted as GOOD via API")
            else:
                print("❌ ERROR: Base image predicted as DEFECT via API")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    finally:
        files['file'].close()

if __name__ == "__main__":
    test_api_prediction()
