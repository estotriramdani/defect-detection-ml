"""
Model sederhana menggunakan scikit-learn untuk deteksi cacat
Menggunakan fitur ekstraksi dasar dari gambar
"""

import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from datetime import datetime

class SimpleDefectDetector:
    # Jumlah fitur yang diekstraksi: 3 mean RGB + 3 std RGB + brightness +
    # contrast + 2 edge density + 5 bin histogram.
    N_FEATURES = 15

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classes = ['good', 'defect']
        self.trained = False
        self.training_data_count = {'good_count': 0, 'defect_count': 0}
        self.trained_at = None
        # Store training features for anomaly detection
        self.training_features = None
        self.feature_stats = None
        # Precomputed values untuk anomaly scoring (diisi saat training/load)
        self.anomaly_reference = None
        self._normalized_training = None

    def _extract_features_from_image(self, image):
        """
        Ekstraksi fitur dari objek PIL.Image (tanpa file sementara).

        Args:
            image (PIL.Image.Image): Gambar yang akan diekstraksi

        Returns:
            np.array: Array fitur
        """
        img = image.convert('RGB').resize((64, 64))
        img_array = np.array(img)

        features = []

        # 1. Mean RGB values
        features.extend([
            np.mean(img_array[:, :, 0]),  # Red mean
            np.mean(img_array[:, :, 1]),  # Green mean
            np.mean(img_array[:, :, 2])   # Blue mean
        ])

        # 2. Standard deviation RGB
        features.extend([
            np.std(img_array[:, :, 0]),   # Red std
            np.std(img_array[:, :, 1]),   # Green std
            np.std(img_array[:, :, 2])    # Blue std
        ])

        # 3. Brightness dan contrast
        gray = np.mean(img_array, axis=2)
        features.append(np.mean(gray))        # Brightness
        features.append(np.std(gray))         # Contrast

        # 4. Edge density (simplified)
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))
        features.append(np.mean(edges_x))     # Horizontal edges
        features.append(np.mean(edges_y))     # Vertical edges

        # 5. Histogram features (simplified)
        hist, _ = np.histogram(gray.flatten(), bins=10, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalize
        features.extend(hist[:5])   # First 5 bins

        return np.array(features, dtype=float)

    def extract_features(self, image_path):
        """
        Ekstraksi fitur dasar dari file gambar
        
        Args:
            image_path (str): Path ke file gambar
            
        Returns:
            np.array: Array fitur
        """
        try:
            return self._extract_features_from_image(Image.open(image_path))
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return np.zeros(self.N_FEATURES)  # Return zero features if error
    
    def load_training_data(self, data_folder):
        """
        Load dan ekstraksi fitur dari folder training data
        
        Args:
            data_folder (str): Path ke folder training_data
            
        Returns:
            tuple: (features, labels)
        """
        features = []
        labels = []
        
        good_folder = os.path.join(data_folder, 'good')
        defect_folder = os.path.join(data_folder, 'defect')
        
        good_count = 0
        defect_count = 0
        
        # Load good images
        if os.path.exists(good_folder):
            for filename in os.listdir(good_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filepath = os.path.join(good_folder, filename)
                    feature = self.extract_features(filepath)
                    features.append(feature)
                    labels.append(0)  # 0 for good
                    good_count += 1
        
        # Load defect images  
        if os.path.exists(defect_folder):
            for filename in os.listdir(defect_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    filepath = os.path.join(defect_folder, filename)
                    feature = self.extract_features(filepath)
                    features.append(feature)
                    labels.append(1)  # 1 for defect
                    defect_count += 1
        
        self.training_data_count = {'good_count': good_count, 'defect_count': defect_count}
        
        return np.array(features), np.array(labels)
    
    def train(self, data_folder):
        """
        Train model dengan data dari folder
        
        Args:
            data_folder (str): Path ke folder training_data
            
        Returns:
            dict: Training results
        """
        print("Loading training data...")
        features, labels = self.load_training_data(data_folder)
        
        if len(features) == 0:
            raise ValueError("No training data found!")
        
        if len(np.unique(labels)) < 2:
            raise ValueError("Need both good and defect samples!")
        
        print(f"Loaded {len(features)} samples:")
        print(f"  - Good: {self.training_data_count['good_count']}")
        print(f"  - Defect: {self.training_data_count['defect_count']}")
        
        # Store training features and calculate statistics for anomaly detection
        self.training_features = features
        self.feature_stats = {
            'mean': np.mean(features, axis=0),
            'std': np.std(features, axis=0),
            'min': np.min(features, axis=0),
            'max': np.max(features, axis=0)
        }
        # Precompute normalized features + referensi jarak (sekali per training)
        self._prepare_anomaly_reference()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        self.trained = True
        self.trained_at = datetime.now()
        
        results = {
            'train_accuracy': train_acc,
            'val_accuracy': test_acc,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_importance': self.model.feature_importances_.tolist()
        }
        
        print(f"Training completed!")
        print(f"  - Train accuracy: {train_acc:.3f}")
        print(f"  - Test accuracy: {test_acc:.3f}")
        
        return results
    
    def _prepare_anomaly_reference(self):
        """
        Hitung nilai referensi anomaly dari data training satu kali.

        Menyimpan fitur training yang sudah dinormalisasi dan persentil ke-90
        dari rata-rata jarak antar-sampel training. Dengan begitu, setiap
        prediksi hanya perlu menghitung jarak ke sampel training (O(n)), bukan
        menghitung ulang seluruh matriks jarak (O(n^2)).
        """
        if self.training_features is None or self.feature_stats is None:
            self._normalized_training = None
            self.anomaly_reference = None
            return

        mean = self.feature_stats['mean']
        std = self.feature_stats['std']
        self._normalized_training = (self.training_features - mean) / (std + 1e-8)

        n_samples = len(self._normalized_training)
        if n_samples < 2:
            self.anomaly_reference = 1.0
            return

        # Jarak antar-sampel training (n x n)
        diff = self._normalized_training[:, None, :] - self._normalized_training[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, np.nan)
        mean_distances = np.nanmean(distances, axis=1)

        self.anomaly_reference = float(np.percentile(mean_distances, 90))

    def calculate_anomaly_score(self, features):
        """
        Calculate anomaly score based on distance from training data
        
        Args:
            features (np.array): Feature vector for the image
            
        Returns:
            float: Anomaly score (lower = more similar to training data)
        """
        if self.training_features is None or self.feature_stats is None:
            return 0.5  # Return neutral score if no training data

        # Pastikan nilai precomputed tersedia (mis. model lama hasil unpickle)
        if self._normalized_training is None or self.anomaly_reference is None:
            self._prepare_anomaly_reference()

        # Normalize features using training statistics
        normalized_features = (features - self.feature_stats['mean']) / (
            self.feature_stats['std'] + 1e-8
        )

        # Rata-rata jarak ke seluruh sampel training
        distances = np.linalg.norm(self._normalized_training - normalized_features, axis=1)
        avg_distance = float(np.mean(distances))

        reference = self.anomaly_reference or 1.0
        # Anomaly score: higher if distance is much larger than typical training distances
        return float(min(avg_distance / (reference + 1e-8), 3.0))  # Cap at 3.0
    
    def predict(self, image_array):
        """
        Prediksi untuk gambar baru dengan anomaly detection
        
        Args:
            image_array (np.array): Array gambar yang sudah dipreprocess
            
        Returns:
            tuple: (probability_array, anomaly_score)
        """
        if not self.trained:
            # Return random for untrained model
            prob = np.random.random()
            return np.array([[prob, 1-prob]]), 0.5
        
        # Extract features from preprocessed image
        # Convert normalized array back to 0-255 range
        if image_array.max() <= 1.0:
            img_uint8 = (image_array[0] * 255).astype(np.uint8)
        else:
            img_uint8 = image_array[0].astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img_uint8)

        # Ekstraksi fitur langsung dari memori (tanpa file sementara)
        features = self._extract_features_from_image(img).reshape(1, -1)

        # Calculate anomaly score
        anomaly_score = self.calculate_anomaly_score(features[0])

        # Get probabilities
        proba = self.model.predict_proba(features)[0]

        # Turunkan confidence saat anomaly tinggi dengan mencampur probabilitas
        # ke distribusi uniform. (Mengalikan semua probabilitas dengan konstanta
        # tidak berpengaruh karena hasilnya dinormalisasi kembali.)
        penalty = float(np.clip((anomaly_score - 1.0) * 0.3, 0.0, 0.6))
        uniform = np.full_like(proba, 1.0 / len(proba))
        adjusted_proba = proba * (1.0 - penalty) + uniform * penalty

        # Return as [prob_good, prob_defect] and anomaly score
        return np.array([adjusted_proba]), anomaly_score
    
    def save(self, filepath):
        """
        Simpan model ke file
        
        Args:
            filepath (str): Path untuk menyimpan model
        """
        model_data = {
            'model': self.model,
            'classes': self.classes,
            'trained': self.trained,
            'training_data_count': self.training_data_count,
            'trained_at': self.trained_at,
            'training_features': self.training_features,
            'feature_stats': self.feature_stats,
            'anomaly_reference': self.anomaly_reference,
            'model_type': 'SimpleDefectDetector'
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load model dari file
        
        Args:
            filepath (str): Path ke file model
            
        Returns:
            SimpleDefectDetector: Model yang sudah diload
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            detector = cls()
            detector.model = model_data['model']
            detector.classes = model_data['classes']
            detector.trained = model_data['trained']
            detector.training_data_count = model_data['training_data_count']
            detector.trained_at = model_data.get('trained_at')
            detector.training_features = model_data.get('training_features')
            detector.feature_stats = model_data.get('feature_stats')

            # Hitung ulang nilai precomputed anomaly (juga untuk model lama
            # yang belum menyimpannya di file pickle).
            detector._prepare_anomaly_reference()
            if model_data.get('anomaly_reference') is not None:
                detector.anomaly_reference = float(model_data['anomaly_reference'])

            print(f"Model loaded from: {filepath}")
            return detector
            
        except Exception as e:
            print(f"Error loading model: {e}")
            # Return dummy model if loading fails
            return cls()

if __name__ == "__main__":
    # Test the model
    detector = SimpleDefectDetector()
    
    # Train with sample data
    try:
        results = detector.train("static/training_data")
        
        # Save trained model
        detector.save("models/defect_detection_model.pkl")
        
        print("\n✅ Model training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
