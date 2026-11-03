#!/usr/bin/env python3
"""
Evaluasi akurasi model deteksi cacat label.

Menjalankan evaluasi pada dataset training yang tersedia:
- 5-fold stratified cross-validation (metrik utama)
- Confusion matrix out-of-fold + interval kepercayaan
- Analisis kesalahan (file, confidence, anomaly)
- Feature importance
- Statistik confidence & anomaly score per kelas
- Sensitivitas threshold
- Prediksi pada gambar uji di static/base-data

Jalankan: python evaluate_model.py
"""

import os
import sys

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from config import Config
from real_model import SimpleDefectDetector
from services.model_service import preprocess_image

DATA_FOLDER = 'static/training_data'
MODEL_PATH = Config.MODEL_PATH
BASE_IMAGES = [
    ('static/base-data/label_fibemini.png', 'good'),
    ('static/base-data/sure-defect.png', 'unknown'),
]

FEATURE_NAMES = [
    'mean_R', 'mean_G', 'mean_B',
    'std_R', 'std_G', 'std_B',
    'brightness', 'contrast',
    'edge_horizontal', 'edge_vertical',
    'hist_bin_0', 'hist_bin_1', 'hist_bin_2', 'hist_bin_3', 'hist_bin_4',
]

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def load_dataset(detector):
    """Muat fitur + label + nama file (urutan sama dengan training)."""
    features, labels, filenames = [], [], []

    for category, label in (('good', 0), ('defect', 1)):
        folder = os.path.join(DATA_FOLDER, category)
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                filepath = os.path.join(folder, filename)
                features.append(detector.extract_features(filepath))
                labels.append(label)
                filenames.append(f'{category}/{filename}')

    return np.array(features), np.array(labels), filenames


def wilson_interval(successes, total, z=1.96):
    """Interval kepercayaan Wilson untuk proporsi binomial."""
    if total == 0:
        return 0.0, 0.0

    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denominator
    margin = z * np.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def adjusted_confidence(probabilities, anomaly_scores):
    """Confidence setelah penalti anomaly (sama seperti model_service)."""
    penalty = np.clip((anomaly_scores - 1.0) * 0.3, 0.0, 0.6)
    adjusted = probabilities * (1.0 - penalty[:, None]) + 0.5 * penalty[:, None]
    return adjusted.max(axis=1), adjusted.argmax(axis=1)


def evaluate():
    detector = SimpleDefectDetector.load(MODEL_PATH)
    features, labels, filenames = load_dataset(detector)

    if len(features) == 0:
        print('Tidak ada data training.')
        return

    class_names = ['good', 'defect']
    n_samples = len(labels)
    n_good = int(np.sum(labels == 0))
    n_defect = int(np.sum(labels == 1))
    anomaly_scores = np.array(
        [detector.calculate_anomaly_score(row) for row in features]
    )

    print('=== DATASET ===')
    print(f'Total sampel      : {n_samples}')
    print(f'Good              : {n_good}')
    print(f'Defect            : {n_defect}')
    print(f'Jumlah fitur      : {features.shape[1]}')
    print(f'Model tersimpan   : trained={detector.trained}, trained_at={detector.trained_at}')
    print(f'Baseline majority : {max(n_good, n_defect) / n_samples:.4f} '
          f'(selalu prediksi kelas terbanyak)')

    # --- 5-fold stratified cross-validation ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)

    fold_rows = []
    oof_predictions = np.zeros(n_samples, dtype=int)
    oof_probabilities = np.zeros((n_samples, 2))

    for fold, (train_idx, test_idx) in enumerate(cv.split(features, labels), start=1):
        model = clone(base_model)
        model.fit(features[train_idx], labels[train_idx])
        predictions = model.predict(features[test_idx])
        oof_predictions[test_idx] = predictions
        oof_probabilities[test_idx] = model.predict_proba(features[test_idx])

        fold_rows.append(
            {
                'fold': fold,
                'accuracy': accuracy_score(labels[test_idx], predictions),
                'precision': precision_score(
                    labels[test_idx], predictions, average='weighted', zero_division=0
                ),
                'recall': recall_score(
                    labels[test_idx], predictions, average='weighted', zero_division=0
                ),
                'f1': f1_score(
                    labels[test_idx], predictions, average='weighted', zero_division=0
                ),
            }
        )

    print('\n=== 5-FOLD STRATIFIED CROSS-VALIDATION ===')
    print(f"{'Fold':<6}{'Accuracy':>10}{'Precision':>12}{'Recall':>9}{'F1':>9}")
    for row in fold_rows:
        print(
            f"{row['fold']:<6}{row['accuracy']:>10.4f}{row['precision']:>12.4f}"
            f"{row['recall']:>9.4f}{row['f1']:>9.4f}"
        )

    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = {metric: np.mean([row[metric] for row in fold_rows]) for metric in metrics}
    stds = {metric: np.std([row[metric] for row in fold_rows]) for metric in metrics}

    print('\nRata-rata (mean ± std):')
    for metric in metrics:
        print(f'  {metric:<10}: {means[metric]:.4f} ± {stds[metric]:.4f}')

    # --- Confusion matrix & report out-of-fold ---
    cm = confusion_matrix(labels, oof_predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    oof_accuracy = accuracy_score(labels, oof_predictions)
    ci_low, ci_high = wilson_interval(int(tn + tp), n_samples)

    print('\n=== CONFUSION MATRIX (OUT-OF-FOLD) ===')
    print('                 Prediksi good   Prediksi defect')
    print(f'Aktual good     {tn:^15}{fp:^16}')
    print(f'Aktual defect   {fn:^15}{tp:^16}')
    print(f'\nTN={tn} FP={fp} FN={fn} TP={tp}')
    print(f'Akurasi out-of-fold : {oof_accuracy:.4f} ({tn + tp}/{n_samples})')
    print(f'95% Wilson CI       : [{ci_low:.4f}, {ci_high:.4f}]')

    print('\n=== CLASSIFICATION REPORT (OUT-OF-FOLD) ===')
    print(
        classification_report(
            labels,
            oof_predictions,
            labels=[0, 1],
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # --- Confidence & error analysis ---
    confidences, _ = adjusted_confidence(oof_probabilities, anomaly_scores)

    print('=== CONFIDENCE PER KELAS (OUT-OF-FOLD) ===')
    print(f"{'Kelas':<8}{'Mean':>8}{'Median':>9}{'Min':>8}{'Max':>8}")
    for label, name in enumerate(class_names):
        values = confidences[labels == label]
        print(
            f'{name:<8}{values.mean():>8.3f}{np.median(values):>9.3f}'
            f'{values.min():>8.3f}{values.max():>8.3f}'
        )

    errors = np.where(oof_predictions != labels)[0]
    print(f'\n=== ANALISIS KESALAHAN (OUT-OF-FOLD): {len(errors)} sampel ===')
    for index in errors:
        print(
            f'{filenames[index]} | aktual={class_names[labels[index]]} '
            f'prediksi={class_names[oof_predictions[index]]} '
            f'confidence={confidences[index]:.3f} anomaly={anomaly_scores[index]:.3f}'
        )

    # --- Akurasi model tersimpan pada data training (resubstitution) ---
    if detector.trained:
        resub_predictions = detector.model.predict(features)
        print('\n=== MODEL TERSIMPAN (RESUBSTITUTION) ===')
        print(f'Akurasi pada data training: {accuracy_score(labels, resub_predictions):.4f}')
        print('(Angka ini optimistis karena data evaluasi = data training.)')

    # --- Feature importance ---
    if detector.trained and hasattr(detector.model, 'feature_importances_'):
        importances = detector.model.feature_importances_
        order = np.argsort(importances)[::-1]
        print('\n=== FEATURE IMPORTANCE ===')
        print(f"{'Fitur':<18}{'Importance':>12}{'%':>8}")
        for index in order:
            print(
                f'{FEATURE_NAMES[index]:<18}{importances[index]:>12.4f}'
                f'{importances[index] * 100:>7.1f}%'
            )
        dominant = order[:3]
        print(
            'Top-3: ' + ', '.join(
                f'{FEATURE_NAMES[i]} ({importances[i] * 100:.1f}%)' for i in dominant
            )
        )

    # --- Statistik anomaly score ---
    print('\n=== ANOMALY SCORE PER KELAS ===')
    print(f"{'Kelas':<8}{'Mean':>8}{'Median':>9}{'Min':>8}{'Max':>8}{'> threshold':>13}")
    for label, name in enumerate(class_names):
        values = anomaly_scores[labels == label]
        above = int(np.sum(values > Config.ANOMALY_THRESHOLD))
        print(
            f'{name:<8}{values.mean():>8.3f}{np.median(values):>9.3f}'
            f'{values.min():>8.3f}{values.max():>8.3f}{f"{above}/{len(values)}":>13}'
        )
    print(f'Ambang anomaly   : {Config.ANOMALY_THRESHOLD}')
    print(f'Ambang confidence: {Config.CONFIDENCE_THRESHOLD}')

    # --- Sensitivitas threshold ---
    print('\n=== SENSITIVITAS CONFIDENCE THRESHOLD (anomaly=1.5) ===')
    print(f"{'Threshold':>10}{'Unknown (training)':>21}{'False unknown':>16}")
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        unknown = (confidences < threshold) | (anomaly_scores > 1.5)
        correct = oof_predictions == labels
        false_unknown = int(np.sum(unknown & correct))
        print(f'{threshold:>10.2f}{int(np.sum(unknown)):>15}/{n_samples:<5}{false_unknown:>16}')

    print('\n=== SENSITIVITAS ANOMALY THRESHOLD (confidence=0.6) ===')
    print(f"{'Threshold':>10}{'Unknown (training)':>21}{'False unknown':>16}")
    for threshold in (1.00, 1.25, 1.50, 2.00, 2.50, 3.00):
        unknown = (confidences < 0.6) | (anomaly_scores > threshold)
        correct = oof_predictions == labels
        false_unknown = int(np.sum(unknown & correct))
        print(f'{threshold:>10.2f}{int(np.sum(unknown)):>15}/{n_samples:<5}{false_unknown:>16}')

    # --- Prediksi gambar uji eksternal (lewat alur aplikasi) ---
    print('\n=== PREDIKSI GAMBAR UJI (static/base-data) ===')
    print(f"{'Gambar':<42}{'Label asli':>12}{'Prediksi':>12}{'Conf':>8}{'Anomaly':>9}")
    for path, expected in BASE_IMAGES:
        if not os.path.exists(path):
            print(f'{path:<42}{expected:>12}{"(tidak ada)":>12}')
            continue

        image_array = preprocess_image(path)
        prediction, anomaly = detector.predict(image_array)
        confidence = float(np.max(prediction[0]))
        predicted_class = int(np.argmax(prediction[0]))

        if confidence < Config.CONFIDENCE_THRESHOLD or anomaly > Config.ANOMALY_THRESHOLD:
            predicted = 'unknown'
        else:
            predicted = class_names[predicted_class]

        print(
            f'{os.path.basename(path):<42}{expected:>12}{predicted:>12}'
            f'{confidence:>8.3f}{anomaly:>9.3f}'
        )


if __name__ == '__main__':
    evaluate()
