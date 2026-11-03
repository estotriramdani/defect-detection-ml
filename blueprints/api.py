"""REST API untuk prediksi."""

import os
from datetime import datetime

from flask import Blueprint, current_app, jsonify, request
from werkzeug.utils import secure_filename

from services import model_service
from utils import allowed_file, ensure_folders

api_bp = Blueprint('api', __name__)


@api_bp.route('/api/predict', methods=['POST'])
def api_predict():
    config = current_app.config

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if not allowed_file(file.filename, config['ALLOWED_EXTENSIONS']):
        return jsonify({'error': 'Format file tidak didukung'}), 400

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
    filename = timestamp + filename

    ensure_folders(config['UPLOAD_FOLDER'], config['TRAINING_DATA_FOLDER'])
    filepath = os.path.join(config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = model_service.predict_defect(
        filepath,
        config['CONFIDENCE_THRESHOLD'],
        config['ANOMALY_THRESHOLD'],
    )

    return jsonify(
        {
            'filename': filename,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'anomaly_score': result['anomaly_score'],
            'status': result['status'],
            'timestamp': datetime.now().isoformat(),
        }
    )
