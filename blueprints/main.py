"""Halaman utama dan upload gambar untuk prediksi."""

import os
from datetime import datetime

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from services import model_service
from utils import allowed_file, ensure_folders

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    return render_template('index.html')


@main_bp.route('/upload', methods=['POST'])
def upload_file():
    config = current_app.config

    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('main.index'))

    file = request.files['file']
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('main.index'))

    if not (file and allowed_file(file.filename, config['ALLOWED_EXTENSIONS'])):
        flash('File tidak valid. Gunakan format: PNG, JPG, JPEG, GIF, atau BMP')
        return redirect(url_for('main.index'))

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

    return render_template(
        'result.html',
        filename=filename,
        prediction=result['prediction'],
        confidence=result['confidence'],
        anomaly_score=result['anomaly_score'],
        status=result['status'],
    )


@main_bp.app_errorhandler(413)
def too_large(_error):
    flash('File terlalu besar. Maksimal 16MB')
    return redirect(url_for('main.index'))
