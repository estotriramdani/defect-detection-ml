"""Halaman training dan manajemen dataset (area admin)."""

import os
from datetime import datetime

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from auth import require_admin
from services.training_service import training_manager
from utils import (
    IMAGE_EXTENSIONS,
    allowed_file,
    count_training_images,
    ensure_folders,
    list_training_images,
)

training_bp = Blueprint('training', __name__)


def _config():
    return current_app.config


def _valid_category(category):
    return category in {'good', 'defect'}


def _is_safe_filename(filename):
    """Cegah path traversal: nama file harus sama dengan hasil sanitasi."""
    return bool(filename) and secure_filename(filename) == filename


@training_bp.route('/training')
@require_admin
def training():
    data = list_training_images(_config()['TRAINING_DATA_FOLDER'])
    return render_template(
        'training.html',
        good_count=data['good_count'],
        defect_count=data['defect_count'],
        good_images=data['good_images'],
        defect_images=data['defect_images'],
    )


@training_bp.route('/upload_training_data', methods=['POST'])
@require_admin
def upload_training_data():
    config = _config()

    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'Tidak ada file yang dikirim'})

    files = request.files.getlist('files')
    category = request.form.get('category')

    if not _valid_category(category):
        return jsonify({'success': False, 'message': 'Kategori tidak valid'})

    if not files or files[0].filename == '':
        return jsonify({'success': False, 'message': 'Tidak ada file yang dipilih'})

    ensure_folders(config['UPLOAD_FOLDER'], config['TRAINING_DATA_FOLDER'])
    target_folder = os.path.join(config['TRAINING_DATA_FOLDER'], category)

    uploaded_count = 0
    for file in files:
        if not (file and allowed_file(file.filename, config['ALLOWED_EXTENSIONS'])):
            continue

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
        filename = f'{timestamp}{uploaded_count}_{filename}'
        file.save(os.path.join(target_folder, filename))
        uploaded_count += 1

    counts = count_training_images(config['TRAINING_DATA_FOLDER'])
    return jsonify(
        {
            'success': True,
            'message': f'{uploaded_count} file berhasil diupload',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count'],
        }
    )


@training_bp.route('/start_training', methods=['POST'])
@require_admin
def start_training():
    config = _config()
    counts = count_training_images(config['TRAINING_DATA_FOLDER'])
    minimum = config['TRAIN_MIN_PER_CLASS']

    if counts['good_count'] < minimum or counts['defect_count'] < minimum:
        return jsonify(
            {
                'success': False,
                'message': f'Minimal {minimum} gambar per kategori diperlukan',
            }
        )

    success, message = training_manager.start(
        config['TRAINING_DATA_FOLDER'], config['MODEL_PATH']
    )
    return jsonify({'success': success, 'message': message})


@training_bp.route('/training_status')
@require_admin
def get_training_status():
    return jsonify(training_manager.snapshot())


@training_bp.route('/get_training_data')
@require_admin
def get_training_data():
    data = list_training_images(_config()['TRAINING_DATA_FOLDER'])
    return jsonify({'success': True, 'data': data})


@training_bp.route('/delete_training_image', methods=['POST'])
@require_admin
def delete_training_image():
    config = _config()
    data = request.get_json(silent=True) or {}
    category = data.get('category')
    filename = data.get('filename')

    if not _valid_category(category):
        return jsonify({'success': False, 'message': 'Kategori tidak valid'})

    if not _is_safe_filename(filename):
        return jsonify({'success': False, 'message': 'Nama file tidak valid'})

    file_path = os.path.join(config['TRAINING_DATA_FOLDER'], category, filename)
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': 'File tidak ditemukan'})

    os.remove(file_path)

    counts = count_training_images(config['TRAINING_DATA_FOLDER'])
    return jsonify(
        {
            'success': True,
            'message': 'Gambar berhasil dihapus',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count'],
        }
    )


@training_bp.route('/clear_training_data', methods=['POST'])
@require_admin
def clear_training_data():
    config = _config()
    data = request.get_json(silent=True) or {}
    category = data.get('category')

    if category not in {'good', 'defect', 'all'}:
        return jsonify({'success': False, 'message': 'Kategori tidak valid'})

    categories = ['good', 'defect'] if category == 'all' else [category]
    deleted_count = 0

    for item in categories:
        folder = os.path.join(config['TRAINING_DATA_FOLDER'], item)
        if not os.path.isdir(folder):
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(IMAGE_EXTENSIONS):
                os.remove(file_path)
                deleted_count += 1

    counts = count_training_images(config['TRAINING_DATA_FOLDER'])
    return jsonify(
        {
            'success': True,
            'message': f'{deleted_count} gambar berhasil dihapus',
            'good_count': counts['good_count'],
            'defect_count': counts['defect_count'],
        }
    )
