"""
Helper untuk pengelolaan file dan dataset training.
"""

import os
from datetime import datetime

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def allowed_file(filename, allowed_extensions):
    """Cek apakah ekstensi file diizinkan."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def ensure_folders(upload_folder, training_folder):
    """Buat folder upload dan dataset training jika belum ada."""
    os.makedirs(upload_folder, exist_ok=True)
    for category in ('good', 'defect'):
        os.makedirs(os.path.join(training_folder, category), exist_ok=True)


def _is_image(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def count_training_images(training_folder):
    """Hitung jumlah gambar per kategori."""
    counts = {}
    for category in ('good', 'defect'):
        folder = os.path.join(training_folder, category)
        if os.path.isdir(folder):
            counts[f'{category}_count'] = len(
                [f for f in os.listdir(folder) if _is_image(f)]
            )
        else:
            counts[f'{category}_count'] = 0
    return counts


def list_training_images(training_folder):
    """Daftar gambar training beserta metadata untuk halaman training."""
    result = {}
    for category in ('good', 'defect'):
        folder = os.path.join(training_folder, category)
        images = []

        if os.path.isdir(folder):
            for filename in os.listdir(folder):
                if not _is_image(filename):
                    continue

                filepath = os.path.join(folder, filename)
                stat = os.stat(filepath)
                images.append(
                    {
                        'filename': filename,
                        'category': category,
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).strftime(
                            '%Y-%m-%d %H:%M:%S'
                        ),
                        'path': f'static/training_data/{category}/{filename}',
                    }
                )

        result[f'{category}_images'] = sorted(
            images, key=lambda item: item['created'], reverse=True
        )
        result[f'{category}_count'] = len(images)

    return result
