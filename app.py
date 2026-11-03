"""
Flask Web Application untuk Deteksi Cacat Label Kemasan.

Entry point aplikasi. Konfigurasi ada di `config.py`, route di `blueprints/`,
dan logika model/training di `services/`.
"""

from flask import Flask

from blueprints.api import api_bp
from blueprints.main import main_bp
from blueprints.training import training_bp
from config import Config
from services import model_service
from utils import ensure_folders


def create_app(config_object=Config):
    """Application factory."""
    app = Flask(__name__)
    app.config.from_object(config_object)

    ensure_folders(app.config['UPLOAD_FOLDER'], app.config['TRAINING_DATA_FOLDER'])
    model_service.load_model(app.config['MODEL_PATH'])

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(training_bp)

    if not app.config.get('ADMIN_PASSWORD'):
        app.logger.warning(
            'ADMIN_PASSWORD belum diset: endpoint training/manajemen data '
            'dapat diakses tanpa autentikasi.'
        )

    return app


app = create_app()

if __name__ == '__main__':
    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG'],
    )
