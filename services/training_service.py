"""
Manajemen training di background thread.

Status training dan proses start dilindungi lock agar dua request tidak dapat
menjalankan training secara bersamaan.
"""

import threading
import time

from real_model import SimpleDefectDetector

from . import model_service

TOTAL_STEPS = 4


def _initial_status():
    return {
        'status': 'idle',  # idle | running | completed | error
        'current_step': 0,
        'total_steps': TOTAL_STEPS,
        'message': '',
        'results': {},
    }


class TrainingManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._status = _initial_status()
        self._worker = None

    def snapshot(self):
        """Salin status agar aman dibaca dari thread lain."""
        with self._lock:
            return {**self._status, 'results': dict(self._status['results'])}

    def is_running(self):
        with self._lock:
            return self._status['status'] == 'running'

    def start(self, data_folder, model_path):
        """Mulai training. Mengembalikan (success, message)."""
        with self._lock:
            if self._status['status'] == 'running':
                return False, 'Training sedang berjalan'

            self._status = {
                **_initial_status(),
                'status': 'running',
                'message': 'Memulai training...',
            }

        self._worker = threading.Thread(
            target=self._run,
            args=(data_folder, model_path),
            daemon=True,
            name='training-worker',
        )
        self._worker.start()
        return True, 'Training dimulai'

    def _update(self, **fields):
        with self._lock:
            self._status.update(fields)

    def _run(self, data_folder, model_path):
        try:
            self._update(current_step=0, message='Memuat dataset...')
            time.sleep(1)

            self._update(current_step=1, message='Preprocessing data...')
            time.sleep(1)

            self._update(current_step=2, message='Training model...')
            detector = SimpleDefectDetector()
            results = detector.train(data_folder)

            self._update(current_step=3, message='Menyimpan model...')
            detector.save(model_path)
            time.sleep(1)

            model_service.set_model(detector)

            self._update(
                status='completed',
                current_step=TOTAL_STEPS,
                message='Training selesai!',
                results=results,
            )

        except Exception as exc:
            self._update(
                status='error',
                message=f'Error during training: {exc}',
            )


# Singleton yang dipakai blueprint training.
training_manager = TrainingManager()
