"""
Autentikasi HTTP Basic untuk endpoint admin.

Aktif hanya jika `ADMIN_PASSWORD` di-set. Jika kosong, endpoint admin dapat
diakses tanpa autentikasi (memudahkan development lokal).
"""

import secrets
from functools import wraps

from flask import Response, current_app, jsonify, request


def _credentials_valid(username, password):
    expected_user = current_app.config.get('ADMIN_USERNAME', 'admin')
    expected_pass = current_app.config.get('ADMIN_PASSWORD', '')

    return secrets.compare_digest(username, expected_user) and secrets.compare_digest(
        password, expected_pass
    )


def require_admin(view):
    """Batasi akses endpoint ke admin (HTTP Basic Auth)."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        # Autentikasi nonaktif jika password belum dikonfigurasi.
        if not current_app.config.get('ADMIN_PASSWORD'):
            return view(*args, **kwargs)

        auth = request.authorization
        if auth and _credentials_valid(auth.username or '', auth.password or ''):
            return view(*args, **kwargs)

        challenge = {'WWW-Authenticate': 'Basic realm="Admin Area"'}

        # Halaman training: tampilkan respons HTML agar browser memunculkan
        # dialog login. Endpoint fetch/JSON: kembalikan JSON 401.
        if request.method == 'GET' and request.path == '/training':
            return Response('Autentikasi diperlukan.', 401, challenge)

        return (
            jsonify({'success': False, 'message': 'Autentikasi diperlukan'}),
            401,
            challenge,
        )

    return wrapped
