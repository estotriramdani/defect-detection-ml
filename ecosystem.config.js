module.exports = {
  apps: [
    {
      name: 'defect-detection-ml',
      script: 'app.py',
      interpreter: 'venv/bin/python',
      watch: false,
      env: {
        FLASK_DEBUG: '0',
        PORT: 5001,
        // Set SECRET_KEY dan ADMIN_PASSWORD di server production:
        // SECRET_KEY: '...',
        // ADMIN_PASSWORD: '...',
      },
      cwd: __dirname,
      autorestart: true,
      max_restarts: 5,
      restart_delay: 5000,
      error_file: './logs/pm2-err.log',
      out_file: './logs/pm2-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
    },
  ],
};
