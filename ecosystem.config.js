module.exports = {
  apps: [
    {
      name: 'defect-detection-ml',
      script: 'app.py',
      interpreter: 'venv/bin/python',
      watch: false,
      env: {
        // You can add environment variables here if needed
        // Example: "FLASK_ENV": "production"
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
