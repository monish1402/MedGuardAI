#!/usr/bin/env python3
import os
from app.routes import app, socketio, threat_analyzer
from app.database import init_database
from utils.logger import medguard_logger

def initialize_application():
    try:
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/logs', exist_ok=True)
        medguard_logger.info("Initializing MedGuard AI database...")
        init_database()
        medguard_logger.info("MedGuard AI system initialized successfully")
    except Exception as e:
        medguard_logger.error(f"Failed to initialize application: {e}")
        raise

if __name__ == '__main__':
    try:
        medguard_logger.info("Starting MedGuard AI - Autonomous Medical Device Security System")
        initialize_application()
        host = '0.0.0.0'
        port = 5000
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        medguard_logger.info(f"Starting MedGuard AI server on {host}:{port}")
        socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
    except Exception as e:
        medguard_logger.error(f"Failed to start MedGuard AI server: {e}")
        raise
