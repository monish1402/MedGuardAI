#!/usr/bin/env python3
"""
MedGuard AI - Autonomous Medical Device Security System
Main application entry point
"""

import os
from app.routes import app, socketio, threat_analyzer
from app.database import init_database
from utils.logger import medguard_logger

def initialize_application():
    """Initialize the MedGuard AI application"""
    try:
        # Ensure data directories exist
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/logs', exist_ok=True)
        
        # Initialize database
        medguard_logger.info("Initializing MedGuard AI database...")
        init_database()
        
        # Initialize threat analyzer (this will train ML model if needed)
        medguard_logger.info("MedGuard AI system initialized successfully")
        
    except Exception as e:
        medguard_logger.error(f"Failed to initialize application: {e}")
        raise

if __name__ == '__main__':
    try:
        medguard_logger.info("Starting MedGuard AI - Autonomous Medical Device Security System")
        
        # Initialize application
        initialize_application()
        
        # Get configuration
        host = '0.0.0.0'
        port = 5000
        debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
        
        medguard_logger.info(f"Starting MedGuard AI server on {host}:{port}")
        
        # Start the Flask-SocketIO server
        socketio.run(
            app, 
            host=host, 
            port=port, 
            debug=debug,
            allow_unsafe_werkzeug=True
        )
        
    except Exception as e:
        medguard_logger.error(f"Failed to start MedGuard AI server: {e}")
        raise