"""
MedGuard AI Application Package
Flask application factory and configuration
"""
import os
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

def create_app():
    """Create and configure the Flask application"""
    import os
    
    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(project_root, 'templates')
    static_path = os.path.join(project_root, 'static')
    
    app = Flask(__name__, 
                template_folder=template_path,
                static_folder=static_path)
    
    app.config.from_object('app.config.Config')
    
    # Initialize extensions
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)
    
    return app, socketio