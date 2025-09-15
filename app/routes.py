from flask import render_template, jsonify, request
from flask_socketio import emit
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app import create_app
from app.database import MedGuardDatabase
from core.threat_analyzer import MedGuardThreatAnalyzer
from utils.logger import medguard_logger

app, socketio = create_app()

db = MedGuardDatabase()
threat_analyzer = MedGuardThreatAnalyzer(db)

def make_json_serializable(obj):
    """Convert numpy types and other non-serializable objects to JSON-serializable types"""
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(v) for v in obj]
    
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  
        return obj.item()
    
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    
    elif hasattr(obj, 'items'):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    else:
        return obj

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def get_system_status():
    """Get current system status"""
    try:
        diagnostics = threat_analyzer.get_system_diagnostics()
        
        system_status = {
            'status': diagnostics.get('system_status', 'UNKNOWN'),
            'timestamp': datetime.now().isoformat(),
            'ml_model': {
                'status': diagnostics.get('ml_model', {}).get('status', 'unknown'),
                'accuracy': diagnostics.get('ml_model', {}).get('accuracy', 0),
                'avg_prediction_time_ms': diagnostics.get('ml_model', {}).get('avg_prediction_time_ms', 0)
            },
            'threat_activity': diagnostics.get('threat_activity', {}),
            'performance_metrics': diagnostics.get('performance_metrics', {}),
            'device_count': diagnostics.get('device_inventory', {}).get('total_devices', 0),
            'active_devices': diagnostics.get('device_inventory', {}).get('active_devices', 0)
        }
        
        return jsonify(system_status)
        
    except Exception as e:
        medguard_logger.error(f"Error getting system status: {e}")
        return jsonify({
            'status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/devices')
def get_devices():
    """Get all medical devices"""
    try:
        device_type = request.args.get('type')
        status = request.args.get('status', 'active')
        
        devices = threat_analyzer.device_manager.get_devices(device_type, status)
        
        formatted_devices = []
        for device in devices:
            formatted_device = {
                'id': device['id'],
                'device_type': device['device_type'],
                'manufacturer': device.get('manufacturer', 'Unknown'),
                'model': device.get('model', 'Unknown'),
                'location': device.get('location', 'Unknown'),
                'criticality_level': device.get('criticality_level', 'MEDIUM'),
                'status': device.get('status', 'unknown'),
                'last_seen': device.get('last_seen'),
                'security_score': round(device.get('security_score', 0), 2),
                'patient_connected': device.get('patient_connected', False)
            }
            formatted_devices.append(formatted_device)
        
        return jsonify({
            'devices': formatted_devices,
            'total_count': len(formatted_devices),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        medguard_logger.error(f"Error getting devices: {e}")
        return jsonify({
            'error': str(e),
            'devices': [],
            'total_count': 0
        }), 500

@app.route('/api/devices/<device_id>')
def get_device_details(device_id: str):
    """Get detailed information about a specific device"""
    try:
        device = threat_analyzer.device_manager.get_device(device_id)
        
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        
        recent_threats = db.get_recent_threats(10)
        device_threats = [t for t in recent_threats if t.get('device_id') == device_id]
        
        device_details = {
            'device': device,
            'recent_threats': device_threats,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(device_details)
        
    except Exception as e:
        medguard_logger.error(f"Error getting device details for {device_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/threats')
def get_threats():
    """Get recent threat detections"""
    try:
        limit = min(int(request.args.get('limit', 50)), 100)  # Max 100
        threats = db.get_recent_threats(limit)
        
        return jsonify({
            'threats': threats,
            'total_count': len(threats),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        medguard_logger.error(f"Error getting threats: {e}")
        return jsonify({
            'error': str(e),
            'threats': []
        }), 500

@app.route('/api/simulate-threat', methods=['POST'])
def simulate_threat():
    """Simulate a threat for demonstration purposes"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        device_id = data.get('device_id')
        threat_type = data.get('threat_type', 'malware')
        
        if not device_id:
            return jsonify({'error': 'device_id is required'}), 400
        
        device = threat_analyzer.device_manager.get_device(device_id)
        if not device:
            return jsonify({'error': 'Device not found'}), 404
        
        simulation_result = threat_analyzer.simulate_threat_scenario(device_id, threat_type)
        
        socketio.emit('threat_alert', {
            'device_id': device_id,
            'device_type': device.get('device_type'),
            'threat_level': simulation_result.get('threat_analysis', {}).get('threat_level', 'UNKNOWN'),
            'confidence': simulation_result.get('threat_analysis', {}).get('confidence', 0),
            'simulation': True,
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({
            'success': True,
            'simulation_result': simulation_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        medguard_logger.error(f"Error simulating threat: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-device/<device_id>', methods=['POST'])
def analyze_device(device_id: str):
    """Analyze a specific device for threats"""
    try:
        analysis_result = threat_analyzer.analyze_device_threat(device_id)
        
        if analysis_result.get('analysis_failed'):
            return jsonify({
                'success': False,
                'error': analysis_result.get('error', 'Analysis failed')
            }), 500
        
        if analysis_result.get('threat_analysis', {}).get('is_anomaly'):
            socketio.emit('threat_alert', {
                'device_id': device_id,
                'device_type': analysis_result.get('device_info', {}).get('device_type'),
                'threat_level': analysis_result.get('threat_analysis', {}).get('threat_level'),
                'confidence': analysis_result.get('threat_analysis', {}).get('confidence'),
                'simulation': False,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'analysis_result': analysis_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        medguard_logger.error(f"Error analyzing device {device_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/diagnostics')
def get_system_diagnostics():
    """Get comprehensive system diagnostics"""
    try:
        diagnostics = threat_analyzer.get_system_diagnostics()
        clean_diagnostics = make_json_serializable(diagnostics)
        return jsonify(clean_diagnostics)
        
    except Exception as e:
        medguard_logger.error(f"Error getting diagnostics: {e}")
        return jsonify({
            'system_status': 'ERROR',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'MedGuard AI',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    medguard_logger.info("Client connected to WebSocket")
    emit('connection_status', {
        'connected': True,
        'message': 'Connected to MedGuard AI',
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    medguard_logger.info("Client disconnected from WebSocket")

@socketio.on('request_system_update')
def handle_system_update_request():
    """Handle request for system status update"""
    try:
        diagnostics = threat_analyzer.get_system_diagnostics()
        emit('system_update', {
            'system_status': diagnostics.get('system_status'),
            'performance_metrics': diagnostics.get('performance_metrics', {}),
            'device_count': diagnostics.get('device_inventory', {}).get('total_devices', 0),
            'recent_threats': diagnostics.get('threat_activity', {}).get('recent_threats_count', 0),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        medguard_logger.error(f"Error handling system update request: {e}")
        emit('system_update', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('request_device_status')
def handle_device_status_request(data):
    """Handle request for device status update"""
    try:
        device_id = data.get('device_id') if data else None
        
        if device_id:
            device = threat_analyzer.device_manager.get_device(device_id)
            if device:
                emit('device_status', {
                    'device_id': device_id,
                    'device': device,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                emit('device_status', {
                    'error': f'Device {device_id} not found',
                    'timestamp': datetime.now().isoformat()
                })
        else:
            devices = threat_analyzer.device_manager.get_devices(status='active')
            emit('device_status', {
                'devices': devices[:20],  
                'total_count': len(devices),
                'timestamp': datetime.now().isoformat()
            })
            
    except Exception as e:
        medguard_logger.error(f"Error handling device status request: {e}")
        emit('device_status', {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

def broadcast_system_metrics():
    """Broadcast system metrics to all connected clients"""
    try:
        diagnostics = threat_analyzer.get_system_diagnostics()
        socketio.emit('system_metrics', {
            'system_health': diagnostics.get('system_status'),
            'performance_metrics': diagnostics.get('performance_metrics', {}),
            'ml_model_status': diagnostics.get('ml_model', {}).get('status'),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        medguard_logger.error(f"Error broadcasting system metrics: {e}")

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500