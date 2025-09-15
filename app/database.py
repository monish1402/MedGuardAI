import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.config import Config

class MedGuardDatabase:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.init_database()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database with medical device security schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY,
                device_type TEXT NOT NULL,
                manufacturer TEXT,
                model TEXT,
                location TEXT,
                criticality_level TEXT,
                status TEXT,
                last_seen DATETIME,
                security_score REAL,
                patient_connected BOOLEAN DEFAULT FALSE,
                network_baseline TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                timestamp DATETIME,
                threat_level TEXT,
                confidence REAL,
                features TEXT,
                response_action TEXT,
                response_time_ms INTEGER,
                FOREIGN KEY (device_id) REFERENCES devices (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                event_type TEXT,
                device_id TEXT,
                details TEXT,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_device(self, device_data: Dict[str, Any]) -> bool:
        """Add a medical device to the database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO devices 
                (id, device_type, manufacturer, model, location, criticality_level, 
                 status, last_seen, security_score, patient_connected, network_baseline)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                device_data.get('id'),
                device_data.get('device_type'),
                device_data.get('manufacturer'),
                device_data.get('model'),
                device_data.get('location'),
                device_data.get('criticality_level'),
                device_data.get('status'),
                device_data.get('last_seen'),
                device_data.get('security_score'),
                device_data.get('patient_connected', False),
                json.dumps(device_data.get('network_baseline', {}))
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding device: {e}")
            return False
    
    def get_devices(self) -> List[Dict[str, Any]]:
        """Get all medical devices"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM devices')
        devices = []
        
        for row in cursor.fetchall():
            device = {
                'id': row[0],
                'device_type': row[1],
                'manufacturer': row[2],
                'model': row[3],
                'location': row[4],
                'criticality_level': row[5],
                'status': row[6],
                'last_seen': row[7],
                'security_score': row[8],
                'patient_connected': bool(row[9]),
                'network_baseline': json.loads(row[10] or '{}')
            }
            devices.append(device)
        
        conn.close()
        return devices
    
    def log_threat_detection(self, detection_data: Dict[str, Any]) -> Optional[int]:
        """Log a threat detection event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO threat_detections 
            (device_id, timestamp, threat_level, confidence, features, response_action, response_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection_data.get('device_id'),
            detection_data.get('timestamp', datetime.now()),
            detection_data.get('threat_level'),
            detection_data.get('confidence'),
            json.dumps(detection_data.get('features', {})),
            detection_data.get('response_action'),
            detection_data.get('response_time_ms')
        ))
        
        detection_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return detection_id
    
    def get_recent_threats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent threat detections"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM threat_detections 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        threats = []
        for row in cursor.fetchall():
            threat = {
                'id': row[0],
                'device_id': row[1],
                'timestamp': row[2],
                'threat_level': row[3],
                'confidence': row[4],
                'features': json.loads(row[5] or '{}'),
                'response_action': row[6],
                'response_time_ms': row[7]
            }
            threats.append(threat)
        
        conn.close()
        return threats
    
    def log_system_event(self, event_data: Dict[str, Any]):
        """Log a system event"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_events (timestamp, event_type, device_id, details, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            event_data.get('timestamp', datetime.now()),
            event_data.get('event_type'),
            event_data.get('device_id'),
            json.dumps(event_data.get('details', {})),
            event_data.get('severity', 'INFO')
        ))
        
        conn.commit()
        conn.close()

def init_database():
    """Initialize the MedGuard database"""
    db = MedGuardDatabase()
    return db