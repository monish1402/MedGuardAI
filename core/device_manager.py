import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from app.database import MedGuardDatabase
from app.config import Config

class MedGuardDeviceManager:
    """Medical device inventory and traffic generation manager"""
    
    def __init__(self, database: MedGuardDatabase = None):
        self.db = database or MedGuardDatabase()
        self.device_templates = self._initialize_device_templates()
        self.active_devices = {}
        self.traffic_baselines = {}
        
        self._initialize_sample_devices()
    
    def _initialize_device_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for different medical device types"""
        return {
            'ventilator': {
                'manufacturer_pool': ['Philips Respironics', 'Medtronic', 'Hamilton Medical', 'Draeger'],
                'model_pool': ['V60', 'PB980', 'G5', 'Evita V500'],
                'criticality_level': 'CRITICAL',
                'typical_locations': ['ICU-1', 'ICU-2', 'ICU-3', 'Emergency Room', 'OR-1', 'OR-2'],
                'network_profile': {
                    'packets_per_second': (40, 80),
                    'avg_packet_size': (256, 768),
                    'bandwidth_utilization': (0.1, 0.3),
                    'unique_destinations': (2, 5),
                    'connection_count': (3, 8),
                    'protocol_diversity': (0.2, 0.4),
                    'encryption_ratio': (0.8, 1.0)
                },
                'patient_connection_probability': 0.9
            },
            'patient_monitor': {
                'manufacturer_pool': ['Philips', 'GE Healthcare', 'Mindray', 'Nihon Kohden'],
                'model_pool': ['IntelliVue MX800', 'B650', 'uMEC12', 'BSM-6701'],
                'criticality_level': 'HIGH',
                'typical_locations': ['ICU-1', 'ICU-2', 'ICU-3', 'CCU', 'Ward-A', 'Ward-B', 'Emergency Room'],
                'network_profile': {
                    'packets_per_second': (20, 60),
                    'avg_packet_size': (128, 512),
                    'bandwidth_utilization': (0.05, 0.2),
                    'unique_destinations': (1, 4),
                    'connection_count': (2, 6),
                    'protocol_diversity': (0.1, 0.3),
                    'encryption_ratio': (0.7, 0.95)
                },
                'patient_connection_probability': 0.85
            },
            'infusion_pump': {
                'manufacturer_pool': ['Baxter', 'B. Braun', 'Smiths Medical', 'ICU Medical'],
                'model_pool': ['Sigma Spectrum', 'Perfusor Space', 'CADD-Legacy', 'Plum A+'],
                'criticality_level': 'HIGH',
                'typical_locations': ['ICU-1', 'ICU-2', 'Ward-A', 'Ward-B', 'Ward-C', 'Oncology'],
                'network_profile': {
                    'packets_per_second': (10, 40),
                    'avg_packet_size': (64, 256),
                    'bandwidth_utilization': (0.02, 0.15),
                    'unique_destinations': (1, 3),
                    'connection_count': (1, 4),
                    'protocol_diversity': (0.1, 0.25),
                    'encryption_ratio': (0.75, 0.9)
                },
                'patient_connection_probability': 0.8
            },
            'ct_scanner': {
                'manufacturer_pool': ['Siemens', 'GE Healthcare', 'Philips', 'Canon Medical'],
                'model_pool': ['Somatom Definition', 'Revolution CT', 'Ingenuity CT', 'Aquilion ONE'],
                'criticality_level': 'MEDIUM',
                'typical_locations': ['Radiology-1', 'Radiology-2', 'Emergency Radiology'],
                'network_profile': {
                    'packets_per_second': (100, 300),
                    'avg_packet_size': (1024, 1500),
                    'bandwidth_utilization': (0.3, 0.8),
                    'unique_destinations': (3, 8),
                    'connection_count': (5, 15),
                    'protocol_diversity': (0.3, 0.6),
                    'encryption_ratio': (0.6, 0.85)
                },
                'patient_connection_probability': 0.6
            },
            'defibrillator': {
                'manufacturer_pool': ['Zoll', 'Philips', 'Medtronic', 'Stryker'],
                'model_pool': ['R Series', 'HeartStart MRx', 'LIFEPAK 15', 'LUCAS 3'],
                'criticality_level': 'CRITICAL',
                'typical_locations': ['Emergency Room', 'ICU-1', 'ICU-2', 'CCU', 'OR-1', 'Code Cart-1', 'Code Cart-2'],
                'network_profile': {
                    'packets_per_second': (5, 30),
                    'avg_packet_size': (128, 512),
                    'bandwidth_utilization': (0.01, 0.1),
                    'unique_destinations': (1, 3),
                    'connection_count': (1, 5),
                    'protocol_diversity': (0.1, 0.2),
                    'encryption_ratio': (0.8, 1.0)
                },
                'patient_connection_probability': 0.3
            }
        }
    
    def _initialize_sample_devices(self):
        """Initialize sample medical devices for demonstration"""
        device_counts = {
            'ventilator': 8,
            'patient_monitor': 15,
            'infusion_pump': 12,
            'ct_scanner': 3,
            'defibrillator': 6
        }
        
        for device_type, count in device_counts.items():
            for i in range(count):
                device = self.create_device(device_type)
                self.add_device(device)
    
    def create_device(self, device_type: str) -> Dict[str, Any]:
        """Create a new medical device with realistic specifications"""
        if device_type not in self.device_templates:
            raise ValueError(f"Unknown device type: {device_type}")
        
        template = self.device_templates[device_type]
        device_id = f"{device_type.upper()}_{uuid.uuid4().hex[:8]}"
        
        network_baseline = {}
        for metric, (min_val, max_val) in template['network_profile'].items():
            network_baseline[metric] = random.uniform(min_val, max_val)
        
        device = {
            'id': device_id,
            'device_type': device_type,
            'manufacturer': random.choice(template['manufacturer_pool']),
            'model': random.choice(template['model_pool']),
            'location': random.choice(template['typical_locations']),
            'criticality_level': template['criticality_level'],
            'status': 'active',
            'last_seen': datetime.now(),
            'security_score': random.uniform(0.8, 0.95),
            'patient_connected': random.random() < template['patient_connection_probability'],
            'network_baseline': network_baseline,
            'created_at': datetime.now(),
            'uptime_hours': random.randint(1, 8760)  
        }
        
        return device
    
    def add_device(self, device_data: Dict[str, Any]) -> bool:
        """Add a device to the inventory"""
        try:
            success = self.db.add_device(device_data)
            
            if success:
                self.active_devices[device_data['id']] = device_data
                self.traffic_baselines[device_data['id']] = device_data.get('network_baseline', {})
                return True
            return False
        except Exception as e:
            print(f"Error adding device {device_data.get('id')}: {e}")
            return False
    
    def get_devices(self, device_type: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get devices from inventory with optional filtering"""
        devices = self.db.get_devices()
        
        if device_type:
            devices = [d for d in devices if d['device_type'] == device_type]
        
        if status:
            devices = [d for d in devices if d['status'] == status]
        
        return devices
    
    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific device by ID"""
        if device_id in self.active_devices:
            return self.active_devices[device_id]
        
        devices = self.db.get_devices()
        for device in devices:
            if device['id'] == device_id:
                self.active_devices[device_id] = device
                return device
        
        return None
    
    def update_device_status(self, device_id: str, status: str) -> bool:
        """Update device status"""
        device = self.get_device(device_id)
        if device:
            device['status'] = status
            device['last_seen'] = datetime.now()
            return self.db.add_device(device)  
        return False
    
    def generate_network_traffic(self, device_id: str, add_noise: bool = True) -> Dict[str, float]:
        """Generate realistic network traffic data for a device"""
        device = self.get_device(device_id)
        if not device:
            raise ValueError(f"Device {device_id} not found")
        
        baseline = self.traffic_baselines.get(device_id, {})
        if not baseline:
            template = self.device_templates.get(device['device_type'], {})
            baseline = {}
            for metric, (min_val, max_val) in template.get('network_profile', {}).items():
                baseline[metric] = random.uniform(min_val, max_val)
            self.traffic_baselines[device_id] = baseline
        
        traffic_data = {}
        
        for metric, base_value in baseline.items():
            if add_noise:
                noise_factor = random.uniform(0.8, 1.2)  
                time_factor = 1 + 0.1 * random.sin(time.time() / 3600)  
                traffic_data[metric] = base_value * noise_factor * time_factor
            else:
                traffic_data[metric] = base_value
        
        traffic_data.update({
            'failed_connections': random.uniform(0, 0.02),
            'suspicious_ports': random.uniform(0, 0.01),
            'malformed_packets': random.uniform(0, 0.001),
            'unusual_timing': random.uniform(0, 0.1),
            'data_exfiltration_score': random.uniform(0, 0.1),
            'device_criticality_score': {
                'CRITICAL': 1.0, 'HIGH': 0.8, 'MEDIUM': 0.6, 'LOW': 0.4
            }.get(device.get('criticality_level', 'MEDIUM'), 0.5),
            'patient_connection_status': 1.0 if device.get('patient_connected') else 0.0,
            'medical_protocol_compliance': random.uniform(0.85, 1.0),
            'emergency_mode_indicator': 0,  
            'baseline_deviation_score': random.uniform(0, 0.2),
            'entropy_score': random.uniform(0.6, 0.8),
            'anomaly_burst_frequency': random.uniform(0, 0.01),
            'network_topology_changes': random.uniform(0, 0.005),
            'authentication_failures': random.uniform(0, 0.01),
            'configuration_changes': random.uniform(0, 0.002)
        })
        
        for key, value in traffic_data.items():
            if value < 0:
                traffic_data[key] = 0
        
        return traffic_data
    
    def simulate_anomalous_traffic(self, device_id: str, anomaly_type: str = 'malware') -> Dict[str, float]:
        """Generate anomalous traffic patterns for threat simulation"""
        normal_traffic = self.generate_network_traffic(device_id, add_noise=False)
        
        if anomaly_type == 'malware':
            normal_traffic.update({
                'packets_per_second': normal_traffic['packets_per_second'] * random.uniform(2.5, 4.0),
                'unique_destinations': normal_traffic['unique_destinations'] * random.uniform(3.0, 5.0),
                'failed_connections': random.uniform(0.15, 0.25),
                'suspicious_ports': random.uniform(0.1, 0.2),
                'data_exfiltration_score': random.uniform(0.6, 0.8),
                'protocol_violations': random.uniform(0.05, 0.1),
                'entropy_score': random.uniform(0.9, 0.95),
                'malformed_packets': random.uniform(0.01, 0.05)
            })
        
        elif anomaly_type == 'ransomware':
            normal_traffic.update({
                'packets_per_second': normal_traffic['packets_per_second'] * random.uniform(2.0, 3.5),
                'encryption_ratio': random.uniform(0.2, 0.4),  
                'network_topology_changes': random.uniform(0.1, 0.2),
                'configuration_changes': random.uniform(0.05, 0.1),
                'anomaly_burst_frequency': random.uniform(0.2, 0.4),
                'authentication_failures': random.uniform(0.15, 0.3),
                'protocol_violations': random.uniform(0.05, 0.1)
            })
        
        elif anomaly_type == 'data_breach':
            normal_traffic.update({
                'bandwidth_utilization': min(0.9, normal_traffic['bandwidth_utilization'] * random.uniform(3.0, 5.0)),
                'avg_packet_size': normal_traffic['avg_packet_size'] * random.uniform(2.0, 3.5),
                'data_exfiltration_score': random.uniform(0.7, 0.9),
                'unique_destinations': normal_traffic['unique_destinations'] * random.uniform(3.0, 5.0),
                'unusual_timing': random.uniform(0.5, 0.8),
                'failed_connections': random.uniform(0.1, 0.2),
                'suspicious_ports': random.uniform(0.1, 0.3)
            })
        
        elif anomaly_type == 'device_compromise':
            normal_traffic.update({
                'medical_protocol_compliance': random.uniform(0.6, 0.75),
                'baseline_deviation_score': random.uniform(0.3, 0.5),
                'protocol_violations': random.uniform(0.02, 0.04),
                'failed_connections': random.uniform(0.04, 0.08),
                'configuration_changes': random.uniform(0.02, 0.04)
            })
        
        return normal_traffic
    
    def get_device_statistics(self) -> Dict[str, Any]:
        """Get comprehensive device inventory statistics"""
        devices = self.get_devices()
        
        if not devices:
            return {'status': 'no_devices', 'total_devices': 0}
        
        type_counts = {}
        criticality_counts = {}
        status_counts = {}
        location_counts = {}
        patient_connected_count = 0
        
        for device in devices:
            device_type = device['device_type']
            type_counts[device_type] = type_counts.get(device_type, 0) + 1
            
            criticality = device['criticality_level']
            criticality_counts[criticality] = criticality_counts.get(criticality, 0) + 1
            
            status = device['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            
            location = device['location']
            location_counts[location] = location_counts.get(location, 0) + 1
            
            if device.get('patient_connected'):
                patient_connected_count += 1
        
        security_scores = [d.get('security_score', 0) for d in devices]
        avg_security_score = sum(security_scores) / len(security_scores) if security_scores else 0
        
        recent_threshold = datetime.now() - timedelta(hours=1)
        active_devices = [d for d in devices if 
                         isinstance(d.get('last_seen'), datetime) and d['last_seen'] > recent_threshold]
        
        statistics = {
            'status': 'active',
            'total_devices': len(devices),
            'active_devices': len(active_devices),
            'patient_connected_devices': patient_connected_count,
            'avg_security_score': round(avg_security_score, 3),
            'device_type_distribution': type_counts,
            'criticality_distribution': criticality_counts,
            'status_distribution': status_counts,
            'location_distribution': location_counts,
            'inventory_health': {
                'critical_devices': criticality_counts.get('CRITICAL', 0),
                'active_rate': len(active_devices) / len(devices) if devices else 0,
                'patient_connection_rate': patient_connected_count / len(devices) if devices else 0
            }
        }
        
        return statistics