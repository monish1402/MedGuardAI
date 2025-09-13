import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from enum import Enum
from app.config import Config

class ResponseAction(Enum):
    MONITOR = "monitor"
    ALERT = "alert"
    ISOLATE = "isolate"
    QUARANTINE = "quarantine"
    EMERGENCY_PROTOCOL = "emergency_protocol"

class MedGuardResponseEngine:
    """Medical context-aware autonomous response engine"""
    
    def __init__(self):
        self.device_policies = self._initialize_device_policies()
        self.response_history = []
        
    def _initialize_device_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize device-specific response policies"""
        return {
            'ventilator': {
                'max_downtime_ms': 0,  # Zero downtime allowed
                'response_time_target_ms': 25,
                'escalation_threshold': 0.3,  # Low threshold for critical device
                'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.EMERGENCY_PROTOCOL],
                'patient_safety_override': True,
                'backup_protocols': ['activate_manual_mode', 'alert_respiratory_therapist', 'prepare_backup_unit'],
                'priority_level': 'CRITICAL',
                'requires_clinical_approval': False  # Immediate action needed
            },
            'defibrillator': {
                'max_downtime_ms': 100,  # Minimal downtime
                'response_time_target_ms': 25,
                'escalation_threshold': 0.3,
                'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.EMERGENCY_PROTOCOL],
                'patient_safety_override': True,
                'backup_protocols': ['activate_backup_unit', 'alert_cardiac_team', 'prepare_external_defibrillator'],
                'priority_level': 'CRITICAL',
                'requires_clinical_approval': False
            },
            'patient_monitor': {
                'max_downtime_ms': 5000,  # 5 seconds max
                'response_time_target_ms': 50,
                'escalation_threshold': 0.5,
                'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.ISOLATE],
                'patient_safety_override': True,
                'backup_protocols': ['enhance_monitoring_frequency', 'activate_redundant_sensors', 'alert_nursing_staff'],
                'priority_level': 'HIGH',
                'requires_clinical_approval': False
            },
            'infusion_pump': {
                'max_downtime_ms': 10000,  # 10 seconds max
                'response_time_target_ms': 75,
                'escalation_threshold': 0.6,
                'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.ISOLATE],
                'patient_safety_override': True,
                'backup_protocols': ['gradual_isolation', 'activate_backup_pump', 'alert_pharmacy'],
                'priority_level': 'HIGH',
                'requires_clinical_approval': True
            },
            'ct_scanner': {
                'max_downtime_ms': 60000,  # 1 minute max
                'response_time_target_ms': 100,
                'escalation_threshold': 0.7,
                'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.ISOLATE, ResponseAction.QUARANTINE],
                'patient_safety_override': False,
                'backup_protocols': ['safe_quarantine', 'schedule_maintenance', 'redirect_to_backup_scanner'],
                'priority_level': 'MEDIUM',
                'requires_clinical_approval': True
            }
        }
    
    def analyze_threat_context(self, threat_data: Dict[str, Any], device_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat in medical context and determine appropriate response"""
        start_time = time.time()
        
        device_type = device_info.get('device_type', 'unknown')
        device_policy = self.device_policies.get(device_type, self._get_default_policy())
        
        # Extract threat characteristics
        threat_level = threat_data.get('threat_level', 'LOW')
        confidence = threat_data.get('confidence', 0.0)
        is_anomaly = threat_data.get('is_anomaly', False)
        
        # Medical context factors
        patient_connected = device_info.get('patient_connected', False)
        emergency_mode = threat_data.get('medical_context', {}).get('emergency_mode', False)
        device_criticality = threat_data.get('device_criticality', 0.5)
        
        # Calculate risk score
        risk_score = self._calculate_medical_risk_score(
            confidence, device_criticality, patient_connected, emergency_mode, threat_level
        )
        
        # Determine response action
        response_action = self._determine_response_action(
            risk_score, device_policy, threat_level, patient_connected
        )
        
        # Apply patient safety checks
        if device_policy['patient_safety_override'] and patient_connected:
            response_action = self._apply_patient_safety_override(
                response_action, device_type, risk_score
            )
        
        # Generate response plan
        response_plan = self._generate_response_plan(
            response_action, device_policy, device_info, threat_data
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log response decision
        self._log_response_decision(device_info['id'], threat_data, response_plan, processing_time)
        
        return {
            'response_action': response_action.value,
            'response_plan': response_plan,
            'risk_score': risk_score,
            'processing_time_ms': processing_time,
            'meets_target': processing_time < device_policy['response_time_target_ms'],
            'patient_safety_applied': device_policy['patient_safety_override'] and patient_connected,
            'requires_approval': device_policy['requires_clinical_approval'],
            'medical_context': {
                'device_type': device_type,
                'priority_level': device_policy['priority_level'],
                'patient_connected': patient_connected,
                'emergency_mode': emergency_mode,
                'max_downtime_ms': device_policy['max_downtime_ms']
            }
        }
    
    def _calculate_medical_risk_score(self, confidence: float, device_criticality: float, 
                                    patient_connected: bool, emergency_mode: bool, threat_level: str) -> float:
        """Calculate comprehensive medical risk score"""
        base_score = confidence
        
        # Device criticality multiplier (0.5x to 1.5x)
        criticality_multiplier = 0.5 + device_criticality
        
        # Patient connection multiplier (1.3x if patient connected)
        patient_multiplier = 1.3 if patient_connected else 1.0
        
        # Emergency mode multiplier (1.5x if in emergency mode)
        emergency_multiplier = 1.5 if emergency_mode else 1.0
        
        # Threat level multiplier
        threat_multipliers = {
            'HIGH': 1.2,
            'MEDIUM': 1.0,
            'LOW': 0.8,
            'NORMAL': 0.5
        }
        threat_multiplier = threat_multipliers.get(threat_level, 1.0)
        
        risk_score = base_score * criticality_multiplier * patient_multiplier * emergency_multiplier * threat_multiplier
        
        return min(1.0, risk_score)
    
    def _determine_response_action(self, risk_score: float, device_policy: Dict[str, Any], 
                                 threat_level: str, patient_connected: bool) -> ResponseAction:
        """Determine appropriate response action based on risk analysis"""
        allowed_actions = device_policy['allowed_actions']
        escalation_threshold = device_policy['escalation_threshold']
        
        if risk_score >= 0.8 and ResponseAction.EMERGENCY_PROTOCOL in allowed_actions:
            return ResponseAction.EMERGENCY_PROTOCOL
        elif risk_score >= 0.7 and ResponseAction.QUARANTINE in allowed_actions:
            return ResponseAction.QUARANTINE
        elif risk_score >= escalation_threshold and ResponseAction.ISOLATE in allowed_actions:
            return ResponseAction.ISOLATE
        elif risk_score >= 0.3 and ResponseAction.ALERT in allowed_actions:
            return ResponseAction.ALERT
        else:
            return ResponseAction.MONITOR
    
    def _apply_patient_safety_override(self, original_action: ResponseAction, 
                                     device_type: str, risk_score: float) -> ResponseAction:
        """Apply patient safety override for critical medical devices"""
        if device_type in ['ventilator', 'defibrillator']:
            # For life-critical devices, never use actions that could interrupt operation
            if original_action in [ResponseAction.ISOLATE, ResponseAction.QUARANTINE]:
                if risk_score > 0.7:
                    return ResponseAction.EMERGENCY_PROTOCOL
                else:
                    return ResponseAction.ALERT
        
        elif device_type in ['patient_monitor', 'infusion_pump']:
            # For monitoring devices, allow isolation but with enhanced monitoring
            if original_action == ResponseAction.QUARANTINE and risk_score < 0.8:
                return ResponseAction.ISOLATE
        
        return original_action
    
    def _generate_response_plan(self, action: ResponseAction, device_policy: Dict[str, Any], 
                              device_info: Dict[str, Any], threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed response plan with medical protocols"""
        plan = {
            'action': action.value,
            'device_id': device_info.get('id'),
            'device_type': device_info.get('device_type'),
            'timestamp': datetime.now().isoformat(),
            'estimated_downtime_ms': 0,
            'backup_protocols': device_policy.get('backup_protocols', []),
            'clinical_notifications': [],
            'technical_actions': [],
            'monitoring_enhancements': []
        }
        
        if action == ResponseAction.MONITOR:
            plan.update({
                'estimated_downtime_ms': 0,
                'technical_actions': ['increase_monitoring_frequency', 'log_detailed_traffic'],
                'monitoring_enhancements': ['enhanced_anomaly_detection', 'baseline_recalculation']
            })
        
        elif action == ResponseAction.ALERT:
            plan.update({
                'estimated_downtime_ms': 0,
                'clinical_notifications': ['notify_it_security', 'alert_device_technician'],
                'technical_actions': ['generate_security_report', 'enhance_logging'],
                'monitoring_enhancements': ['continuous_threat_monitoring']
            })
        
        elif action == ResponseAction.ISOLATE:
            estimated_downtime = min(device_policy['max_downtime_ms'], 30000)
            plan.update({
                'estimated_downtime_ms': estimated_downtime,
                'clinical_notifications': ['notify_clinical_staff', 'prepare_backup_device'],
                'technical_actions': ['network_isolation', 'forensic_data_collection'],
                'monitoring_enhancements': ['isolated_network_monitoring']
            })
        
        elif action == ResponseAction.QUARANTINE:
            plan.update({
                'estimated_downtime_ms': device_policy['max_downtime_ms'],
                'clinical_notifications': ['notify_clinical_manager', 'schedule_maintenance'],
                'technical_actions': ['full_network_quarantine', 'device_imaging', 'malware_scan'],
                'monitoring_enhancements': ['quarantine_zone_monitoring']
            })
        
        elif action == ResponseAction.EMERGENCY_PROTOCOL:
            plan.update({
                'estimated_downtime_ms': 0,
                'clinical_notifications': ['emergency_alert_clinical_team', 'activate_backup_systems'],
                'technical_actions': ['immediate_threat_containment', 'emergency_response_team'],
                'monitoring_enhancements': ['emergency_monitoring_protocol', 'realtime_threat_analysis'],
                'emergency_contacts': ['chief_medical_officer', 'it_security_manager', 'device_manufacturer']
            })
        
        return plan
    
    def _get_default_policy(self) -> Dict[str, Any]:
        """Get default policy for unknown device types"""
        return {
            'max_downtime_ms': 30000,
            'response_time_target_ms': 150,
            'escalation_threshold': 0.6,
            'allowed_actions': [ResponseAction.MONITOR, ResponseAction.ALERT, ResponseAction.ISOLATE],
            'patient_safety_override': False,
            'backup_protocols': ['standard_isolation', 'it_notification'],
            'priority_level': 'LOW',
            'requires_clinical_approval': True
        }
    
    def _log_response_decision(self, device_id: str, threat_data: Dict[str, Any], 
                             response_plan: Dict[str, Any], processing_time: float):
        """Log response decision for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'device_id': device_id,
            'threat_level': threat_data.get('threat_level'),
            'confidence': threat_data.get('confidence'),
            'response_action': response_plan['action'],
            'processing_time_ms': processing_time,
            'estimated_downtime_ms': response_plan['estimated_downtime_ms']
        }
        
        self.response_history.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.response_history) > 1000:
            self.response_history = self.response_history[-1000:]
    
    def get_response_statistics(self) -> Dict[str, Any]:
        """Get response engine performance statistics"""
        if not self.response_history:
            return {'status': 'no_data', 'total_responses': 0}
        
        recent_responses = self.response_history[-100:]  # Last 100 responses
        
        avg_processing_time = sum(r['processing_time_ms'] for r in recent_responses) / len(recent_responses)
        
        action_counts = {}
        for response in recent_responses:
            action = response['response_action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        threat_level_counts = {}
        for response in recent_responses:
            level = response['threat_level']
            threat_level_counts[level] = threat_level_counts.get(level, 0) + 1
        
        # Performance metrics
        fast_responses = sum(1 for r in recent_responses if r['processing_time_ms'] < 100)
        performance_rate = fast_responses / len(recent_responses)
        
        statistics = {
            'status': 'active',
            'total_responses': len(self.response_history),
            'recent_responses': len(recent_responses),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'performance_rate': round(performance_rate, 3),
            'response_distribution': action_counts,
            'threat_level_distribution': threat_level_counts,
            'meets_performance_target': performance_rate > 0.8 and avg_processing_time < 100
        }
        
        return statistics
    
    def simulate_threat_response(self, device_type: str, threat_level: str) -> Dict[str, Any]:
        """Simulate threat response for demonstration purposes"""
        # Generate simulated threat data
        threat_data = {
            'threat_level': threat_level,
            'confidence': 0.85 if threat_level == 'HIGH' else 0.6,
            'is_anomaly': True,
            'device_criticality': {
                'ventilator': 1.0,
                'defibrillator': 0.95,
                'patient_monitor': 0.8,
                'infusion_pump': 0.7,
                'ct_scanner': 0.5
            }.get(device_type, 0.5),
            'medical_context': {
                'emergency_mode': threat_level == 'HIGH',
                'device_type': device_type
            }
        }
        
        # Generate simulated device info
        device_info = {
            'id': f'SIM_{device_type}_{int(time.time())}',
            'device_type': device_type,
            'patient_connected': True,
            'status': 'active'
        }
        
        # Process threat
        response = self.analyze_threat_context(threat_data, device_info)
        
        return {
            'simulation': True,
            'threat_data': threat_data,
            'device_info': device_info,
            'response': response
        }