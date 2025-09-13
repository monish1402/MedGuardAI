import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from core.anomaly_detector import MedGuardAnomalyDetector
from core.response_engine import MedGuardResponseEngine
from core.device_manager import MedGuardDeviceManager
from app.database import MedGuardDatabase
from utils.logger import medguard_logger

class MedGuardThreatAnalyzer:
    """Central threat analysis and coordination engine"""
    
    def __init__(self, database: MedGuardDatabase = None):
        self.db = database or MedGuardDatabase()
        self.anomaly_detector = MedGuardAnomalyDetector()
        self.response_engine = MedGuardResponseEngine()
        self.device_manager = MedGuardDeviceManager(self.db)
        
        self.analysis_history = []
        self.system_metrics = {
            'total_analyses': 0,
            'threats_detected': 0,
            'responses_triggered': 0,
            'avg_analysis_time_ms': 0,
            'system_health': 'HEALTHY'
        }
        
        # Initialize ML model
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the threat analysis system"""
        try:
            medguard_logger.info("Initializing MedGuard AI Threat Analysis System")
            
            # Train anomaly detector if needed
            if not self.anomaly_detector.is_trained:
                medguard_logger.info("Training anomaly detection model...")
                training_metrics = self.anomaly_detector.train()
                medguard_logger.info(f"Model training completed: {training_metrics}")
            
            # Log system initialization
            self.db.log_system_event({
                'event_type': 'SYSTEM_STARTUP',
                'details': {
                    'timestamp': datetime.now().isoformat(),
                    'components': ['anomaly_detector', 'response_engine', 'device_manager'],
                    'status': 'initialized'
                }
            })
            
            medguard_logger.info("MedGuard AI system initialization completed successfully")
            
        except Exception as e:
            medguard_logger.error(f"System initialization failed: {e}")
            self.system_metrics['system_health'] = 'ERROR'
    
    def analyze_device_threat(self, device_id: str, traffic_data: Dict[str, float] = None) -> Dict[str, Any]:
        """Comprehensive threat analysis for a medical device"""
        start_time = time.time()
        
        try:
            # Get device information
            device_info = self.device_manager.get_device(device_id)
            if not device_info:
                raise ValueError(f"Device {device_id} not found")
            
            # Generate or use provided traffic data
            if traffic_data is None:
                traffic_data = self.device_manager.generate_network_traffic(device_id)
            
            # Perform anomaly detection
            anomaly_result = self.anomaly_detector.predict(traffic_data)
            
            # Medical context analysis
            response_result = self.response_engine.analyze_threat_context(
                anomaly_result, device_info
            )
            
            # Calculate total analysis time
            analysis_time = (time.time() - start_time) * 1000
            
            # Compile comprehensive analysis result
            analysis_result = {
                'device_id': device_id,
                'device_info': device_info,
                'timestamp': datetime.now().isoformat(),
                'threat_analysis': {
                    'is_anomaly': anomaly_result['is_anomaly'],
                    'confidence': anomaly_result['confidence'],
                    'threat_level': anomaly_result['threat_level'],
                    'anomaly_score': anomaly_result['anomaly_score']
                },
                'medical_context': anomaly_result['medical_context'],
                'response_plan': response_result,
                'performance_metrics': {
                    'total_analysis_time_ms': analysis_time,
                    'ml_prediction_time_ms': anomaly_result['prediction_time_ms'],
                    'response_processing_time_ms': response_result['processing_time_ms'],
                    'meets_target_time': analysis_time < 150  # Overall target
                },
                'traffic_features': traffic_data
            }
            
            # Update system metrics
            self._update_system_metrics(analysis_result)
            
            # Log to database if threat detected
            if anomaly_result['is_anomaly']:
                self.db.log_threat_detection({
                    'device_id': device_id,
                    'timestamp': datetime.now(),
                    'threat_level': anomaly_result['threat_level'],
                    'confidence': anomaly_result['confidence'],
                    'features': traffic_data,
                    'response_action': response_result['response_action'],
                    'response_time_ms': int(analysis_time)
                })
                
                medguard_logger.warning(
                    f"Threat detected on {device_info.get('device_type')} {device_id}: "
                    f"{anomaly_result['threat_level']} ({anomaly_result['confidence']:.2f} confidence)"
                )
            
            # Store in analysis history
            self.analysis_history.append(analysis_result)
            
            # Keep history manageable
            if len(self.analysis_history) > 1000:
                self.analysis_history = self.analysis_history[-1000:]
            
            return analysis_result
            
        except Exception as e:
            error_result = {
                'device_id': device_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'analysis_failed': True
            }
            
            medguard_logger.error(f"Threat analysis failed for device {device_id}: {e}")
            return error_result
    
    def simulate_threat_scenario(self, device_id: str, threat_type: str) -> Dict[str, Any]:
        """Simulate threat scenario for demonstration purposes"""
        medguard_logger.info(f"Simulating {threat_type} threat on device {device_id}")
        
        try:
            # Get device info
            device_info = self.device_manager.get_device(device_id)
            if not device_info:
                raise ValueError(f"Device {device_id} not found")
            
            # Generate anomalous traffic
            anomalous_traffic = self.device_manager.simulate_anomalous_traffic(
                device_id, threat_type
            )
            
            # Analyze the simulated threat
            analysis_result = self.analyze_device_threat(device_id, anomalous_traffic)
            
            # Mark as simulation
            analysis_result['simulation'] = {
                'is_simulation': True,
                'threat_type': threat_type,
                'simulated_at': datetime.now().isoformat()
            }
            
            # Log simulation event
            self.db.log_system_event({
                'event_type': 'THREAT_SIMULATION',
                'device_id': device_id,
                'details': {
                    'threat_type': threat_type,
                    'result': analysis_result['threat_analysis']
                }
            })
            
            medguard_logger.info(f"Threat simulation completed: {threat_type} on {device_id}")
            
            return analysis_result
            
        except Exception as e:
            medguard_logger.error(f"Threat simulation failed: {e}")
            return {
                'simulation_failed': True,
                'error': str(e),
                'device_id': device_id,
                'threat_type': threat_type
            }
    
    def run_continuous_monitoring(self, duration_minutes: int = 60) -> List[Dict[str, Any]]:
        """Run continuous monitoring for specified duration"""
        medguard_logger.info(f"Starting continuous monitoring for {duration_minutes} minutes")
        
        monitoring_results = []
        devices = self.device_manager.get_devices(status='active')
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            for device in devices:
                try:
                    # Analyze each device
                    result = self.analyze_device_threat(device['id'])
                    monitoring_results.append(result)
                    
                    # Brief pause between analyses
                    time.sleep(1)
                    
                except Exception as e:
                    medguard_logger.error(f"Monitoring error for device {device['id']}: {e}")
            
            # Pause between monitoring cycles
            time.sleep(10)
        
        medguard_logger.info(f"Continuous monitoring completed. Analyzed {len(monitoring_results)} samples")
        return monitoring_results
    
    def get_system_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics"""
        try:
            # ML model diagnostics
            ml_diagnostics = self.anomaly_detector.get_model_diagnostics()
            
            # Response engine statistics
            response_stats = self.response_engine.get_response_statistics()
            
            # Device inventory statistics
            device_stats = self.device_manager.get_device_statistics()
            
            # Recent threat activity
            recent_threats = self.db.get_recent_threats(50)
            
            # System performance metrics
            recent_analyses = self.analysis_history[-100:] if self.analysis_history else []
            
            if recent_analyses:
                avg_analysis_time = sum(
                    a.get('performance_metrics', {}).get('total_analysis_time_ms', 0) 
                    for a in recent_analyses
                ) / len(recent_analyses)
                
                target_met_rate = sum(
                    1 for a in recent_analyses 
                    if a.get('performance_metrics', {}).get('meets_target_time', False)
                ) / len(recent_analyses)
            else:
                avg_analysis_time = 0
                target_met_rate = 0
            
            diagnostics = {
                'system_status': self.system_metrics['system_health'],
                'timestamp': datetime.now().isoformat(),
                'ml_model': ml_diagnostics,
                'response_engine': response_stats,
                'device_inventory': device_stats,
                'threat_activity': {
                    'recent_threats_count': len(recent_threats),
                    'total_threats_detected': self.system_metrics['threats_detected'],
                    'threat_detection_rate': len(recent_threats) / len(recent_analyses) if recent_analyses else 0
                },
                'performance_metrics': {
                    'total_analyses': self.system_metrics['total_analyses'],
                    'avg_analysis_time_ms': round(avg_analysis_time, 2),
                    'target_performance_rate': round(target_met_rate, 3),
                    'recent_analyses_count': len(recent_analyses),
                    'system_uptime_healthy': self.system_metrics['system_health'] == 'HEALTHY'
                },
                'recommendations': self._generate_system_recommendations(ml_diagnostics, response_stats)
            }
            
            return diagnostics
            
        except Exception as e:
            medguard_logger.error(f"System diagnostics failed: {e}")
            return {
                'system_status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_system_metrics(self, analysis_result: Dict[str, Any]):
        """Update system performance metrics"""
        self.system_metrics['total_analyses'] += 1
        
        if analysis_result.get('threat_analysis', {}).get('is_anomaly', False):
            self.system_metrics['threats_detected'] += 1
        
        if analysis_result.get('response_plan', {}).get('response_action') != 'monitor':
            self.system_metrics['responses_triggered'] += 1
        
        # Update average analysis time
        current_time = analysis_result.get('performance_metrics', {}).get('total_analysis_time_ms', 0)
        current_avg = self.system_metrics['avg_analysis_time_ms']
        total_analyses = self.system_metrics['total_analyses']
        
        self.system_metrics['avg_analysis_time_ms'] = (
            (current_avg * (total_analyses - 1) + current_time) / total_analyses
        )
    
    def _generate_system_recommendations(self, ml_diagnostics: Dict[str, Any], 
                                       response_stats: Dict[str, Any]) -> List[str]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        # ML model recommendations
        if ml_diagnostics.get('accuracy', 0) < 0.9:
            recommendations.append("Consider retraining ML model with more diverse data")
        
        if ml_diagnostics.get('avg_prediction_time_ms', 0) > 50:
            recommendations.append("ML prediction time is high - consider model optimization")
        
        # Response engine recommendations
        if response_stats.get('avg_processing_time_ms', 0) > 75:
            recommendations.append("Response processing time is above target - review response policies")
        
        # System performance recommendations
        if self.system_metrics['avg_analysis_time_ms'] > 100:
            recommendations.append("Overall analysis time exceeds target - system optimization needed")
        
        # Threat detection recommendations
        threat_rate = self.system_metrics['threats_detected'] / max(1, self.system_metrics['total_analyses'])
        if threat_rate > 0.1:
            recommendations.append("High threat detection rate - review network security posture")
        
        if not recommendations:
            recommendations.append("System is performing within optimal parameters")
        
        return recommendations