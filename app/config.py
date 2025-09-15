import os

class Config:
    SECRET_KEY = os.environ.get('SESSION_SECRET') or 'medguard-ai-secret-key'
    DATABASE_PATH = 'data/medguard.db'
    ML_MODEL_PATH = 'data/models/anomaly_detector.joblib'
    LOG_PATH = 'data/logs/medguard.log'
    
    ANOMALY_THRESHOLD = 0.1
    MODEL_RETRAIN_INTERVAL = 3600  
    
    SUPPORTED_DEVICE_TYPES = [
        'ventilator', 'patient_monitor', 'infusion_pump', 
        'ct_scanner', 'defibrillator'
    ]
    

    RESPONSE_TIME_TARGETS = {
        'ventilator': 25,
        'patient_monitor': 50,
        'infusion_pump': 75,
        'ct_scanner': 100,
        'defibrillator': 25
    }
    
  
    SIMULATION_ENABLED = True
    DEMO_MODE = True