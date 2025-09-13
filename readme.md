# Overview

MedGuard AI is an autonomous medical device security system that uses machine learning to detect cyber threats on medical devices while prioritizing patient safety. The system monitors network traffic from critical medical devices like ventilators, defibrillators, and patient monitors, using an Isolation Forest algorithm to identify anomalies and trigger appropriate security responses. It features a real-time web dashboard, automated threat response capabilities, and comprehensive logging for medical environments.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Web Dashboard**: Single-page HTML5 application with real-time updates
- **Client Framework**: Vanilla JavaScript with Socket.IO for WebSocket communication
- **UI Components**: Bootstrap-based responsive design with medical theme styling
- **Real-time Updates**: Socket.IO client for live threat alerts and system status updates

## Backend Architecture
- **Web Framework**: Flask application with RESTful API endpoints
- **Real-time Communication**: Flask-SocketIO for bidirectional WebSocket connections
- **Application Structure**: Modular architecture with separate packages for core logic, utilities, and web interface
- **Configuration Management**: Centralized configuration with environment variable support

## Machine Learning Engine
- **Anomaly Detection**: Isolation Forest algorithm (scikit-learn) for detecting network traffic anomalies
- **Feature Engineering**: 20+ network traffic features including packet analysis, protocol diversity, and security indicators
- **Model Persistence**: Joblib-based model serialization for training persistence
- **Device-Specific Profiles**: Customized baseline behavior profiles for different medical device types

## Data Storage
- **Primary Database**: SQLite for device inventory, threat detections, and system logs
- **Schema Design**: Relational structure with devices, threat_detections, and system_events tables
- **Data Models**: Device management with criticality levels, patient connection status, and security scores

## Security Response System
- **Response Engine**: Medical context-aware autonomous response with device-specific policies
- **Response Actions**: Graduated response levels from monitoring to emergency protocols
- **Patient Safety Override**: Zero-downtime policies for life-critical devices like ventilators
- **Response Time Targets**: Device-specific SLA targets (25ms for ventilators, 50ms for monitors)

## Device Management
- **Device Discovery**: Automated device inventory with manufacturer, model, and location tracking
- **Traffic Simulation**: Realistic network traffic generation for different medical device types
- **Baseline Learning**: Dynamic establishment of normal network behavior patterns
- **Criticality Weighting**: Device prioritization based on patient safety impact

# External Dependencies

## Core Python Libraries
- **Flask 2.3.3**: Web application framework
- **Flask-SocketIO 5.3.6**: Real-time WebSocket communication
- **Flask-CORS 4.0.0**: Cross-origin resource sharing support
- **scikit-learn 1.3.0**: Machine learning algorithms and preprocessing
- **pandas 2.0.3**: Data manipulation and analysis
- **numpy 1.24.3**: Numerical computing and array operations
- **joblib 1.3.2**: Model serialization and parallel processing

## Network and Messaging
- **python-socketio 5.8.0**: Server-side WebSocket implementation
- **eventlet 0.33.3**: Concurrent networking library for Socket.IO

## Frontend Libraries
- **Socket.IO Client 4.0.0**: Client-side WebSocket communication
- **Chart.js**: Data visualization for metrics and alerts
- **Bootstrap/Custom CSS**: Responsive UI framework with medical theme

## Database
- **SQLite**: Embedded relational database (no external server required)
- **Built-in Python sqlite3**: Database connectivity and operations

## Development and Deployment
- **Python 3.10+**: Runtime environment
- **File-based Configuration**: Environment variable and config file support
- **Logging**: Built-in Python logging with file and console output