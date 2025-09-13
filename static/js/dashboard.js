// MedGuard AI Dashboard JavaScript
class MedGuardDashboard {
    constructor() {
        this.socket = null;
        this.devices = [];
        this.alerts = [];
        this.activityLog = [];
        this.isConnected = false;
        
        this.init();
    }
    
    init() {
        this.initWebSocket();
        this.initEventListeners();
        this.loadInitialData();
        
        // Start periodic updates
        setInterval(() => this.updateMetrics(), 30000); // Every 30 seconds
    }
    
    // WebSocket initialization and management
    initWebSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.isConnected = true;
            this.updateConnectionStatus('online', 'Connected');
            this.addToLog('WebSocket connected to MedGuard AI');
        });
        
        this.socket.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus('offline', 'Disconnected');
            this.addToLog('WebSocket disconnected');
        });
        
        this.socket.on('threat_alert', (data) => {
            this.handleThreatAlert(data);
        });
        
        this.socket.on('system_update', (data) => {
            this.handleSystemUpdate(data);
        });
        
        this.socket.on('device_status', (data) => {
            this.handleDeviceUpdate(data);
        });
        
        this.socket.on('connection_status', (data) => {
            this.addToLog(`Connection status: ${data.message}`);
        });
    }
    
    // Event listeners for UI interactions
    initEventListeners() {
        // Refresh buttons
        document.getElementById('refreshDevices').addEventListener('click', () => {
            this.loadDevices();
        });
        
        document.getElementById('refreshData').addEventListener('click', () => {
            this.loadInitialData();
        });
        
        // Simulation controls
        document.getElementById('simulateThreat').addEventListener('click', () => {
            this.simulateThreat();
        });
        
        // Diagnostics
        document.getElementById('runDiagnostics').addEventListener('click', () => {
            this.runDiagnostics();
        });
        
        // Clear controls
        document.getElementById('clearAlerts').addEventListener('click', () => {
            this.clearAlerts();
        });
        
        document.getElementById('clearLog').addEventListener('click', () => {
            this.clearActivityLog();
        });
        
        // Modal close
        document.querySelector('.modal-close').addEventListener('click', () => {
            this.closeModal();
        });
        
        // Click outside modal to close
        document.getElementById('diagnosticsModal').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.closeModal();
            }
        });
    }
    
    // Data loading functions
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadSystemStatus(),
                this.loadDevices(),
                this.loadThreats()
            ]);
            
            this.addToLog('Initial data loaded successfully');
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.showToast('Failed to load initial data', 'error');
        }
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            this.updateSystemStatus(data);
        } catch (error) {
            console.error('Error loading system status:', error);
            this.updateSystemStatus({ status: 'ERROR' });
        }
    }
    
    async loadDevices() {
        try {
            const response = await fetch('/api/devices');
            const data = await response.json();
            
            this.devices = data.devices || [];
            this.updateDeviceList();
            this.populateDeviceSelect();
            
        } catch (error) {
            console.error('Error loading devices:', error);
            this.showToast('Failed to load devices', 'error');
        }
    }
    
    async loadThreats() {
        try {
            const response = await fetch('/api/threats?limit=20');
            const data = await response.json();
            
            if (data.threats) {
                this.alerts = data.threats;
                this.updateAlertsList();
            }
        } catch (error) {
            console.error('Error loading threats:', error);
        }
    }
    
    // UI update functions
    updateConnectionStatus(status, text) {
        const statusElement = document.getElementById('connectionStatus');
        statusElement.className = `status-indicator ${status}`;
        statusElement.querySelector('.status-text').textContent = text;
    }
    
    updateSystemStatus(data) {
        // System health
        const healthElement = document.getElementById('systemHealth');
        healthElement.textContent = data.status || 'UNKNOWN';
        healthElement.className = this.getStatusClass(data.status);
        
        // Device count
        document.getElementById('deviceCount').textContent = data.device_count || 0;
        
        // Threat count
        const threatCount = data.threat_activity?.recent_threats_count || 0;
        document.getElementById('threatCount').textContent = threatCount;
        
        // ML Accuracy
        const accuracy = data.ml_model?.accuracy || 0;
        document.getElementById('mlAccuracy').textContent = `${(accuracy * 100).toFixed(1)}%`;
        
        // Performance metrics
        const perfMetrics = data.performance_metrics || {};
        document.getElementById('avgResponseTime').textContent = 
            `${Math.round(perfMetrics.avg_analysis_time_ms || 0)}ms`;
        document.getElementById('targetPerformance').textContent = 
            `${Math.round((perfMetrics.target_performance_rate || 0) * 100)}%`;
        
        // Analysis speed (approximate)
        const analysisSpeed = Math.round(perfMetrics.recent_analyses_count / 5) || 0;
        document.getElementById('analysisSpeed').textContent = `${analysisSpeed}/min`;
    }
    
    updateDeviceList() {
        const deviceList = document.getElementById('deviceList');
        
        if (!this.devices || this.devices.length === 0) {
            deviceList.innerHTML = '<div class="loading">No devices found</div>';
            return;
        }
        
        const deviceHTML = this.devices.map(device => `
            <div class="device-item">
                <div class="device-info">
                    <h4>${device.device_type.toUpperCase()}</h4>
                    <p>${device.manufacturer} ${device.model} - ${device.location}</p>
                    <p>Security Score: ${device.security_score}/1.00</p>
                </div>
                <div class="device-status">
                    <span class="status-badge ${device.criticality_level.toLowerCase()}">${device.criticality_level}</span>
                    <span class="status-badge ${device.status}">${device.status}</span>
                    ${device.patient_connected ? '<span class="patient-indicator">👤</span>' : ''}
                </div>
            </div>
        `).join('');
        
        deviceList.innerHTML = deviceHTML;
    }
    
    populateDeviceSelect() {
        const deviceSelect = document.getElementById('deviceSelect');
        deviceSelect.innerHTML = '<option value="">Select a device...</option>';
        
        this.devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.id;
            option.textContent = `${device.device_type.toUpperCase()} - ${device.location}`;
            deviceSelect.appendChild(option);
        });
    }
    
    updateAlertsList() {
        const alertsList = document.getElementById('alertsList');
        
        if (!this.alerts || this.alerts.length === 0) {
            alertsList.innerHTML = '<div class="no-alerts">No active threats detected</div>';
            return;
        }
        
        const alertsHTML = this.alerts.map(alert => {
            const timestamp = new Date(alert.timestamp).toLocaleString();
            return `
                <div class="alert-item ${alert.threat_level.toLowerCase()}">
                    <div class="alert-header">
                        <span class="alert-title">
                            ${alert.threat_level} Threat Detected
                        </span>
                        <span class="alert-time">${timestamp}</span>
                    </div>
                    <div class="alert-message">
                        Device: ${alert.device_id} | Confidence: ${(alert.confidence * 100).toFixed(1)}%
                        ${alert.response_action ? ` | Action: ${alert.response_action}` : ''}
                    </div>
                </div>
            `;
        }).join('');
        
        alertsList.innerHTML = alertsHTML;
    }
    
    // WebSocket event handlers
    handleThreatAlert(data) {
        const alert = {
            id: Date.now(),
            device_id: data.device_id,
            device_type: data.device_type,
            threat_level: data.threat_level,
            confidence: data.confidence,
            timestamp: data.timestamp,
            simulation: data.simulation
        };
        
        this.alerts.unshift(alert);
        if (this.alerts.length > 50) {
            this.alerts = this.alerts.slice(0, 50);
        }
        
        this.updateAlertsList();
        
        const message = data.simulation 
            ? `Simulated ${data.threat_level} threat on ${data.device_type}`
            : `${data.threat_level} threat detected on ${data.device_type}`;
            
        this.addToLog(`THREAT ALERT: ${message} (${(data.confidence * 100).toFixed(1)}% confidence)`);
        this.showToast(message, data.threat_level.toLowerCase() === 'high' ? 'error' : 'warning');
        
        // Update threat count
        this.updateMetrics();
    }
    
    handleSystemUpdate(data) {
        if (data.error) {
            console.error('System update error:', data.error);
            return;
        }
        
        // Update system metrics
        if (data.performance_metrics) {
            const perfMetrics = data.performance_metrics;
            document.getElementById('avgResponseTime').textContent = 
                `${Math.round(perfMetrics.avg_analysis_time_ms || 0)}ms`;
        }
        
        if (data.system_status) {
            document.getElementById('systemHealth').textContent = data.system_status;
        }
        
        if (data.device_count !== undefined) {
            document.getElementById('deviceCount').textContent = data.device_count;
        }
        
        this.addToLog('System metrics updated');
    }
    
    handleDeviceUpdate(data) {
        if (data.error) {
            console.error('Device update error:', data.error);
            return;
        }
        
        if (data.devices) {
            this.devices = data.devices;
            this.updateDeviceList();
            this.populateDeviceSelect();
        } else if (data.device) {
            // Update single device
            const index = this.devices.findIndex(d => d.id === data.device_id);
            if (index !== -1) {
                this.devices[index] = data.device;
                this.updateDeviceList();
            }
        }
        
        this.addToLog('Device status updated');
    }
    
    // Action functions
    async simulateThreat() {
        const deviceSelect = document.getElementById('deviceSelect');
        const threatTypeSelect = document.getElementById('threatType');
        
        const deviceId = deviceSelect.value;
        const threatType = threatTypeSelect.value;
        
        if (!deviceId) {
            this.showToast('Please select a device', 'warning');
            return;
        }
        
        const button = document.getElementById('simulateThreat');
        button.disabled = true;
        button.textContent = '🔄 Simulating...';
        
        try {
            const response = await fetch('/api/simulate-threat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    device_id: deviceId,
                    threat_type: threatType
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.addToLog(`Threat simulation initiated: ${threatType} on ${deviceId}`);
                this.showToast(`${threatType} threat simulation started`, 'success');
            } else {
                this.showToast(`Simulation failed: ${data.error}`, 'error');
            }
            
        } catch (error) {
            console.error('Simulation error:', error);
            this.showToast('Failed to simulate threat', 'error');
        } finally {
            button.disabled = false;
            button.textContent = '🚨 Simulate Threat';
        }
    }
    
    async runDiagnostics() {
        const modal = document.getElementById('diagnosticsModal');
        const content = document.getElementById('diagnosticsContent');
        
        modal.classList.add('show');
        content.innerHTML = '<div class="loading">Running comprehensive diagnostics...</div>';
        
        try {
            const response = await fetch('/api/diagnostics');
            const data = await response.json();
            
            this.displayDiagnostics(data);
            this.addToLog('System diagnostics completed');
            
        } catch (error) {
            console.error('Diagnostics error:', error);
            content.innerHTML = `<div class="error">Failed to run diagnostics: ${error.message}</div>`;
        }
    }
    
    displayDiagnostics(data) {
        const content = document.getElementById('diagnosticsContent');
        
        const diagnosticsHTML = `
            <div class="diagnostics-results">
                <h4>System Status: ${data.system_status}</h4>
                
                <h5>ML Model Performance:</h5>
                <ul>
                    <li>Status: ${data.ml_model?.status || 'Unknown'}</li>
                    <li>Accuracy: ${((data.ml_model?.accuracy || 0) * 100).toFixed(1)}%</li>
                    <li>Prediction Time: ${data.ml_model?.avg_prediction_time_ms || 0}ms</li>
                    <li>Features: ${data.ml_model?.features_count || 0}</li>
                </ul>
                
                <h5>System Performance:</h5>
                <ul>
                    <li>Total Analyses: ${data.performance_metrics?.total_analyses || 0}</li>
                    <li>Avg Analysis Time: ${Math.round(data.performance_metrics?.avg_analysis_time_ms || 0)}ms</li>
                    <li>Target Performance Rate: ${Math.round((data.performance_metrics?.target_performance_rate || 0) * 100)}%</li>
                </ul>
                
                <h5>Device Inventory:</h5>
                <ul>
                    <li>Total Devices: ${data.device_inventory?.total_devices || 0}</li>
                    <li>Active Devices: ${data.device_inventory?.active_devices || 0}</li>
                    <li>Patient Connected: ${data.device_inventory?.patient_connected_devices || 0}</li>
                </ul>
                
                <h5>Threat Activity:</h5>
                <ul>
                    <li>Recent Threats: ${data.threat_activity?.recent_threats_count || 0}</li>
                    <li>Total Detected: ${data.threat_activity?.total_threats_detected || 0}</li>
                    <li>Detection Rate: ${Math.round((data.threat_activity?.threat_detection_rate || 0) * 100)}%</li>
                </ul>
                
                ${data.recommendations && data.recommendations.length > 0 ? `
                <h5>Recommendations:</h5>
                <ul>
                    ${data.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                </ul>
                ` : ''}
            </div>
        `;
        
        content.innerHTML = diagnosticsHTML;
    }
    
    // Utility functions
    updateMetrics() {
        if (this.isConnected) {
            this.socket.emit('request_system_update');
        }
    }
    
    addToLog(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = {
            timestamp,
            message
        };
        
        this.activityLog.unshift(logEntry);
        if (this.activityLog.length > 100) {
            this.activityLog = this.activityLog.slice(0, 100);
        }
        
        this.updateActivityLog();
    }
    
    updateActivityLog() {
        const logContainer = document.getElementById('activityLog');
        
        const logHTML = this.activityLog.map(entry => `
            <div class="log-entry">
                <span class="timestamp">[${entry.timestamp}]</span>
                <span class="message">${entry.message}</span>
            </div>
        `).join('');
        
        logContainer.innerHTML = logHTML;
    }
    
    clearAlerts() {
        this.alerts = [];
        this.updateAlertsList();
        this.addToLog('Security alerts cleared');
    }
    
    clearActivityLog() {
        this.activityLog = [];
        this.updateActivityLog();
    }
    
    closeModal() {
        document.getElementById('diagnosticsModal').classList.remove('show');
    }
    
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <strong>${type.toUpperCase()}:</strong> ${message}
            </div>
        `;
        
        const container = document.getElementById('toastContainer');
        container.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 5000);
    }
    
    getStatusClass(status) {
        switch (status?.toLowerCase()) {
            case 'healthy':
                return 'status-healthy';
            case 'warning':
                return 'status-warning';
            case 'error':
                return 'status-critical';
            default:
                return 'status-offline';
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MedGuardDashboard();
});