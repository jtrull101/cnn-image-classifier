/**
 * WebSocket Client with auto-reconnect and event handling
 */

class WebSocketClient {
    constructor(url) {
        this.url = url || this.getWebSocketURL();
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.pingInterval = null;
        this.isIntentionallyClosed = false;
        this.listeners = {};
        
        this.connect();
    }
    
    getWebSocketURL() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        return `${protocol}//${host}/ws`;
    }
    
    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.CONNECTING || this.ws.readyState === WebSocket.OPEN)) {
            console.log('WebSocket already connected or connecting');
            return;
        }
        
        console.log('Connecting to WebSocket:', this.url);
        this.isIntentionallyClosed = false;
        
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = (event) => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                this.updateConnectionStatus(true);
                this.startPingInterval();
                this.emit('connected', event);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    console.log('WebSocket message received:', data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('Error parsing WebSocket message:', e);
                }
            };
            
            this.ws.onerror = (event) => {
                console.error('WebSocket error:', event);
                this.emit('error', event);
            };
            
            this.ws.onclose = (event) => {
                console.log('WebSocket disconnected:', event);
                this.updateConnectionStatus(false);
                this.stopPingInterval();
                this.emit('disconnected', event);
                
                if (!this.isIntentionallyClosed) {
                    this.reconnect();
                }
            };
        } catch (e) {
            console.error('Error creating WebSocket:', e);
            this.reconnect();
        }
    }
    
    reconnect() {
        if (this.isIntentionallyClosed) {
            return;
        }
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnect attempts reached');
            window.showToast('WebSocket connection failed. Please refresh the page.', 'error');
            return;
        }
        
        this.reconnectAttempts++;
        const delay = Math.min(
            this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts),
            this.maxReconnectDelay
        );
        
        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (!this.isIntentionallyClosed) {
                this.connect();
            }
        }, delay);
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
            return true;
        } else {
            console.warn('WebSocket not connected, cannot send message');
            return false;
        }
    }
    
    close() {
        this.isIntentionallyClosed = true;
        this.stopPingInterval();
        if (this.ws) {
            this.ws.close();
        }
    }
    
    startPingInterval() {
        this.stopPingInterval();
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.send({ type: 'ping', timestamp: Date.now() });
            }
        }, 30000); // Ping every 30 seconds
    }
    
    stopPingInterval() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    handleMessage(data) {
        const { type, ...payload } = data;
        
        switch (type) {
            case 'pong':
                // Heartbeat response
                break;
                
            case 'model_loading':
                this.emit('model_loading', payload);
                if (payload.status === 'started') {
                    window.showToast(`Loading model: ${payload.model_name}`, 'info');
                } else if (payload.status === 'completed') {
                    window.showToast(`Model loaded: ${payload.model_name}`, 'success');
                } else if (payload.status === 'error') {
                    window.showToast(`Failed to load model: ${payload.error}`, 'error');
                }
                break;
                
            case 'prediction_progress':
                this.emit('prediction_progress', payload);
                break;
                
            case 'batch_progress':
                this.emit('batch_progress', payload);
                if (payload.completed && payload.total) {
                    const percent = Math.round((payload.completed / payload.total) * 100);
                    if (percent % 25 === 0) { // Show toast every 25%
                        window.showToast(`Batch progress: ${percent}%`, 'info', 2000);
                    }
                }
                break;
                
            case 'error':
                this.emit('error_message', payload);
                window.showToast(payload.message || 'An error occurred', 'error');
                break;
                
            default:
                // Custom event type
                this.emit(type, payload);
                break;
        }
    }
    
    updateConnectionStatus(connected) {
        // Update Alpine.js store (avoids relying on internal _x_dataStack API)
        if (window.Alpine) {
            try {
                Alpine.store('app').wsConnected = connected;
            } catch (_) {
                // Store not yet initialised — the body x-data getter will pick it up when ready
            }
        }
        // Dispatch a DOM event so any component can react without store coupling
        window.dispatchEvent(new CustomEvent('ws-status', { detail: { connected } }));
    }
    
    // Event listener management
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }
    
    off(event, callback) {
        if (!this.listeners[event]) return;
        
        if (callback) {
            this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
        } else {
            delete this.listeners[event];
        }
    }
    
    emit(event, data) {
        if (!this.listeners[event]) return;
        
        this.listeners[event].forEach(callback => {
            try {
                callback(data);
            } catch (e) {
                console.error(`Error in ${event} listener:`, e);
            }
        });
    }
}

// Initialize WebSocket client globally
let wsClient = null;

document.addEventListener('DOMContentLoaded', () => {
    wsClient = new WebSocketClient();
    
    // Make it globally accessible
    window.wsClient = wsClient;
    
    // Example: Listen for custom events
    // wsClient.on('model_loading', (data) => {
    //     console.log('Model loading event:', data);
    // });
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (wsClient) {
        wsClient.close();
    }
});
