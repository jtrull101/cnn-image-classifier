/**
 * Main Alpine.js application and global configurations
 */

document.addEventListener('alpine:init', () => {
    // Global app state store
    Alpine.store('app', {
        loading: false,
        loadingMessage: 'Loading...',
        wsConnected: false,
        currentPrediction: null,
        
        setLoading(loading, message = 'Loading...') {
            this.loading = loading;
            this.loadingMessage = message;
        },
        
        setPrediction(prediction) {
            this.currentPrediction = prediction;
        },
        
        clearPrediction() {
            this.currentPrediction = null;
        }
    });
    
    // History management store
    Alpine.store('history', {
        local: [],
        remote: [],
        synced: false,
        
        init() {
            this.loadLocal();
        },
        
        loadLocal() {
            this.local = window.utils.storage.get('prediction_history', []);
        },
        
        addPrediction(prediction) {
            // Add to local storage
            this.local.unshift(prediction);
            
            // Keep only last 50 in local storage
            if (this.local.length > 50) {
                this.local = this.local.slice(0, 50);
            }
            
            window.utils.storage.set('prediction_history', this.local);
        },
        
        async syncWithServer() {
            try {
                // Get server predictions
                const response = await window.utils.api.get('/api/history?limit=100');
                this.remote = response.predictions || [];
                
                // Merge local and remote (deduplicate by image hash)
                const merged = new Map();
                
                this.remote.forEach(p => merged.set(p.image_hash || p.id, p));
                this.local.forEach(p => {
                    if (!merged.has(p.image_hash || p.id)) {
                        merged.set(p.image_hash || p.id, p);
                    }
                });
                
                this.local = Array.from(merged.values())
                    .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                    .slice(0, 50);
                
                window.utils.storage.set('prediction_history', this.local);
                this.synced = true;
                
                return true;
            } catch (e) {
                console.error('Error syncing history:', e);
                return false;
            }
        },
        
        clearLocal() {
            this.local = [];
            window.utils.storage.remove('prediction_history');
        }
    });
    
    // Batch processing store
    Alpine.store('batch', {
        queue: [],
        processing: false,
        currentIndex: 0,
        results: [],
        
        addFiles(files) {
            files.forEach((file, index) => {
                this.queue.push({
                    id: window.utils.generateId(),
                    file: file,
                    name: file.name,
                    size: file.size,
                    status: 'pending', // pending, processing, completed, error
                    progress: 0,
                    result: null,
                    error: null
                });
            });
        },
        
        clear() {
            this.queue = [];
            this.currentIndex = 0;
            this.results = [];
            this.processing = false;
        },
        
        removeFile(id) {
            this.queue = this.queue.filter(item => item.id !== id);
        },
        
        updateProgress(id, progress) {
            const item = this.queue.find(item => item.id === id);
            if (item) {
                item.progress = progress;
            }
        },
        
        setResult(id, result, error = null) {
            const item = this.queue.find(item => item.id === id);
            if (item) {
                item.status = error ? 'error' : 'completed';
                item.result = result;
                item.error = error;
                item.progress = 100;
            }
            
            if (!error) {
                this.results.push(result);
            }
        },
        
        getCompletedCount() {
            return this.queue.filter(item => item.status === 'completed').length;
        },
        
        getErrorCount() {
            return this.queue.filter(item => item.status === 'error').length;
        },
        
        getPendingCount() {
            return this.queue.filter(item => item.status === 'pending').length;
        }
    });
});

// Alpine.js components

// File upload component (enhanced)
function fileUploadComponent() {
    return {
        file: null,
        previewUrl: null,
        isDragging: false,
        isProcessing: false,
        error: null,
        
        handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                this.setFile(file);
            }
        },
        
        handleDrop(event) {
            this.isDragging = false;
            const file = event.dataTransfer.files[0];
            if (file) {
                this.setFile(file);
            }
        },
        
        setFile(file) {
            // Validate
            if (!window.utils.isValidImageFile(file)) {
                this.error = `Invalid file type: ${file.name}`;
                window.showToast(this.error, 'error');
                return;
            }
            
            if (!window.utils.isValidFileSize(file)) {
                this.error = `File too large: ${file.name} (max ${window.utils.formatBytes(window.utils.getMaxFileSize())})`;
                window.showToast(this.error, 'error');
                return;
            }
            
            this.file = file;
            this.error = null;
            
            // Create preview
            window.utils.createImagePreview(file)
                .then(preview => {
                    this.previewUrl = preview;
                })
                .catch(e => {
                    console.error('Error creating preview:', e);
                    this.error = 'Failed to create preview';
                });
        },
        
        clearFile() {
            this.file = null;
            this.previewUrl = null;
            this.error = null;
        },
        
        async predictImage() {
            if (!this.file || this.isProcessing) return;
            
            this.isProcessing = true;
            this.error = null;
            
            try {
                const formData = new FormData();
                formData.append('file', this.file);
                
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Prediction failed');
                }
                
                const result = await response.json();
                
                // Store in Alpine store
                Alpine.store('app').setPrediction(result);
                
                // Add to history
                const prediction = {
                    id: Date.now(),
                    timestamp: new Date().toISOString(),
                    image_name: this.file.name,
                    model_name: result.model_name,
                    predicted_class: result.class_name,
                    confidence: result.confidence,
                    probabilities: result.probabilities
                };
                Alpine.store('history').addPrediction(prediction);
                
                window.showToast('Classification complete!', 'success');
                
                // Trigger history update
                document.body.dispatchEvent(new CustomEvent('historyUpdate'));
                document.body.dispatchEvent(new CustomEvent('statsUpdate'));
                
            } catch (err) {
                this.error = err.message;
                window.showToast(err.message, 'error');
            } finally {
                this.isProcessing = false;
            }
        }
    };
}

// Prediction result component (enhanced)
function predictionResultComponent() {
    return {
        get prediction() {
            return Alpine.store('app').currentPrediction;
        },
        
        formatConfidence(value) {
            return window.utils.formatConfidence(value);
        },
        
        sortedProbabilities() {
            if (!this.prediction || !this.prediction.probabilities) return {};
            
            return Object.entries(this.prediction.probabilities)
                .sort((a, b) => b[1] - a[1])
                .reduce((acc, [key, val]) => {
                    acc[key] = val;
                    return acc;
                }, {});
        }
    };
}

// Model selector component (enhanced)
function modelSelectorComponent() {
    return {
        models: [],
        currentModel: null,
        loading: false,
        
        async init() {
            await this.loadModels();
            
            // Listen for WebSocket updates
            window.addEventListener('ws:model_loaded', () => {
                this.loadModels();
            });
        },
        
        async loadModels() {
            this.loading = true;
            try {
                const data = await window.utils.api.get('/api/models');
                this.models = data.models || [];
                this.currentModel = data.current_model;
            } catch (e) {
                console.error('Error loading models:', e);
                window.showToast('Failed to load models', 'error');
            } finally {
                this.loading = false;
            }
        },
        
        async selectModel(modelName) {
            if (modelName === this.currentModel) return;
            
            try {
                await window.utils.api.post(`/api/models/${modelName}/activate`);
                this.currentModel = modelName;
                window.showToast(`Switched to model: ${modelName}`, 'success');
                
                // Trigger update events
                document.body.dispatchEvent(new CustomEvent('modelChange', { 
                    detail: { modelName } 
                }));
            } catch (e) {
                console.error('Error switching model:', e);
                window.showToast('Failed to switch model', 'error');
            }
        }
    };
}

// Export components for use in templates
if (typeof window !== 'undefined') {
    window.alpineComponents = {
        fileUploadComponent,
        predictionResultComponent,
        modelSelectorComponent
    };
}

// Initialize history store on load
document.addEventListener('DOMContentLoaded', () => {
    if (window.Alpine && Alpine.store('history')) {
        Alpine.store('history').init();
    }
});
