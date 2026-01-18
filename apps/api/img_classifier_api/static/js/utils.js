/**
 * Utility functions for the Image Classification UI
 */

// Format bytes to human-readable string
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Format date to readable string
function formatDate(date) {
    const d = new Date(date);
    return d.toLocaleDateString() + ' ' + d.toLocaleTimeString();
}

// Format confidence percentage
function formatConfidence(confidence) {
    return (confidence * 100).toFixed(2) + '%';
}

// Validate image file
function isValidImageFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/jpg', 'image/gif', 'image/bmp'];
    return file && validTypes.includes(file.type);
}

// Get max file size in bytes (default 10MB)
function getMaxFileSize() {
    return 10 * 1024 * 1024;
}

// Validate file size
function isValidFileSize(file, maxSize = getMaxFileSize()) {
    return file && file.size <= maxSize;
}

// Create image preview URL
function createImagePreview(file) {
    return new Promise((resolve, reject) => {
        if (!isValidImageFile(file)) {
            reject(new Error('Invalid image file'));
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsDataURL(file);
    });
}

// Hash string (simple djb2 algorithm for client-side hashing)
function hashString(str) {
    let hash = 5381;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) + hash) + str.charCodeAt(i);
    }
    return hash >>> 0; // Convert to unsigned 32-bit integer
}

// Generate unique ID
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Local storage helpers
const storage = {
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Error reading from localStorage:', e);
            return defaultValue;
        }
    },
    
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (e) {
            console.error('Error writing to localStorage:', e);
            return false;
        }
    },
    
    remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Error removing from localStorage:', e);
            return false;
        }
    },
    
    clear() {
        try {
            localStorage.clear();
            return true;
        } catch (e) {
            console.error('Error clearing localStorage:', e);
            return false;
        }
    },
    
    // Get storage usage
    getUsage() {
        let total = 0;
        for (let key in localStorage) {
            if (localStorage.hasOwnProperty(key)) {
                total += localStorage[key].length + key.length;
            }
        }
        return total;
    }
};

// API helpers
const api = {
    async get(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error('API GET error:', e);
            throw e;
        }
    },
    
    async post(url, data, isFormData = false) {
        try {
            const options = {
                method: 'POST',
                body: isFormData ? data : JSON.stringify(data)
            };
            
            if (!isFormData) {
                options.headers = {
                    'Content-Type': 'application/json'
                };
            }
            
            const response = await fetch(url, options);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error('API POST error:', e);
            throw e;
        }
    },
    
    async delete(url) {
        try {
            const response = await fetch(url, { method: 'DELETE' });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (e) {
            console.error('API DELETE error:', e);
            throw e;
        }
    }
};

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.utils = {
        formatBytes,
        formatDate,
        formatConfidence,
        isValidImageFile,
        isValidFileSize,
        getMaxFileSize,
        createImagePreview,
        hashString,
        generateId,
        debounce,
        throttle,
        storage,
        api
    };
}
