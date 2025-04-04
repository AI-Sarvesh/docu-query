/**
 * Utility functions for the Document Q&A App
 */
const DocumentUtils = (function() {
    'use strict';

    // Format timestamps
    function formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString();
    }

    // Format file sizes
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        if (!unsafe) return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Format message with markdown-like syntax
    function formatMessage(message) {
        if (!message) return '';
        return message
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    // Generate a random session ID
    function generateSessionId() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Check if WebSocket is supported
    function isWebSocketSupported() {
        return 'WebSocket' in window;
    }

    // Create a debounced function
    function debounce(func, wait) {
        if (typeof func !== 'function') {
            throw new TypeError('Expected a function');
        }
        
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

    // Validate file type
    function isValidFileType(file, validTypes) {
        if (!file || !validTypes || !Array.isArray(validTypes)) {
            return false;
        }
        return validTypes.includes(file.type);
    }

    // Format error messages
    function formatErrorMessage(error) {
        if (typeof error === 'string') return error;
        if (error && error.message) return error.message;
        return 'An unknown error occurred';
    }

    // Throttle function to limit how often a function can be called
    function throttle(func, limit) {
        if (typeof func !== 'function') {
            throw new TypeError('Expected a function');
        }
        
        let inThrottle;
        return function(...args) {
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    // Helper function to render entity groups
    function renderEntityGroup(containerId, entities, className) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container element with ID "${containerId}" not found`);
            return;
        }
        
        container.innerHTML = '';
        
        if (!entities || entities.length === 0) {
            container.innerHTML = '<p class="text-sm text-gray-500">No entities detected in this document.</p>';
            return;
        }
        
        entities.forEach(entity => {
            const entityElement = document.createElement('span');
            entityElement.className = `entity-tag ${className || ''}`;
            entityElement.textContent = entity;
            container.appendChild(entityElement);
        });
    }

    // Show error message
    function showError(message, element) {
        if (element) {
            element.textContent = message;
            element.classList.add('text-red-500');
        }
        console.error(message);
    }

    // Show notification
    function showNotification(message, duration = 3000) {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification-toast');
        existingNotifications.forEach(notification => {
            notification.remove();
        });
        
        const notification = document.createElement('div');
        notification.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded shadow-lg notification-toast z-50';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Add fade-in effect
        notification.style.opacity = '0';
        notification.style.transition = 'opacity 0.3s ease';
        
        // Trigger reflow
        notification.offsetHeight;
        
        // Apply fade-in
        notification.style.opacity = '1';
        
        // Remove notification after specified duration with fade-out
        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, duration);
    }
    
    // Public API
    return {
        formatTimestamp,
        formatFileSize,
        escapeHtml,
        formatMessage,
        generateSessionId,
        isWebSocketSupported,
        debounce,
        throttle,
        isValidFileType,
        formatErrorMessage,
        renderEntityGroup,
        showError,
        showNotification
    };
})();

// Export utilities for backward compatibility
window.utils = DocumentUtils; 