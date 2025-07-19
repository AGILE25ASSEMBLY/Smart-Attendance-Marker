/**
 * Face Recognition Attendance System - Client-side JavaScript
 * Handles UI interactions, form validation, and AJAX requests
 */

(function() {
    'use strict';

    // Initialize the application when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        initializeApp();
    });

    function initializeApp() {
        initializeTooltips();
        initializeFormValidation();
        initializeImageUpload();
        initializeSearchFunctionality();
        initializeAttendanceSystem();
        initializeModalHandlers();
        initializePrintFunctionality();
    }

    /**
     * Initialize Bootstrap tooltips
     */
    function initializeTooltips() {
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltipTriggerList.forEach(function(tooltipTriggerEl) {
            new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    /**
     * Enhanced form validation
     */
    function initializeFormValidation() {
        const forms = document.querySelectorAll('.needs-validation');
        
        forms.forEach(function(form) {
            form.addEventListener('submit', function(event) {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });

        // Time validation for period forms
        const startTimeInput = document.getElementById('start_time');
        const endTimeInput = document.getElementById('end_time');
        
        if (startTimeInput && endTimeInput) {
            endTimeInput.addEventListener('change', function() {
                validateTimeRange(startTimeInput.value, this.value);
            });
        }

        // Roll number uniqueness check
        const rollNumberInput = document.getElementById('roll_number');
        if (rollNumberInput) {
            rollNumberInput.addEventListener('blur', function() {
                // In a real implementation, this would make an API call to check uniqueness
                // For now, we'll just ensure it's not empty
                if (this.value.trim() === '') {
                    this.setCustomValidity('Roll number is required');
                } else {
                    this.setCustomValidity('');
                }
            });
        }
    }

    /**
     * Validate time range for periods
     */
    function validateTimeRange(startTime, endTime) {
        const endTimeInput = document.getElementById('end_time');
        
        if (startTime && endTime && startTime >= endTime) {
            endTimeInput.setCustomValidity('End time must be after start time');
            showAlert('End time must be after start time', 'warning');
        } else {
            endTimeInput.setCustomValidity('');
        }
    }

    /**
     * Image upload and preview functionality
     */
    function initializeImageUpload() {
        const imageInput = document.getElementById('image');
        const imagePreview = document.getElementById('imagePreview');
        const previewImg = document.getElementById('previewImg');

        if (imageInput) {
            imageInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                
                if (file) {
                    // Validate file size (16MB max)
                    const maxSize = 16 * 1024 * 1024;
                    if (file.size > maxSize) {
                        showAlert('File size must be less than 16MB', 'error');
                        this.value = '';
                        if (imagePreview) imagePreview.style.display = 'none';
                        return;
                    }

                    // Validate file type
                    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif'];
                    if (!allowedTypes.includes(file.type)) {
                        showAlert('Please select a valid image file (JPG, PNG, or GIF)', 'error');
                        this.value = '';
                        if (imagePreview) imagePreview.style.display = 'none';
                        return;
                    }

                    // Show preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        if (previewImg && imagePreview) {
                            previewImg.src = e.target.result;
                            imagePreview.style.display = 'block';
                        }
                    };
                    reader.readAsDataURL(file);
                }
            });
        }
    }

    /**
     * Search functionality for tables
     */
    function initializeSearchFunctionality() {
        const searchInputs = document.querySelectorAll('[data-search-target]');
        
        searchInputs.forEach(function(input) {
            const targetSelector = input.getAttribute('data-search-target');
            const targetTable = document.querySelector(targetSelector);
            
            if (targetTable) {
                input.addEventListener('keyup', function() {
                    filterTable(this.value, targetTable);
                });
            }
        });

        // Generic search input handler
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('keyup', function() {
                const searchTerm = this.value.toLowerCase();
                const tableRows = document.querySelectorAll('#studentsTable tbody tr, .list-group-item');
                
                tableRows.forEach(function(row) {
                    const text = row.textContent.toLowerCase();
                    row.style.display = text.includes(searchTerm) ? '' : 'none';
                });
            });
        }
    }

    /**
     * Filter table rows based on search term
     */
    function filterTable(searchTerm, table) {
        const rows = table.querySelectorAll('tbody tr');
        const term = searchTerm.toLowerCase();
        
        rows.forEach(function(row) {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(term) ? '' : 'none';
        });
    }

    /**
     * Attendance system functionality
     */
    function initializeAttendanceSystem() {
        // Auto-refresh attendance status
        if (window.location.pathname.includes('/attendance/')) {
            setInterval(function() {
                refreshAttendanceStats();
            }, 30000); // Refresh every 30 seconds
        }

        // Bulk attendance actions
        window.markAllPresent = function() {
            const students = document.querySelectorAll('[data-student-id]');
            students.forEach(function(element) {
                const studentId = element.getAttribute('data-student-id');
                markAttendance(studentId, 'present', true);
            });
        };

        window.markAllAbsent = function() {
            const students = document.querySelectorAll('[data-student-id]');
            students.forEach(function(element) {
                const studentId = element.getAttribute('data-student-id');
                markAttendance(studentId, 'absent', true);
            });
        };
    }

    /**
     * Mark attendance for a student
     */
    window.markAttendance = function(studentId, status, skipRefresh = false) {
        const periodId = getCurrentPeriodId();
        if (!periodId) return;

        const formData = new FormData();
        formData.append('student_id', studentId);
        formData.append('status', status);
        formData.append('method', 'manual');

        // Show loading state
        const statusElement = document.getElementById(`status-${studentId}`);
        const originalContent = statusElement ? statusElement.innerHTML : '';
        if (statusElement) {
            statusElement.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span>';
        }

        fetch(`/attendance/${periodId}/mark`, {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.text();
        })
        .then(() => {
            if (!skipRefresh) {
                updateStudentStatus(studentId, status, 'manual');
                showAlert(`Attendance marked successfully`, 'success');
            }
        })
        .catch(error => {
            console.error('Error marking attendance:', error);
            if (statusElement) {
                statusElement.innerHTML = originalContent;
            }
            showAlert('Error marking attendance. Please try again.', 'error');
        });
    };

    /**
     * Update student status in UI
     */
    window.updateStudentStatus = function(studentId, status, method) {
        const statusElement = document.getElementById(`status-${studentId}`);
        const methodElement = document.getElementById(`method-${studentId}`);
        
        if (statusElement) {
            const badgeClass = status === 'present' ? 'bg-success' : 'bg-warning';
            const icon = status === 'present' ? 'check' : 'x';
            const text = status === 'present' ? 'Present' : 'Absent';
            
            statusElement.innerHTML = `<span class="badge ${badgeClass}">
                <i data-feather="${icon}" style="width: 12px; height: 12px;"></i> ${text}
            </span>`;
        }
        
        if (methodElement) {
            const methodIcon = method === 'face_recognition' ? 'camera' : 'user';
            const methodText = method === 'face_recognition' ? 'Auto' : 'Manual';
            const methodClass = method === 'face_recognition' ? 'text-primary' : 'text-muted';
            
            methodElement.innerHTML = `<small class="${methodClass}">
                <i data-feather="${methodIcon}" style="width: 12px; height: 12px;"></i> ${methodText}
            </small>`;
        }
        
        // Re-initialize feather icons
        if (typeof feather !== 'undefined') {
            feather.replace();
        }
    };

    /**
     * Get current period ID from URL
     */
    function getCurrentPeriodId() {
        const match = window.location.pathname.match(/\/attendance\/(\d+)/);
        return match ? match[1] : null;
    }

    /**
     * Refresh attendance statistics
     */
    function refreshAttendanceStats() {
        const periodId = getCurrentPeriodId();
        if (!periodId) return;

        fetch(`/attendance/${periodId}/stats`, {
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())
        .then(data => {
            updateStatsDisplay(data);
        })
        .catch(error => {
            console.error('Error refreshing stats:', error);
        });
    }

    /**
     * Update statistics display
     */
    function updateStatsDisplay(stats) {
        const presentElement = document.querySelector('[data-stat="present"]');
        const absentElement = document.querySelector('[data-stat="absent"]');
        
        if (presentElement) presentElement.textContent = stats.present || 0;
        if (absentElement) absentElement.textContent = stats.absent || 0;
    }

    /**
     * Initialize modal handlers
     */
    function initializeModalHandlers() {
        // Auto-focus first input in modals
        const modals = document.querySelectorAll('.modal');
        modals.forEach(function(modal) {
            modal.addEventListener('shown.bs.modal', function() {
                const firstInput = modal.querySelector('input, textarea, select');
                if (firstInput) firstInput.focus();
            });
        });
    }

    /**
     * Initialize print functionality
     */
    function initializePrintFunctionality() {
        window.printReport = function() {
            window.print();
        };

        // Add print styles when printing
        window.addEventListener('beforeprint', function() {
            document.body.classList.add('printing');
        });

        window.addEventListener('afterprint', function() {
            document.body.classList.remove('printing');
        });
    }

    /**
     * Show alert messages
     */
    function showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        // Find or create alerts container
        let alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) {
            alertsContainer = document.createElement('div');
            alertsContainer.id = 'alerts-container';
            alertsContainer.className = 'container mt-3';
            
            const main = document.querySelector('main');
            if (main) {
                main.insertBefore(alertsContainer, main.firstChild);
            } else {
                document.body.insertBefore(alertsContainer, document.body.firstChild);
            }
        }

        // Add alert to container
        alertsContainer.appendChild(alertDiv);

        // Auto-dismiss after 5 seconds
        setTimeout(function() {
            if (alertDiv.parentNode) {
                alertDiv.classList.remove('show');
                setTimeout(function() {
                    if (alertDiv.parentNode) {
                        alertDiv.parentNode.removeChild(alertDiv);
                    }
                }, 150);
            }
        }, 5000);
    }

    /**
     * Utility function to format dates
     */
    window.formatDate = function(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    /**
     * Utility function to format times
     */
    window.formatTime = function(timeString) {
        const time = new Date(`2000-01-01T${timeString}`);
        return time.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
    };

    /**
     * Handle network errors gracefully
     */
    window.addEventListener('online', function() {
        showAlert('Connection restored', 'success');
    });

    window.addEventListener('offline', function() {
        showAlert('Connection lost. Some features may not work.', 'warning');
    });

    /**
     * Keyboard shortcuts
     */
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + S to save forms
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            const form = document.querySelector('form');
            if (form) {
                e.preventDefault();
                form.dispatchEvent(new Event('submit'));
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const modal = bootstrap.Modal.getInstance(openModal);
                if (modal) modal.hide();
            }
        }
    });

    /**
     * Enhanced error handling
     */
    window.onerror = function(msg, url, line, col, error) {
        console.error('JavaScript error:', {
            message: msg,
            source: url,
            line: line,
            column: col,
            error: error
        });
        
        // Only show user-friendly errors in production
        if (window.location.hostname !== 'localhost') {
            showAlert('An unexpected error occurred. Please refresh the page.', 'error');
        }
    };

    // Export utility functions for global access
    window.AttendanceApp = {
        showAlert: showAlert,
        formatDate: window.formatDate,
        formatTime: window.formatTime,
        markAttendance: window.markAttendance,
        updateStudentStatus: window.updateStudentStatus
    };

})();
