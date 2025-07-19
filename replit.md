# Face Recognition Attendance System

## Overview

This is a Flask-based web application designed for educational institutions to manage student attendance using face recognition technology. The system allows teachers to create class periods, enroll students with their photos, and take attendance through automated face recognition.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: HTML templates using Jinja2 templating engine
- **UI Framework**: Bootstrap 5 with dark theme
- **JavaScript**: Vanilla JavaScript for client-side interactions
- **Icons**: Feather Icons for consistent iconography
- **Styling**: Custom CSS built on top of Bootstrap dark theme

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Database ORM**: SQLAlchemy with Flask-SQLAlchemy extension
- **Authentication**: Replit Auth integration with OAuth2
- **Session Management**: Flask sessions with permanent session configuration
- **File Handling**: Werkzeug for secure file uploads with PIL for image processing

### Database Architecture
- **ORM**: SQLAlchemy with declarative base model
- **Connection**: Environment-based database URL configuration
- **Connection Pooling**: Configured with pool recycling and pre-ping for reliability

## Key Components

### Authentication System
- **Replit OAuth Integration**: Custom OAuth2 blueprint for Replit authentication
- **User Management**: Flask-Login for session management
- **Database Storage**: Custom UserSessionStorage class for OAuth token persistence
- **User Model**: Mandatory User and OAuth tables for Replit Auth compatibility

### Core Models
- **User**: Teacher accounts with profile information and Replit Auth compatibility
- **Period**: Class periods with time schedules and teacher relationships
- **Student**: Student records with face encoding data (referenced but not fully shown)
- **Attendance**: Attendance records linking students to periods (referenced but not fully shown)

### Face Recognition System
- **Mock Implementation**: Currently uses mock face encoding based on image hashing
- **Image Processing**: PIL for image handling and validation
- **File Upload**: Secure filename handling with allowed extensions validation
- **Face Matching**: Mock distance-based matching algorithm (placeholder for real implementation)

### User Interface Components
- **Dashboard**: Teacher overview with statistics and quick actions
- **Period Management**: CRUD operations for class periods
- **Student Management**: Student enrollment with photo upload
- **Attendance Taking**: Face recognition interface for marking attendance
- **Reports**: Attendance analytics and CSV export functionality

## Data Flow

### Authentication Flow
1. User clicks login on index page
2. Redirected to Replit OAuth endpoint
3. OAuth callback creates/updates User record
4. Session established with Flask-Login
5. User redirected to dashboard

### Attendance Taking Flow
1. Teacher selects a period from attendance page
2. System displays enrolled students for that class
3. Camera interface captures student photos
4. Face recognition matches against stored encodings
5. Attendance status updated in database
6. Real-time feedback provided to teacher

### Student Enrollment Flow
1. Teacher navigates to student management
2. Form submission with student details and photo
3. Image validation and secure storage
4. Face encoding generation and storage
5. Student record created with period associations

## External Dependencies

### Python Packages
- **Flask**: Web framework and extensions (SQLAlchemy, Login)
- **Flask-Dance**: OAuth2 integration for Replit Auth
- **SQLAlchemy**: Database ORM and connection management
- **Werkzeug**: WSGI utilities and security helpers
- **PIL (Pillow)**: Image processing and validation
- **JWT**: Token handling for authentication

### Frontend Libraries
- **Bootstrap 5**: UI framework with dark theme
- **Feather Icons**: Icon library for consistent UI
- **JavaScript**: Native browser APIs for client-side functionality

### Infrastructure Dependencies
- **Database**: PostgreSQL or SQLite (environment-configured)
- **File Storage**: Local filesystem for uploaded images
- **Session Storage**: Server-side session management

## Deployment Strategy

### Environment Configuration
- **Database URL**: Environment variable for database connection
- **Session Secret**: Environment variable for session encryption
- **Upload Directory**: Configurable upload folder with size limits
- **Debug Mode**: Development/production mode switching

### Server Configuration
- **WSGI**: Flask application with ProxyFix middleware for HTTPS
- **File Uploads**: 16MB maximum file size limit
- **Database Pooling**: Connection recycling and health checks
- **Static Assets**: CSS and JavaScript served through Flask

### Production Considerations
- Application runs on host 0.0.0.0:5000 for containerized deployment
- ProxyFix middleware handles reverse proxy headers
- Database connection pooling for scalability
- Secure file upload handling with validation

## Recent Changes (July 19, 2025)

### Enhanced Face Recognition System
- **Enhanced Multi-Face Detection**: Implemented advanced face detection using both MediaPipe and OpenCV for comprehensive coverage
- **Batch Processing**: Added ability to process multiple images simultaneously for efficient attendance marking
- **Multiple Recognition Methods**: Integrated LBP, HOG, and statistical feature extraction with cosine similarity matching
- **Confidence Scoring**: Added detailed confidence tracking and adaptive thresholds for improved accuracy
- **Image Enhancement**: Implemented automatic image preprocessing for optimal face detection
- **Database Enhancements**: Added confidence_score field and enhanced marking methods tracking
- **Robust Error Handling**: Comprehensive fallback mechanisms and quality filtering
- **Batch Upload Interface**: New web interface for processing group photos and multiple student images

### Technical Improvements
- **Multi-Method Detection**: Up to 10 faces per image with overlap detection and quality filtering
- **Enhanced Recognition Pipeline**: Advanced feature extraction with multiple algorithms
- **Performance Optimization**: Efficient processing with progress tracking and statistics
- **User Experience**: Intuitive interfaces for both single and batch processing modes

### Face Recognition Implementation Note
The system now uses real computer vision technology with OpenCV and MediaPipe. Features include:
- Real-time face detection and recognition
- Multiple face processing in single images
- Advanced feature extraction algorithms
- Batch processing capabilities for group photos
- Confidence scoring and quality assessment
- Automatic image enhancement and preprocessing

## Deployment Configuration
The system is now ready for production deployment with comprehensive configuration files for multiple hosting platforms:
- **Render**: render.yaml with PostgreSQL auto-configuration
- **Railway**: railway.json with Nixpacks optimization
- **Heroku**: Procfile and app.json for easy deployment
- **Docker**: Dockerfile and docker-compose.yml for containerized deployment
- **DigitalOcean/Google Cloud**: Platform-specific configurations included

### Deployment Features
- Automated dependency installation with uv package manager
- System-level OpenCV and MediaPipe dependencies
- PostgreSQL database auto-configuration
- Environment variable management
- File upload handling with persistent storage options
- Production-optimized Gunicorn configuration
- Health checks and monitoring setup