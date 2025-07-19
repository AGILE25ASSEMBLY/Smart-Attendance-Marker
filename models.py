from datetime import datetime, time
from app import db
from flask_dance.consumer.storage.sqla import OAuthConsumerMixin
from flask_login import UserMixin
from sqlalchemy import UniqueConstraint

# (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.String, primary_key=True)
    email = db.Column(db.String, unique=True, nullable=True)
    first_name = db.Column(db.String, nullable=True)
    last_name = db.Column(db.String, nullable=True)
    profile_image_url = db.Column(db.String, nullable=True)
    
    # Teacher-specific fields
    is_teacher = db.Column(db.Boolean, default=True, nullable=False)
    teacher_id = db.Column(db.String, unique=True, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    periods = db.relationship('Period', backref='teacher', lazy=True)

# (IMPORTANT) This table is mandatory for Replit Auth, don't drop it.
class OAuth(OAuthConsumerMixin, db.Model):
    user_id = db.Column(db.String, db.ForeignKey(User.id))
    browser_session_key = db.Column(db.String, nullable=False)
    user = db.relationship(User)

    __table_args__ = (UniqueConstraint(
        'user_id',
        'browser_session_key',
        'provider',
        name='uq_user_browser_session_key_provider',
    ),)

class Period(db.Model):
    __tablename__ = 'periods'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    teacher_id = db.Column(db.String, db.ForeignKey('users.id'), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    attendance_records = db.relationship('Attendance', backref='period', lazy=True)

class Student(db.Model):
    __tablename__ = 'students'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_number = db.Column(db.String(20), unique=True, nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), nullable=True)
    
    # Face recognition data (mock implementation)
    face_encoding = db.Column(db.Text, nullable=True)  # JSON string of face encoding
    image_path = db.Column(db.String(255), nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    attendance_records = db.relationship('Attendance', backref='student', lazy=True)

class Attendance(db.Model):
    __tablename__ = 'attendance'
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    period_id = db.Column(db.Integer, db.ForeignKey('periods.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=datetime.now().date)
    status = db.Column(db.String(10), nullable=False, default='absent')  # present, absent, late
    marked_at = db.Column(db.DateTime, default=datetime.now)
    marked_by_method = db.Column(db.String(50), default='manual')  # manual, face_recognition, face_recognition_enhanced, batch_face_recognition
    confidence_score = db.Column(db.Float, nullable=True)  # For face recognition confidence
    
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Unique constraint to prevent duplicate attendance for same student, period, and date
    __table_args__ = (UniqueConstraint('student_id', 'period_id', 'date', name='unique_attendance'),)
