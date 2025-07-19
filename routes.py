import os
import json
import hashlib
import csv
import io
from datetime import datetime, date, time
from flask import render_template, request, redirect, url_for, flash, session, jsonify, make_response
from flask_login import current_user
from werkzeug.utils import secure_filename
from PIL import Image

from app import app, db
from models import User, Period, Student, Attendance
from replit_auth import require_login, make_replit_blueprint
from face_recognition_engine import face_engine
from face_recognition_enhanced import enhanced_face_engine

app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")

# Make session permanent
@app.before_request
def make_session_permanent():
    session.permanent = True

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def real_face_encoding(image_path):
    """Generate real face encoding using OpenCV and MediaPipe"""
    try:
        encoding = face_engine.extract_face_encoding(image_path)
        if encoding is not None:
            # Convert numpy array to list for JSON serialization
            return encoding.tolist()
        return None
    except Exception as e:
        print(f"Error generating face encoding: {e}")
        return None

def real_face_match(encoding1, encoding2, threshold=0.15):
    """Real face matching using OpenCV correlation"""
    try:
        if not encoding1 or not encoding2:
            return False
        
        import numpy as np
        enc1 = np.array(encoding1) if isinstance(encoding1, list) else encoding1
        enc2 = np.array(encoding2) if isinstance(encoding2, list) else encoding2
        
        return face_engine.compare_faces(enc1, enc2, threshold)
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
@require_login
def dashboard():
    teacher = current_user
    
    # Get today's periods
    today_periods = Period.query.filter_by(teacher_id=teacher.id).all()
    
    # Get today's attendance stats
    today = date.today()
    attendance_stats = {}
    
    for period in today_periods:
        students_in_class = Student.query.filter_by(class_name=period.class_name).all()
        total_students = len(students_in_class)
        
        present_count = db.session.query(Attendance).filter_by(
            period_id=period.id,
            date=today,
            status='present'
        ).count()
        
        attendance_stats[period.id] = {
            'total': total_students,
            'present': present_count,
            'absent': total_students - present_count
        }
    
    return render_template('dashboard.html', 
                         teacher=teacher, 
                         periods=today_periods, 
                         attendance_stats=attendance_stats)

@app.route('/periods')
@require_login
def periods():
    teacher_periods = Period.query.filter_by(teacher_id=current_user.id).all()
    return render_template('periods.html', periods=teacher_periods)

@app.route('/periods/new', methods=['GET', 'POST'])
@require_login
def new_period():
    if request.method == 'POST':
        period = Period(
            name=request.form['name'],
            subject=request.form['subject'],
            start_time=datetime.strptime(request.form['start_time'], '%H:%M').time(),
            end_time=datetime.strptime(request.form['end_time'], '%H:%M').time(),
            class_name=request.form['class_name'],
            teacher_id=current_user.id
        )
        
        db.session.add(period)
        db.session.commit()
        flash('Period created successfully!', 'success')
        return redirect(url_for('periods'))
    
    return render_template('period_form.html', period=None)

@app.route('/periods/<int:period_id>/edit', methods=['GET', 'POST'])
@require_login
def edit_period(period_id):
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only edit your own periods.', 'error')
        return redirect(url_for('periods'))
    
    if request.method == 'POST':
        period.name = request.form['name']
        period.subject = request.form['subject']
        period.start_time = datetime.strptime(request.form['start_time'], '%H:%M').time()
        period.end_time = datetime.strptime(request.form['end_time'], '%H:%M').time()
        period.class_name = request.form['class_name']
        
        db.session.commit()
        flash('Period updated successfully!', 'success')
        return redirect(url_for('periods'))
    
    return render_template('period_form.html', period=period)

@app.route('/periods/<int:period_id>/delete', methods=['POST'])
@require_login
def delete_period(period_id):
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only delete your own periods.', 'error')
        return redirect(url_for('periods'))
    
    db.session.delete(period)
    db.session.commit()
    flash('Period deleted successfully!', 'success')
    return redirect(url_for('periods'))

@app.route('/students')
@require_login
def students():
    # Get all classes taught by this teacher
    teacher_classes = db.session.query(Period.class_name).filter_by(teacher_id=current_user.id).distinct().all()
    class_names = [cls[0] for cls in teacher_classes]
    
    # Get students from those classes
    students_list = Student.query.filter(Student.class_name.in_(class_names)).all() if class_names else []
    
    return render_template('students.html', students=students_list)

@app.route('/students/new', methods=['GET', 'POST'])
@require_login
def new_student():
    if request.method == 'POST':
        # Handle file upload
        image_file = request.files.get('image')
        image_path = None
        face_encoding = None
        
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(f"{request.form['roll_number']}_{image_file.filename}")
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Create upload directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Save and process image
            image_file.save(image_path)
            
            # Generate real face encoding
            face_encoding = real_face_encoding(image_path)
        
        student = Student(
            name=request.form['name'],
            roll_number=request.form['roll_number'],
            class_name=request.form['class_name'],
            email=request.form.get('email'),
            image_path=image_path,
            face_encoding=json.dumps(face_encoding) if face_encoding else None
        )
        
        try:
            db.session.add(student)
            db.session.commit()
            flash('Student enrolled successfully!', 'success')
            return redirect(url_for('students'))
        except Exception as e:
            db.session.rollback()
            flash('Error enrolling student. Roll number might already exist.', 'error')
    
    # Get classes taught by this teacher
    teacher_classes = db.session.query(Period.class_name).filter_by(teacher_id=current_user.id).distinct().all()
    class_names = [cls[0] for cls in teacher_classes]
    
    return render_template('student_form.html', student=None, class_names=class_names)

@app.route('/attendance')
@require_login
def attendance():
    periods = Period.query.filter_by(teacher_id=current_user.id).all()
    return render_template('attendance.html', periods=periods)

@app.route('/attendance/<int:period_id>')
@require_login
def take_attendance(period_id):
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only take attendance for your own periods.', 'error')
        return redirect(url_for('attendance'))
    
    # Check if current time is within period time (simplified check)
    current_time = datetime.now().time()
    
    # Get students in this class
    students = Student.query.filter_by(class_name=period.class_name).all()
    
    # Get today's attendance for this period
    today = date.today()
    attendance_records = {}
    for student in students:
        record = Attendance.query.filter_by(
            student_id=student.id,
            period_id=period.id,
            date=today
        ).first()
        attendance_records[student.id] = record
    
    return render_template('take_attendance.html', 
                         period=period, 
                         students=students, 
                         attendance_records=attendance_records,
                         current_time=current_time)

@app.route('/attendance/<int:period_id>/webcam')
@require_login
def webcam_attendance(period_id):
    """Real-time webcam attendance interface"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only access your own periods.', 'error')
        return redirect(url_for('attendance'))
    
    # Get today's attendance stats
    students = Student.query.filter_by(class_name=period.class_name).all()
    today = date.today()
    
    present_count = db.session.query(Attendance).filter_by(
        period_id=period_id,
        date=today,
        status='present'
    ).count()
    
    return render_template('webcam_capture.html',
                         period=period,
                         total_students=len(students),
                         present_count=present_count)

@app.route('/attendance/<int:period_id>/stats')
@require_login
def attendance_stats(period_id):
    """Get current attendance statistics"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    today = date.today()
    
    present_count = db.session.query(Attendance).filter_by(
        period_id=period_id,
        date=today,
        status='present'
    ).count()
    
    absent_count = db.session.query(Attendance).filter_by(
        period_id=period_id,
        date=today,
        status='absent'
    ).count()
    
    return jsonify({
        'present': present_count,
        'absent': absent_count,
        'total': present_count + absent_count
    })

@app.route('/attendance/<int:period_id>/mark', methods=['POST'])
@require_login
def mark_attendance(period_id):
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only mark attendance for your own periods.', 'error')
        return redirect(url_for('attendance'))
    
    student_id = request.form.get('student_id')
    status = request.form.get('status', 'present')
    method = request.form.get('method', 'manual')
    
    student = Student.query.get_or_404(student_id)
    today = date.today()
    
    # Check if attendance already exists
    existing = Attendance.query.filter_by(
        student_id=student_id,
        period_id=period_id,
        date=today
    ).first()
    
    if existing:
        existing.status = status
        existing.marked_by_method = method
        existing.marked_at = datetime.now()
    else:
        attendance = Attendance(
            student_id=student_id,
            period_id=period_id,
            date=today,
            status=status,
            marked_by_method=method
        )
        db.session.add(attendance)
    
    db.session.commit()
    flash(f'Attendance marked for {student.name}', 'success')
    return redirect(url_for('take_attendance', period_id=period_id))

@app.route('/attendance/<int:period_id>/face_recognition', methods=['POST'])
@require_login
def face_recognition_attendance(period_id):
    """Enhanced multi-face recognition attendance using webcam feed"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get image data from request (base64 encoded)
        image_data = request.form.get('image_data')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(image_data.split(',')[1])
        
        # Train enhanced recognizer if not already trained
        if not enhanced_face_engine.is_trained:
            train_enhanced_recognizer_for_class(period.class_name)
        
        # Recognize all faces in the image
        recognition_results = enhanced_face_engine.recognize_multiple_faces(image_bytes)
        
        today = date.today()
        marked_students = []
        failed_recognitions = []
        
        # Process each recognized face
        for result in recognition_results:
            student_id = result['student_id']
            confidence = result['confidence']
            
            if student_id and confidence > 0.75:  # High confidence threshold
                student = Student.query.get(student_id)
                if student:
                    # Check if attendance already exists
                    existing = Attendance.query.filter_by(
                        student_id=student.id,
                        period_id=period_id,
                        date=today
                    ).first()
                    
                    if not existing:
                        attendance = Attendance(
                            student_id=student.id,
                            period_id=period_id,
                            date=today,
                            status='present',
                            marked_by_method='face_recognition_enhanced',
                            confidence_score=confidence
                        )
                        db.session.add(attendance)
                        marked_students.append({
                            'name': student.name,
                            'confidence': confidence,
                            'bbox': result['bbox']
                        })
                    else:
                        failed_recognitions.append({
                            'reason': f'Already marked: {student.name}',
                            'bbox': result['bbox']
                        })
                else:
                    failed_recognitions.append({
                        'reason': 'Student not found in database',
                        'bbox': result['bbox']
                    })
            else:
                failed_recognitions.append({
                    'reason': f'Low confidence: {confidence:.2f}',
                    'bbox': result['bbox']
                })
        
        # Commit all changes
        if marked_students:
            db.session.commit()
        
        return jsonify({
            'success': len(marked_students) > 0,
            'marked_students': marked_students,
            'failed_recognitions': failed_recognitions,
            'total_faces_detected': len(recognition_results),
            'message': f'Marked attendance for {len(marked_students)} students. {len(failed_recognitions)} faces not processed.'
        })
        
    except Exception as e:
        app.logger.error(f"Enhanced face recognition error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Face recognition failed', 'details': str(e)}), 500

def train_face_recognizer_for_class(class_name):
    """Train the basic face recognizer with students from a specific class"""
    try:
        students = Student.query.filter_by(class_name=class_name).all()
        
        encodings = []
        labels = []
        
        for student in students:
            if student.face_encoding:
                encoding_data = json.loads(student.face_encoding)
                if encoding_data:
                    import numpy as np
                    encoding = np.array(encoding_data)
                    encodings.append(encoding)
                    labels.append(student.id)
        
        if encodings:
            face_engine.train_recognizer(encodings, labels)
            app.logger.info(f"Basic face recognizer trained for class {class_name} with {len(encodings)} faces")
        
    except Exception as e:
        app.logger.error(f"Error training basic face recognizer: {e}")

def train_enhanced_recognizer_for_class(class_name):
    """Train the enhanced face recognizer with multiple faces per student"""
    try:
        students = Student.query.filter_by(class_name=class_name).all()
        
        student_face_data = []
        
        for student in students:
            if student.face_encoding and student.image_path:
                # Extract multiple face encodings from the student's image
                face_encodings_data = enhanced_face_engine.extract_multiple_face_encodings(student.image_path)
                
                if face_encodings_data:
                    # Convert to the format expected by the trainer
                    encodings = [face_data['encoding'] for face_data in face_encodings_data]
                    
                    student_face_data.append({
                        'student_id': student.id,
                        'face_encodings': encodings
                    })
        
        if student_face_data:
            enhanced_face_engine.train_recognizer_with_multiple_faces(student_face_data)
            app.logger.info(f"Enhanced face recognizer trained for class {class_name} with {len(student_face_data)} students")
        else:
            app.logger.warning(f"No valid face data found for class {class_name}")
        
    except Exception as e:
        app.logger.error(f"Error training enhanced face recognizer: {e}")

@app.route('/attendance/<int:period_id>/batch_recognition', methods=['POST'])
@require_login
def batch_face_recognition(period_id):
    """Process multiple images for batch attendance marking"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get multiple image files from request
        uploaded_files = request.files.getlist('batch_images')
        
        if not uploaded_files:
            return jsonify({'error': 'No images provided'}), 400
        
        # Train enhanced recognizer if not already trained
        if not enhanced_face_engine.is_trained:
            train_enhanced_recognizer_for_class(period.class_name)
        
        today = date.today()
        batch_results = []
        total_marked = 0
        
        for i, image_file in enumerate(uploaded_files[:10]):  # Limit to 10 images
            if not allowed_file(image_file.filename):
                continue
            
            try:
                # Read image bytes
                image_bytes = image_file.read()
                
                # Recognize faces in this image
                recognition_results = enhanced_face_engine.recognize_multiple_faces(image_bytes)
                
                image_marked = []
                image_failed = []
                
                for result in recognition_results:
                    student_id = result['student_id']
                    confidence = result['confidence']
                    
                    if student_id and confidence > 0.75:
                        student = Student.query.get(student_id)
                        if student:
                            # Check if attendance already exists
                            existing = Attendance.query.filter_by(
                                student_id=student.id,
                                period_id=period_id,
                                date=today
                            ).first()
                            
                            if not existing:
                                attendance = Attendance(
                                    student_id=student.id,
                                    period_id=period_id,
                                    date=today,
                                    status='present',
                                    marked_by_method='batch_face_recognition',
                                    confidence_score=confidence
                                )
                                db.session.add(attendance)
                                image_marked.append(student.name)
                                total_marked += 1
                            else:
                                image_failed.append(f'{student.name} (already marked)')
                        else:
                            image_failed.append('Unknown student')
                    else:
                        image_failed.append('Low confidence face')
                
                batch_results.append({
                    'image_index': i,
                    'filename': image_file.filename,
                    'marked_students': image_marked,
                    'failed_recognitions': image_failed,
                    'faces_detected': len(recognition_results)
                })
                
            except Exception as e:
                batch_results.append({
                    'image_index': i,
                    'filename': image_file.filename,
                    'error': str(e)
                })
        
        # Commit all changes
        if total_marked > 0:
            db.session.commit()
        
        return jsonify({
            'success': True,
            'total_marked': total_marked,
            'images_processed': len(batch_results),
            'results': batch_results
        })
        
    except Exception as e:
        app.logger.error(f"Batch recognition error: {e}")
        db.session.rollback()
        return jsonify({'error': 'Batch recognition failed', 'details': str(e)}), 500

@app.route('/attendance/<int:period_id>/batch_upload')
@require_login
def batch_upload_page(period_id):
    """Batch upload interface for multiple image processing"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only access your own periods.', 'error')
        return redirect(url_for('attendance'))
    
    return render_template('batch_upload.html', period=period)

@app.route('/enhanced-features')
@require_login
def enhanced_features():
    """Display enhanced features page"""
    return render_template('enhanced_features.html')

@app.route('/reports')
@require_login
def reports():
    periods = Period.query.filter_by(teacher_id=current_user.id).all()
    
    # Get attendance data for reports
    report_data = {}
    for period in periods:
        students = Student.query.filter_by(class_name=period.class_name).all()
        attendance_records = db.session.query(Attendance).filter_by(period_id=period.id).all()
        
        report_data[period.id] = {
            'period': period,
            'total_students': len(students),
            'total_classes': len(set(record.date for record in attendance_records)),
            'attendance_records': attendance_records
        }
    
    return render_template('reports.html', periods=periods, report_data=report_data)

@app.route('/reports/<int:period_id>/export')
@require_login
def export_attendance(period_id):
    """Export attendance to CSV format"""
    period = Period.query.get_or_404(period_id)
    
    if period.teacher_id != current_user.id:
        flash('You can only export your own period data.', 'error')
        return redirect(url_for('reports'))
    
    # Get all students and attendance records for this period
    students = Student.query.filter_by(class_name=period.class_name).all()
    attendance_records = db.session.query(Attendance).filter_by(period_id=period_id).all()
    
    # Group attendance by date and student
    attendance_by_date = {}
    for record in attendance_records:
        date_str = record.date.strftime('%Y-%m-%d')
        if date_str not in attendance_by_date:
            attendance_by_date[date_str] = {}
        attendance_by_date[date_str][record.student_id] = record.status
    
    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    dates = sorted(attendance_by_date.keys())
    header = ['Student Name', 'Roll Number'] + dates + ['Total Present', 'Attendance %']
    writer.writerow(header)
    
    # Data rows
    for student in students:
        row = [student.name, student.roll_number]
        present_count = 0
        total_classes = len(dates)
        
        for date_str in dates:
            status = attendance_by_date.get(date_str, {}).get(student.id, 'absent')
            row.append(status)
            if status == 'present':
                present_count += 1
        
        # Add totals
        row.append(present_count)
        percentage = (present_count / total_classes * 100) if total_classes > 0 else 0
        row.append(f"{percentage:.1f}%")
        
        writer.writerow(row)
    
    # Create response
    response = make_response(output.getvalue())
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = f'attachment; filename=attendance_{period.class_name}_{period.name}.csv'
    
    return response

@app.errorhandler(404)
def not_found(error):
    return render_template('403.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('403.html'), 500
