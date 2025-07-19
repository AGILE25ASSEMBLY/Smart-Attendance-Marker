"""
Enhanced Face Recognition Engine with Multiple Face Detection and Robust Performance
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageEnhance
import logging
from typing import List, Dict, Tuple, Optional
import json
import os

class EnhancedFaceRecognitionEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        
        # Initialize OpenCV Face Cascade (backup detection)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Face recognition components - Use alternative if cv2.face not available
        try:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.use_lbph = True
        except AttributeError:
            # Fallback to custom recognition method
            self.face_recognizer = None
            self.use_lbph = False
            self.logger.warning("LBPH recognizer not available, using custom method")
        
        self.is_trained = False
        self.student_labels = {}  # Maps student_id to label
        self.label_to_student = {}  # Maps label to student_id
        self.face_database = []  # Store face encodings for custom recognition
        
        # Performance settings
        self.confidence_threshold = 0.75
        self.max_faces_per_frame = 10
        
        self.logger.info("Enhanced Face Recognition Engine initialized")
    
    def detect_multiple_faces(self, image: np.ndarray) -> List[Dict]:
        """Enhanced face detection supporting multiple faces with various methods"""
        try:
            faces = []
            h, w = image.shape[:2]
            
            # Method 1: MediaPipe detection (primary for accuracy)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_image)
            
            if results.detections:
                for idx, detection in enumerate(results.detections):
                    if idx >= self.max_faces_per_frame:
                        break
                        
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert relative to absolute coordinates
                    x = max(0, int(bbox.xmin * w))
                    y = max(0, int(bbox.ymin * h))
                    width = min(int(bbox.width * w), w - x)
                    height = min(int(bbox.height * h), h - y)
                    
                    # Quality checks
                    if width < 50 or height < 50:  # Too small
                        continue
                    
                    confidence = detection.score[0] if detection.score else 0.5
                    
                    if confidence > 0.3:  # Minimum confidence threshold
                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': float(confidence),
                            'method': 'mediapipe',
                            'face_id': f"mp_{idx}",
                            'area': width * height
                        })
            
            # Method 2: OpenCV Haar Cascade (backup for missed faces)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multiple scale detection for better coverage
            for scale_factor in [1.1, 1.2, 1.3]:
                cv_faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=4, 
                    minSize=(40, 40),
                    maxSize=(int(w*0.8), int(h*0.8))
                )
                
                for idx, (x, y, w_face, h_face) in enumerate(cv_faces):
                    if len(faces) >= self.max_faces_per_frame:
                        break
                    
                    # Check if this face overlaps significantly with existing detections
                    if not self._is_overlapping(faces, (x, y, w_face, h_face)):
                        faces.append({
                            'bbox': (x, y, w_face, h_face),
                            'confidence': 0.6,  # Default for OpenCV
                            'method': 'opencv',
                            'face_id': f"cv_{scale_factor}_{idx}",
                            'area': w_face * h_face
                        })
                
                if len(faces) >= self.max_faces_per_frame:
                    break
            
            # Sort by confidence and area (larger faces typically better quality)
            faces.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)
            
            # Limit to max faces and filter low quality
            faces = [f for f in faces[:self.max_faces_per_frame] if f['confidence'] > 0.3]
            
            self.logger.info(f"Detected {len(faces)} faces using enhanced multi-method detection")
            return faces
            
        except Exception as e:
            self.logger.error(f"Enhanced face detection failed: {e}")
            return []
    
    def _is_overlapping(self, existing_faces: List[Dict], new_bbox: Tuple[int, int, int, int], threshold: float = 0.5) -> bool:
        """Check if a new face overlaps significantly with existing detections"""
        x1, y1, w1, h1 = new_bbox
        
        for face in existing_faces:
            x2, y2, w2, h2 = face['bbox']
            
            # Calculate intersection
            xi1 = max(x1, x2)
            yi1 = max(y1, y2)
            xi2 = min(x1 + w1, x2 + w2)
            yi2 = min(y1 + h1, y2 + h2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                continue
                
            intersection = (xi2 - xi1) * (yi2 - yi1)
            union = w1 * h1 + w2 * h2 - intersection
            
            if intersection / union > threshold:
                return True
        
        return False
    
    def extract_multiple_face_encodings(self, image_path: str) -> List[Dict]:
        """Extract face encodings for all detected faces in an image"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not load image: {image_path}")
                return []
            
            # Enhance image quality
            enhanced_image = self._enhance_image_quality(image)
            
            # Detect all faces
            faces = self.detect_multiple_faces(enhanced_image)
            
            face_encodings = []
            for face in faces:
                encoding = self._extract_face_encoding_from_bbox(enhanced_image, face['bbox'])
                if encoding is not None:
                    face_encodings.append({
                        'encoding': encoding,
                        'bbox': face['bbox'],
                        'confidence': face['confidence'],
                        'method': face['method'],
                        'face_id': face['face_id']
                    })
            
            self.logger.info(f"Extracted {len(face_encodings)} face encodings from {image_path}")
            return face_encodings
            
        except Exception as e:
            self.logger.error(f"Error extracting multiple face encodings: {e}")
            return []
    
    def _extract_face_encoding_from_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face encoding from a specific bounding box"""
        try:
            x, y, w, h = bbox
            
            # Extract face region with padding
            padding = int(min(w, h) * 0.1)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None
            
            # Resize to standard size for consistent encoding
            face_resized = cv2.resize(face_region, (100, 100))
            
            # Convert to grayscale and normalize
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            normalized_face = cv2.equalizeHist(gray_face)
            
            # Apply additional preprocessing
            blurred = cv2.GaussianBlur(normalized_face, (3, 3), 0)
            
            # Extract features using multiple methods
            encoding = self._extract_robust_features(blurred)
            
            return encoding
            
        except Exception as e:
            self.logger.error(f"Error extracting encoding from bbox: {e}")
            return None
    
    def _extract_robust_features(self, face_image: np.ndarray) -> np.ndarray:
        """Extract robust facial features using multiple techniques"""
        try:
            features = []
            
            # 1. LBP (Local Binary Pattern) features
            lbp_features = self._calculate_lbp(face_image)
            features.extend(lbp_features)
            
            # 2. HOG (Histogram of Oriented Gradients) features
            hog_features = self._calculate_hog(face_image)
            features.extend(hog_features)
            
            # 3. Statistical features
            stat_features = self._calculate_statistical_features(face_image)
            features.extend(stat_features)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting robust features: {e}")
            return face_image.flatten().astype(np.float32)
    
    def _calculate_lbp(self, image: np.ndarray) -> List[float]:
        """Calculate Local Binary Pattern features"""
        try:
            # Simple LBP implementation
            lbp = np.zeros_like(image)
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Create histogram
            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            return hist.astype(float).tolist()
            
        except Exception:
            return [0.0] * 256
    
    def _calculate_hog(self, image: np.ndarray) -> List[float]:
        """Calculate HOG features"""
        try:
            # Simple gradient calculation
            gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx)
            
            # Create histogram of gradients
            hist, _ = np.histogram(direction.ravel(), bins=18, range=(-np.pi, np.pi), weights=magnitude.ravel())
            return hist.astype(float).tolist()
            
        except Exception:
            return [0.0] * 18
    
    def _calculate_statistical_features(self, image: np.ndarray) -> List[float]:
        """Calculate statistical features"""
        try:
            features = []
            features.append(float(np.mean(image)))
            features.append(float(np.std(image)))
            features.append(float(np.median(image)))
            features.append(float(np.min(image)))
            features.append(float(np.max(image)))
            features.append(float(np.percentile(image, 25)))
            features.append(float(np.percentile(image, 75)))
            return features
            
        except Exception:
            return [0.0] * 7
    
    def _enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better face detection"""
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = contrast_enhancer.enhance(1.2)
            
            # Enhance sharpness
            sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = sharpness_enhancer.enhance(1.1)
            
            # Convert back to OpenCV
            enhanced_cv = cv2.cvtColor(np.array(enhanced), cv2.COLOR_RGB2BGR)
            
            # Reduce noise
            denoised = cv2.bilateralFilter(enhanced_cv, 9, 75, 75)
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image
    
    def train_recognizer_with_multiple_faces(self, student_face_data: List[Dict]):
        """Train the recognizer with multiple faces per student"""
        try:
            self.student_labels.clear()
            self.label_to_student.clear()
            self.face_database.clear()
            
            label_counter = 0
            
            for student_data in student_face_data:
                student_id = student_data['student_id']
                face_encodings = student_data['face_encodings']
                
                if not face_encodings:
                    continue
                
                # Assign label to student
                if student_id not in self.student_labels:
                    self.student_labels[student_id] = label_counter
                    self.label_to_student[label_counter] = student_id
                    label_counter += 1
                
                # Store encodings in database
                for encoding in face_encodings:
                    if isinstance(encoding, list):
                        encoding = np.array(encoding)
                    
                    self.face_database.append({
                        'student_id': student_id,
                        'encoding': encoding.flatten(),  # Flatten for consistent comparison
                        'label': self.student_labels[student_id]
                    })
            
            if self.use_lbph and len(self.face_database) > 0:
                try:
                    # Try LBPH training
                    all_encodings = []
                    all_labels = []
                    
                    for face_data in self.face_database:
                        encoding = face_data['encoding']
                        
                        # Reshape for LBPH
                        size = int(np.sqrt(len(encoding)))
                        if size * size == len(encoding):
                            reshaped = encoding.reshape(size, size)
                        else:
                            reshaped = encoding.reshape(10, -1)[:10, :10]
                        
                        all_encodings.append(reshaped.astype(np.uint8))
                        all_labels.append(face_data['label'])
                    
                    self.face_recognizer.train(all_encodings, np.array(all_labels))
                    self.logger.info(f"LBPH training successful with {len(all_encodings)} samples")
                except Exception as e:
                    self.logger.warning(f"LBPH training failed, using custom method: {e}")
                    self.use_lbph = False
            
            if len(self.face_database) > 0:
                self.is_trained = True
                self.logger.info(f"Trained recognizer with {len(self.face_database)} face samples from {len(self.student_labels)} students")
            else:
                self.logger.warning("No valid face encodings provided for training")
                
        except Exception as e:
            self.logger.error(f"Error training recognizer: {e}")
            self.is_trained = False
    
    def recognize_multiple_faces(self, image_bytes: bytes) -> List[Dict]:
        """Recognize multiple faces in a single image"""
        try:
            if not self.is_trained:
                self.logger.warning("Recognizer not trained")
                return []
            
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.error("Could not decode image from bytes")
                return []
            
            # Enhance image quality
            enhanced_image = self._enhance_image_quality(image)
            
            # Detect all faces
            faces = self.detect_multiple_faces(enhanced_image)
            
            recognition_results = []
            
            for face in faces:
                # Extract encoding for this face
                encoding = self._extract_face_encoding_from_bbox(enhanced_image, face['bbox'])
                
                if encoding is not None:
                    # Recognize face
                    student_id, confidence = self._recognize_single_encoding(encoding)
                    
                    recognition_results.append({
                        'student_id': student_id,
                        'confidence': confidence,
                        'bbox': face['bbox'],
                        'detection_confidence': face['confidence'],
                        'detection_method': face['method'],
                        'face_id': face['face_id']
                    })
            
            # Sort by recognition confidence
            recognition_results.sort(key=lambda x: x['confidence'], reverse=True)
            
            self.logger.info(f"Recognized {len(recognition_results)} faces")
            return recognition_results
            
        except Exception as e:
            self.logger.error(f"Error in multiple face recognition: {e}")
            return []
    
    def _recognize_single_encoding(self, encoding: np.ndarray) -> Tuple[Optional[int], float]:
        """Recognize a single face encoding"""
        try:
            if not self.is_trained or not self.face_database:
                return None, 0.0
            
            # Flatten encoding for comparison
            query_encoding = encoding.flatten()
            
            if self.use_lbph and self.face_recognizer:
                try:
                    # Try LBPH recognition
                    if len(encoding.shape) == 1:
                        size = int(np.sqrt(len(encoding)))
                        if size * size == len(encoding):
                            reshaped = encoding.reshape(size, size)
                        else:
                            reshaped = encoding.reshape(10, -1)[:10, :10]
                    else:
                        reshaped = encoding
                    
                    reshaped = reshaped.astype(np.uint8)
                    label, confidence = self.face_recognizer.predict(reshaped)
                    
                    # Convert confidence (lower is better for LBPH)
                    normalized_confidence = max(0, min(1, 1 - (confidence / 100)))
                    
                    if normalized_confidence > self.confidence_threshold:
                        student_id = self.label_to_student.get(label)
                        return student_id, normalized_confidence
                    
                except Exception as e:
                    self.logger.warning(f"LBPH recognition failed, using custom method: {e}")
            
            # Custom recognition using cosine similarity
            best_match = None
            best_confidence = 0.0
            
            for face_data in self.face_database:
                stored_encoding = face_data['encoding']
                
                # Calculate cosine similarity
                similarity = self._calculate_cosine_similarity(query_encoding, stored_encoding)
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = face_data['student_id']
            
            if best_confidence > self.confidence_threshold:
                return best_match, best_confidence
            
            return None, best_confidence
            
        except Exception as e:
            self.logger.error(f"Error recognizing encoding: {e}")
            return None, 0.0
    
    def _calculate_cosine_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate cosine similarity between two encodings"""
        try:
            # Ensure same length
            min_len = min(len(encoding1), len(encoding2))
            enc1 = encoding1[:min_len]
            enc2 = encoding2[:min_len]
            
            # Calculate cosine similarity
            dot_product = np.dot(enc1, enc2)
            norm1 = np.linalg.norm(enc1)
            norm2 = np.linalg.norm(enc2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

# Create global instance
enhanced_face_engine = EnhancedFaceRecognitionEngine()