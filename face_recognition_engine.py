"""
Real Face Recognition Engine using OpenCV and MediaPipe
Replaces the mock implementation with actual computer vision algorithms
"""

import logging
from typing import List, Tuple, Optional, Dict
import os
from pathlib import Path
import numpy as np

# Try to import OpenCV and MediaPipe with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("OpenCV not available, using PIL-based implementation")
    CV2_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    print("MediaPipe not available, using basic face detection")
    MP_AVAILABLE = False

# Fallback imports
from PIL import Image, ImageEnhance
import hashlib

class FaceRecognitionEngine:
    """
    Advanced face recognition engine using MediaPipe for detection and OpenCV for recognition
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.is_trained = False
        self.trained_encodings = []
        self.trained_labels = []
        
        # Initialize MediaPipe if available
        if MP_AVAILABLE:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.mp_drawing = mp.solutions.drawing_utils
                self.face_detection = self.mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.7
                )
                self.logger.info("MediaPipe face detection initialized")
            except Exception as e:
                self.logger.error(f"MediaPipe initialization failed: {e}")
                self.face_detection = None
        else:
            self.face_detection = None
        
        # Initialize OpenCV components if available
        if CV2_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.logger.info("OpenCV face detection initialized")
            except Exception as e:
                self.logger.error(f"OpenCV initialization failed: {e}")
                self.face_cascade = None
        else:
            self.face_cascade = None
        
    def detect_faces_mediapipe(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MediaPipe
        Returns list of (x, y, width, height) tuples
        """
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)
            
            faces = []
            if results.detections:
                h, w, _ = image.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    faces.append((x, y, width, height))
            
            return faces
        except Exception as e:
            self.logger.error(f"MediaPipe face detection error: {e}")
            return []
    
    def detect_faces_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using OpenCV Haar cascades as fallback
        Returns list of (x, y, width, height) tuples
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            return [(x, y, w, h) for x, y, w, h in faces]
        except Exception as e:
            self.logger.error(f"OpenCV face detection error: {e}")
            return []
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces using MediaPipe first, fallback to OpenCV
        """
        faces = self.detect_faces_mediapipe(image)
        if not faces:
            faces = self.detect_faces_opencv(image)
        return faces
    
    def extract_face_encoding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face encoding from image file
        Returns normalized face features or None if no face detected
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            if CV2_AVAILABLE:
                return self._extract_face_encoding_cv2(image_path)
            else:
                return self._extract_face_encoding_pil(image_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting face encoding from {image_path}: {e}")
            return None
    
    def _extract_face_encoding_cv2(self, image_path: str) -> Optional[np.ndarray]:
        """OpenCV-based face encoding extraction"""
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        faces = self.detect_faces(image)
        if not faces:
            return None
        
        # Use the largest face detected
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        
        # Extract and normalize face ROI
        face_roi = image[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        normalized_face = cv2.resize(gray_face, (100, 100))
        
        return normalized_face.flatten()
    
    def _extract_face_encoding_pil(self, image_path: str) -> Optional[np.ndarray]:
        """PIL-based face encoding extraction (fallback)"""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize
                gray_img = img.convert('L')
                resized_img = gray_img.resize((100, 100))
                
                # Convert to numpy array and create encoding
                img_array = np.array(resized_img)
                
                # Apply basic enhancement
                enhanced = self._enhance_image_pil(img_array)
                
                return enhanced.flatten()
        except Exception as e:
            self.logger.error(f"PIL encoding extraction failed: {e}")
            return None
    
    def _enhance_image_pil(self, img_array: np.ndarray) -> np.ndarray:
        """Enhance image using PIL operations"""
        try:
            # Convert numpy array to PIL Image
            img = Image.fromarray(img_array.astype(np.uint8))
            
            # Apply contrast enhancement
            enhancer = ImageEnhance.Contrast(img)
            enhanced = enhancer.enhance(1.2)
            
            # Apply sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.1)
            
            return np.array(enhanced)
        except Exception as e:
            self.logger.error(f"PIL enhancement failed: {e}")
            return img_array
    
    def extract_face_encoding_from_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extract face encoding from image bytes (for webcam captures)
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                self.logger.error("Could not decode image bytes")
                return None
            
            faces = self.detect_faces(image)
            if not faces:
                return None
            
            # Use the largest face detected
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract and normalize face ROI
            face_roi = image[y:y+h, x:x+w]
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            normalized_face = cv2.resize(gray_face, (100, 100))
            
            return normalized_face.flatten()
            
        except Exception as e:
            self.logger.error(f"Error extracting face encoding from bytes: {e}")
            return None
    
    def train_recognizer(self, face_encodings: List[np.ndarray], labels: List[int]) -> bool:
        """
        Train the face recognizer with known faces
        """
        try:
            if not face_encodings or not labels:
                self.logger.warning("No training data provided")
                return False
            
            if len(face_encodings) != len(labels):
                self.logger.error("Mismatch between face encodings and labels")
                return False
            
            # Store training data for comparison-based recognition
            self.trained_encodings = []
            self.trained_labels = []
            
            for i, encoding in enumerate(face_encodings):
                if encoding is not None and i < len(labels):
                    self.trained_encodings.append(encoding)
                    self.trained_labels.append(labels[i])
            
            if not self.trained_encodings:
                self.logger.warning("No valid face encodings for training")
                return False
            
            self.is_trained = True
            self.logger.info(f"Face recognizer trained with {len(self.trained_encodings)} faces")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training face recognizer: {e}")
            return False
    
    def recognize_face(self, face_encoding: np.ndarray, confidence_threshold: float = 0.85) -> Tuple[Optional[int], float]:
        """
        Recognize a face encoding against trained faces
        Returns (student_id, confidence) or (None, 0.0) if no match
        Higher confidence values indicate better matches
        """
        try:
            if not self.is_trained or not self.trained_encodings:
                self.logger.warning("Face recognizer not trained")
                return None, 0.0
            
            if face_encoding is None:
                return None, 0.0
            
            best_match_label = None
            best_confidence = 0.0
            
            # Compare against all trained encodings
            for i, trained_encoding in enumerate(self.trained_encodings):
                if trained_encoding is not None:
                    confidence = self._calculate_similarity(face_encoding, trained_encoding)
                    
                    if confidence > best_confidence and confidence >= confidence_threshold:
                        best_confidence = confidence
                        best_match_label = self.trained_labels[i]
            
            return best_match_label, best_confidence
                
        except Exception as e:
            self.logger.error(f"Error recognizing face: {e}")
            return None, 0.0
    
    def _calculate_similarity(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """Calculate similarity between two face encodings"""
        try:
            if CV2_AVAILABLE:
                # Use OpenCV template matching
                img1 = encoding1.reshape(100, 100).astype(np.float32)
                img2 = encoding2.reshape(100, 100).astype(np.float32)
                
                result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
                return float(result[0][0])
            else:
                # Use normalized correlation
                enc1_norm = encoding1 / np.linalg.norm(encoding1)
                enc2_norm = encoding2 / np.linalg.norm(encoding2)
                correlation = np.dot(enc1_norm, enc2_norm)
                return float(correlation)
                
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray, threshold: float = 0.85) -> bool:
        """
        Compare two face encodings for similarity
        """
        try:
            if encoding1 is None or encoding2 is None:
                return False
            
            similarity = self._calculate_similarity(encoding1, encoding2)
            return similarity >= threshold
            
        except Exception as e:
            self.logger.error(f"Error comparing faces: {e}")
            return False
    
    def get_face_landmarks(self, image: np.ndarray) -> List[Dict]:
        """
        Get facial landmarks for advanced recognition
        """
        try:
            # This would use MediaPipe Face Mesh for more detailed landmarks
            # For now, returning basic face detection results
            faces = self.detect_faces(image)
            landmarks = []
            
            for face in faces:
                x, y, w, h = face
                landmarks.append({
                    'bbox': face,
                    'center': (x + w//2, y + h//2),
                    'area': w * h
                })
            
            return landmarks
            
        except Exception as e:
            self.logger.error(f"Error getting face landmarks: {e}")
            return []
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better face recognition
        """
        try:
            # Apply histogram equalization
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge and convert back
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image

# Global instance
face_engine = FaceRecognitionEngine()