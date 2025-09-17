import os
import cv2
import numpy as np
import argparse
from mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import json

# Import EfficientNet for model loading compatibility
try:
    from efficientnet.tfkeras import EfficientNetB0
except ImportError:
    print("Warning: EfficientNet not found. Model loading may fail if using EfficientNet architecture.")
    EfficientNetB0 = None

class DeepFakePredictor:
    def __init__(self, model_path=None):
        """
        Initialize the DeepFake Predictor
        
        Args:
            model_path (str): Path to the trained model file
        """
        # Use cross-platform path handling
        if model_path is None:
            model_path = os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
        
        self.model_path = model_path
        self.model = None
        self.detector = MTCNN()
        self.input_size = 128
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained deepfake detection model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        
        # Try to load model with custom objects for EfficientNet compatibility
        custom_objects = {}
        if EfficientNetB0 is not None:
            custom_objects['EfficientNetB0'] = EfficientNetB0
        
        try:
            self.model = load_model(self.model_path, custom_objects=custom_objects)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model with custom objects: {e}")
            print("Trying to load model without custom objects...")
            try:
                self.model = load_model(self.model_path)
                print("Model loaded successfully!")
            except Exception as e2:
                raise Exception(f"Failed to load model: {e2}")
    
    def detect_and_crop_face(self, image, margin=0.3, confidence_threshold=0.95):
        """
        Detect and crop face from image using MTCNN
        
        Args:
            image: Input image (numpy array)
            margin (float): Margin to add around detected face (30% as in training)
            confidence_threshold (float): Minimum confidence for face detection (95% as in training)
            
        Returns:
            cropped_face: Cropped face image or None if no face detected
        """
        # Detect faces
        results = self.detector.detect_faces(image)
        
        if not results:
            return None
        
        # Get the face with highest confidence
        best_face = max(results, key=lambda x: x['confidence'])
        
        if best_face['confidence'] < confidence_threshold:
            return None
        
        # Extract bounding box
        x, y, width, height = best_face['box']
        
        # Add margin (30% as mentioned in README and training)
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        # Calculate crop coordinates
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + width + margin_x)
        y2 = min(image.shape[0], y + height + margin_y)
        
        # Crop face
        cropped_face = image[y1:y2, x1:x2]
        
        return cropped_face
    
    def preprocess_image(self, image):
        """
        Preprocess image for model prediction - MUST match training preprocessing exactly
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            preprocessed_image: Preprocessed image ready for prediction
        """
        # Resize to model input size (128x128 as in training)
        resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # Convert to array and normalize to [0,1] exactly as in training (rescale = 1/255)
        img_array = img_to_array(resized)
        img_array = img_array / 255.0  # This matches the training rescale = 1/255
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_image(self, image_path):
        """
        Predict if an image contains a deepfake
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect and crop face
        face = self.detect_and_crop_face(image)
        
        if face is None:
            return {
                'filename': os.path.basename(image_path),
                'face_detected': False,
                'prediction': None,
                'confidence': None,
                'classification': 'No face detected'
            }
        
        # Preprocess face
        processed_face = self.preprocess_image(face)
        
        # Make prediction
        prediction = self.model.predict(processed_face, verbose=0)[0][0]
        
        # Interpret results (0 = fake, 1 = real)
        classification = 'Real' if prediction > 0.5 else 'Fake'
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        return {
            'filename': os.path.basename(image_path),
            'face_detected': True,
            'prediction': float(prediction),
            'confidence': float(confidence),
            'classification': classification
        }
    
    def predict_video(self, video_path, frame_interval=30):
        """
        Predict if a video contains deepfakes by analyzing frames
        
        Args:
            video_path (str): Path to the video file
            frame_interval (int): Analyze every nth frame
            
        Returns:
            dict: Prediction results for the video
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_path}")
        
        frame_predictions = []
        frame_count = 0
        analyzed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every nth frame
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and crop face
                face = self.detect_and_crop_face(frame_rgb)
                
                if face is not None:
                    # Preprocess and predict
                    processed_face = self.preprocess_image(face)
                    prediction = self.model.predict(processed_face, verbose=0)[0][0]
                    frame_predictions.append(prediction)
                    analyzed_frames += 1
            
            frame_count += 1
        
        cap.release()
        
        if not frame_predictions:
            return {
                'filename': os.path.basename(video_path),
                'total_frames': frame_count,
                'analyzed_frames': 0,
                'faces_detected': 0,
                'average_prediction': None,
                'classification': 'No faces detected',
                'confidence': None
            }
        
        # Calculate average prediction
        avg_prediction = np.mean(frame_predictions)
        classification = 'Real' if avg_prediction > 0.5 else 'Fake'
        confidence = avg_prediction if avg_prediction > 0.5 else 1 - avg_prediction
        
        return {
            'filename': os.path.basename(video_path),
            'total_frames': frame_count,
            'analyzed_frames': analyzed_frames,
            'faces_detected': len(frame_predictions),
            'average_prediction': float(avg_prediction),
            'classification': classification,
            'confidence': float(confidence),
            'frame_predictions': [float(p) for p in frame_predictions]
        }
    
    def predict_batch(self, input_paths, output_file=None):
        """
        Predict multiple files and optionally save results
        
        Args:
            input_paths (list): List of file paths
            output_file (str): Optional path to save results as JSON
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for path in input_paths:
            print(f"Processing: {path}")
            
            try:
                # Check if it's an image or video
                ext = os.path.splitext(path)[1].lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    result = self.predict_image(path)
                elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                    result = self.predict_video(path)
                else:
                    result = {
                        'filename': os.path.basename(path),
                        'error': f'Unsupported file format: {ext}'
                    }
                
                results.append(result)
                print(f"Result: {result['classification'] if 'classification' in result else 'Error'}")
                
            except Exception as e:
                error_result = {
                    'filename': os.path.basename(path),
                    'error': str(e)
                }
                results.append(error_result)
                print(f"Error processing {path}: {e}")
        
        # Save results if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='DeepFake Detection Prediction Tool')
    parser.add_argument('input', nargs='+', help='Input image(s) or video(s) to analyze')
    parser.add_argument('--model', default=None, 
                       help='Path to trained model (default: ./tmp_checkpoint/best_model.h5)')
    parser.add_argument('--output', help='Output JSON file to save results')
    parser.add_argument('--frame-interval', type=int, default=30,
                       help='For videos: analyze every nth frame (default: 30)')
    
    args = parser.parse_args()
    
    # Use default model path if not specified
    model_path = args.model if args.model else os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
    
    # Initialize predictor
    try:
        predictor = DeepFakePredictor(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Process inputs
    results = predictor.predict_batch(args.input, args.output)
    
    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    
    for result in results:
        if 'error' in result:
            print(f"{result['filename']}: ERROR - {result['error']}")
        elif 'classification' in result:
            print(f"{result['filename']}: {result['classification']} "
                  f"(Confidence: {result['confidence']:.2%})")
        else:
            print(f"{result['filename']}: Unknown result")


if __name__ == "__main__":
    main()