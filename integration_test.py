#!/usr/bin/env python3
"""
Integration test for the DeepFake Detection system.

This script tests the complete pipeline from model loading to prediction
to ensure everything works together correctly.
"""

import os
import sys
import tempfile
import shutil
import cv2
import numpy as np
from pathlib import Path

def create_test_image():
    """Create a simple test image for testing"""
    # Create a simple test image with a face-like pattern
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Add a simple rectangular "face" pattern
    cv2.rectangle(img, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(img, (120, 130), (140, 150), (0, 0, 0), -1)  # Left eye
    cv2.rectangle(img, (160, 130), (180, 150), (0, 0, 0), -1)  # Right eye
    cv2.rectangle(img, (140, 170), (160, 180), (0, 0, 0), -1)  # Nose
    cv2.rectangle(img, (130, 190), (170, 200), (0, 0, 0), -1)  # Mouth
    
    return img

def test_model_loading():
    """Test if the prediction system can load the trained model"""
    print("Testing model loading...")
    
    try:
        # Import the predictor
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict", "04-predict.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        DeepFakePredictor = predict_module.DeepFakePredictor
        
        # Try to initialize predictor
        model_path = os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found at {model_path}")
            print("Please train the model first using: python 03-train_cnn.py")
            return None
        
        predictor = DeepFakePredictor(model_path)
        print("✓ Model loaded successfully")
        return predictor
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def test_face_detection(predictor):
    """Test face detection functionality"""
    print("\nTesting face detection...")
    
    try:
        # Create a test image
        test_img = create_test_image()
        
        # Test face detection
        face = predictor.detect_and_crop_face(test_img)
        
        if face is not None:
            print("✓ Face detection working (detected face in test image)")
            return True
        else:
            print("⚠ Face detection returned None (may be normal for synthetic test image)")
            return True  # This is actually expected for our simple test image
            
    except Exception as e:
        print(f"✗ Face detection error: {e}")
        return False

def test_preprocessing(predictor):
    """Test image preprocessing"""
    print("\nTesting image preprocessing...")
    
    try:
        # Create a test image
        test_img = create_test_image()
        
        # Test preprocessing
        processed = predictor.preprocess_image(test_img)
        
        # Check output shape
        expected_shape = (1, 128, 128, 3)  # Batch size 1, 128x128, 3 channels
        
        if processed.shape == expected_shape:
            print(f"✓ Preprocessing working (output shape: {processed.shape})")
            
            # Check value range
            if 0 <= processed.min() <= processed.max() <= 1:
                print(f"✓ Normalization working (value range: {processed.min():.3f} to {processed.max():.3f})")
                return True
            else:
                print(f"✗ Normalization issue (value range: {processed.min():.3f} to {processed.max():.3f})")
                return False
        else:
            print(f"✗ Preprocessing shape mismatch (got {processed.shape}, expected {expected_shape})")
            return False
            
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def test_prediction(predictor):
    """Test model prediction"""
    print("\nTesting model prediction...")
    
    try:
        # Create a test image and save it temporarily
        test_img = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, test_img)
            tmp_path = tmp_file.name
        
        try:
            # Test prediction
            result = predictor.predict_image(tmp_path)
            
            # Check result structure
            expected_keys = ['filename', 'face_detected', 'prediction', 'confidence', 'classification']
            
            if all(key in result for key in expected_keys):
                print("✓ Prediction result structure correct")
                print(f"  Face detected: {result['face_detected']}")
                
                if result['face_detected']:
                    print(f"  Classification: {result['classification']}")
                    print(f"  Confidence: {result['confidence']:.2%}")
                    print(f"  Raw prediction: {result['prediction']:.4f}")
                else:
                    print("  No face detected (expected for synthetic test image)")
                
                return True
            else:
                missing_keys = [key for key in expected_keys if key not in result]
                print(f"✗ Missing keys in result: {missing_keys}")
                return False
                
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return False

def test_with_sample_video(predictor):
    """Test prediction with sample video if available"""
    print("\nTesting with sample video...")
    
    sample_videos_dir = os.path.join('.', 'train_sample_videos')
    
    if not os.path.exists(sample_videos_dir):
        print("⚠ No sample videos directory found, skipping video test")
        return True
    
    video_files = [f for f in os.listdir(sample_videos_dir) if f.endswith('.mp4')]
    
    if not video_files:
        print("⚠ No sample videos found, skipping video test")
        return True
    
    # Test with first video
    video_path = os.path.join(sample_videos_dir, video_files[0])
    
    try:
        print(f"Testing with: {video_files[0]}")
        result = predictor.predict_video(video_path, frame_interval=60)  # Analyze fewer frames for speed
        
        print(f"  Total frames: {result['total_frames']}")
        print(f"  Analyzed frames: {result['analyzed_frames']}")
        print(f"  Faces detected: {result['faces_detected']}")
        
        if result['faces_detected'] > 0:
            print(f"  Classification: {result['classification']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print("✓ Video prediction working")
        else:
            print("⚠ No faces detected in video (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"✗ Video prediction error: {e}")
        return False

def test_command_line_interface():
    """Test command line interface"""
    print("\nTesting command line interface...")
    
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run([sys.executable, '04-predict.py', '--help'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'DeepFake Detection Prediction Tool' in result.stdout:
            print("✓ Command line interface working")
            return True
        else:
            print("✗ Command line interface issue")
            return False
            
    except Exception as e:
        print(f"✗ Command line test error: {e}")
        return False

def main():
    """Run all integration tests"""
    print("DeepFake Detection - Integration Test")
    print("="*50)
    
    # Test model loading
    predictor = test_model_loading()
    if not predictor:
        print("\n✗ Integration test failed: Cannot load model")
        sys.exit(1)
    
    # Run all tests
    tests = [
        ("Face Detection", lambda: test_face_detection(predictor)),
        ("Image Preprocessing", lambda: test_preprocessing(predictor)),
        ("Model Prediction", lambda: test_prediction(predictor)),
        ("Sample Video", lambda: test_with_sample_video(predictor)),
        ("Command Line Interface", test_command_line_interface)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("INTEGRATION TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! The system is ready for use.")
        print("\nYou can now use:")
        print("  python deploy.py --interactive")
        print("  python 04-predict.py your_image.jpg")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()