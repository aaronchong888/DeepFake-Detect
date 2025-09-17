#!/usr/bin/env python3
"""
Test script for the DeepFake Detection prediction system.

This script tests the prediction functionality using the sample videos
provided in the repository.
"""

import os
import sys
import glob
import json
from pathlib import Path

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("Testing model loading...")
    
    try:
        # Import the predictor
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import from the 04-predict.py file
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict", "04-predict.py")
        predict_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(predict_module)
        DeepFakePredictor = predict_module.DeepFakePredictor
        
        # Initialize predictor
        predictor = DeepFakePredictor()
        print("✓ Model loaded successfully")
        return predictor
        
    except FileNotFoundError as e:
        print(f"✗ Model file not found: {e}")
        print("Please train the model first using: python 03-train_cnn.py")
        return None
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def test_sample_videos(predictor):
    """Test prediction on sample videos"""
    print("\nTesting sample videos...")
    
    # Look for sample videos
    video_dir = "./train_sample_videos"
    if not os.path.exists(video_dir):
        print(f"✗ Sample video directory not found: {video_dir}")
        return
    
    # Get all video files
    video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
    
    if not video_files:
        print("✗ No sample videos found")
        return
    
    print(f"Found {len(video_files)} sample videos")
    
    # Test first few videos (to avoid long processing time)
    test_videos = video_files[:3]  # Test first 3 videos
    
    results = []
    for video_path in test_videos:
        print(f"Testing: {os.path.basename(video_path)}")
        
        try:
            result = predictor.predict_video(video_path, frame_interval=60)  # Analyze fewer frames for speed
            results.append(result)
            
            if result['faces_detected'] > 0:
                print(f"  ✓ Classification: {result['classification']} "
                      f"(Confidence: {result['confidence']:.2%})")
                print(f"  ✓ Analyzed {result['faces_detected']} faces in {result['analyzed_frames']} frames")
            else:
                print(f"  ⚠ No faces detected")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'filename': os.path.basename(video_path),
                'error': str(e)
            })
    
    # Save test results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Test results saved to test_results.json")
    return results

def test_face_detection():
    """Test face detection functionality"""
    print("\nTesting face detection...")
    
    try:
        import cv2
        from mtcnn import MTCNN
        
        detector = MTCNN()
        print("✓ MTCNN face detector initialized")
        
        # Test with a sample video frame
        video_dir = "./train_sample_videos"
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        
        if video_files:
            # Extract a frame from the first video
            cap = cv2.VideoCapture(video_files[0])
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                results = detector.detect_faces(frame_rgb)
                
                if results:
                    print(f"✓ Detected {len(results)} face(s) in sample frame")
                    for i, result in enumerate(results):
                        print(f"  Face {i+1}: Confidence {result['confidence']:.2%}")
                else:
                    print("⚠ No faces detected in sample frame")
            else:
                print("✗ Could not extract frame from sample video")
        else:
            print("⚠ No sample videos available for face detection test")
            
    except Exception as e:
        print(f"✗ Face detection test failed: {e}")

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    dependencies = [
        ('tensorflow', 'TensorFlow'),
        ('cv2', 'OpenCV'),
        ('mtcnn', 'MTCNN'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas')
    ]
    
    missing_deps = []
    
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Missing")
            missing_deps.append(display_name)
    
    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("✓ All dependencies available")
        return True

def main():
    """Run all tests"""
    print("DeepFake Detection - Prediction System Test")
    print("="*50)
    
    # Test dependencies first
    if not test_dependencies():
        print("\n✗ Dependency test failed. Please install missing packages.")
        return
    
    # Test model loading
    predictor = test_model_loading()
    if not predictor:
        print("\n✗ Model loading test failed. Cannot proceed with other tests.")
        return
    
    # Test face detection
    test_face_detection()
    
    # Test sample videos
    results = test_sample_videos(predictor)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if results:
        successful_predictions = sum(1 for r in results if 'classification' in r)
        total_tests = len(results)
        
        print(f"Video prediction tests: {successful_predictions}/{total_tests} successful")
        
        if successful_predictions > 0:
            print("✓ Prediction system is working correctly!")
            print("\nYou can now use the prediction system with:")
            print("  python deploy.py --interactive")
            print("  python 04-predict.py your_file.jpg")
        else:
            print("⚠ Some issues detected. Check the error messages above.")
    else:
        print("⚠ No prediction tests were run")
    
    print("\nFor more examples, run: python example_predict.py")

if __name__ == "__main__":
    main()