#!/usr/bin/env python3
"""
Example script demonstrating how to use the DeepFake Detection prediction functionality.

This script shows different ways to use the prediction system and can be used as a
starting point for integrating the deepfake detection into your own applications.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our prediction module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from predict import DeepFakePredictor
except ImportError:
    # If the above doesn't work, try importing from the 04-predict.py file
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", "04-predict.py")
    predict_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict_module)
    DeepFakePredictor = predict_module.DeepFakePredictor

def example_single_image_prediction():
    """Example: Predict a single image"""
    print("="*50)
    print("EXAMPLE 1: Single Image Prediction")
    print("="*50)
    
    # Initialize predictor
    try:
        predictor = DeepFakePredictor()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Make sure you have trained the model first using: python 03-train_cnn.py")
        return
    
    # Example image path (you can change this to your own image)
    image_path = "example_image.jpg"  # Replace with actual image path
    
    if not os.path.exists(image_path):
        print(f"Example image not found at {image_path}")
        print("Please provide a valid image path to test the prediction.")
        return
    
    try:
        result = predictor.predict_image(image_path)
        
        print(f"File: {result['filename']}")
        print(f"Face Detected: {result['face_detected']}")
        
        if result['face_detected']:
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Raw Prediction Score: {result['prediction']:.4f}")
        else:
            print("No face detected in the image")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

def example_video_prediction():
    """Example: Predict a video"""
    print("\n" + "="*50)
    print("EXAMPLE 2: Video Prediction")
    print("="*50)
    
    # Initialize predictor
    try:
        predictor = DeepFakePredictor()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Example video path (you can change this to your own video)
    video_path = "example_video.mp4"  # Replace with actual video path
    
    if not os.path.exists(video_path):
        print(f"Example video not found at {video_path}")
        print("Please provide a valid video path to test the prediction.")
        return
    
    try:
        result = predictor.predict_video(video_path, frame_interval=30)
        
        print(f"File: {result['filename']}")
        print(f"Total Frames: {result['total_frames']}")
        print(f"Analyzed Frames: {result['analyzed_frames']}")
        print(f"Faces Detected: {result['faces_detected']}")
        
        if result['faces_detected'] > 0:
            print(f"Classification: {result['classification']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Average Prediction Score: {result['average_prediction']:.4f}")
        else:
            print("No faces detected in the video")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

def example_batch_prediction():
    """Example: Batch prediction with multiple files"""
    print("\n" + "="*50)
    print("EXAMPLE 3: Batch Prediction")
    print("="*50)
    
    # Initialize predictor
    try:
        predictor = DeepFakePredictor()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    # Example file paths (replace with actual files)
    file_paths = [
        "example_image1.jpg",
        "example_image2.jpg", 
        "example_video.mp4"
    ]
    
    # Filter to only existing files
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        print("No example files found. Please add some images or videos to test.")
        print("Expected files:", file_paths)
        return
    
    try:
        results = predictor.predict_batch(existing_files, output_file="example_results.json")
        
        print(f"Processed {len(results)} files:")
        for result in results:
            if 'error' in result:
                print(f"  {result['filename']}: ERROR - {result['error']}")
            elif 'classification' in result:
                print(f"  {result['filename']}: {result['classification']} "
                      f"(Confidence: {result['confidence']:.2%})")
        
        print("Results saved to example_results.json")
        
    except Exception as e:
        print(f"Error during batch prediction: {e}")

def example_programmatic_usage():
    """Example: How to use the predictor in your own code"""
    print("\n" + "="*50)
    print("EXAMPLE 4: Programmatic Usage")
    print("="*50)
    
    print("Here's how you can integrate the deepfake detector into your own Python code:")
    print()
    
    code_example = '''
# Import the predictor class
from predict import DeepFakePredictor

# Initialize the predictor
predictor = DeepFakePredictor(model_path='./tmp_checkpoint/best_model.h5')

# Predict a single image
result = predictor.predict_image('path/to/image.jpg')

# Check the result
if result['face_detected']:
    if result['classification'] == 'Fake':
        print(f"Warning: Potential deepfake detected! Confidence: {result['confidence']:.2%}")
    else:
        print(f"Image appears to be real. Confidence: {result['confidence']:.2%}")
else:
    print("No face detected in the image")

# For videos
video_result = predictor.predict_video('path/to/video.mp4')
if video_result['faces_detected'] > 0:
    print(f"Video classification: {video_result['classification']}")
    print(f"Analyzed {video_result['faces_detected']} faces across {video_result['analyzed_frames']} frames")
'''
    
    print(code_example)

def main():
    """Run all examples"""
    print("DeepFake Detection - Prediction Examples")
    print("="*50)
    
    # Check if model exists
    model_path = './tmp_checkpoint/best_model.h5'
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("Please train the model first using: python 03-train_cnn.py")
        print("Or use the deployment script: python deploy.py --check")
        return
    
    # Run examples
    example_single_image_prediction()
    example_video_prediction()
    example_batch_prediction()
    example_programmatic_usage()
    
    print("\n" + "="*50)
    print("Examples completed!")
    print("="*50)
    print("To run predictions on your own files, use:")
    print("  python deploy.py --interactive")
    print("  python 04-predict.py your_file.jpg")

if __name__ == "__main__":
    main()