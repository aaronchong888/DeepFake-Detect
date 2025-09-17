#!/usr/bin/env python3
"""
Setup script for DeepFake Detection prediction system.

This script helps users set up and verify the prediction system is ready to use.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    
    if sys.version_info < (3, 6):
        print(f"✗ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        print("This project requires Python 3.6 or higher")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    if not os.path.exists('requirements.txt'):
        print("✗ requirements.txt not found")
        return False
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        print("Please install manually using: pip install -r requirements.txt")
        return False

def check_model_status():
    """Check if model is trained and available"""
    print("\nChecking model status...")
    
    model_path = os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
    
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
        print(f"✓ Trained model found: {model_path} ({model_size:.1f} MB)")
        return True
    else:
        print(f"✗ No trained model found at: {model_path}")
        print("You need to train the model first. Follow these steps:")
        print("  1. Prepare your dataset using steps 0-2")
        print("  2. Run: python 03-train_cnn.py")
        return False

def check_sample_data():
    """Check if sample data is available for testing"""
    print("\nChecking sample data...")
    
    sample_videos_dir = os.path.join('.', 'train_sample_videos')
    
    if os.path.exists(sample_videos_dir):
        video_files = [f for f in os.listdir(sample_videos_dir) if f.endswith('.mp4')]
        if video_files:
            print(f"✓ Found {len(video_files)} sample videos for testing")
            return True
        else:
            print("⚠ Sample video directory exists but no videos found")
    else:
        print("⚠ No sample videos directory found")
    
    print("Sample videos are useful for testing the prediction system")
    return False

def create_prediction_examples():
    """Create example files for testing"""
    print("\nCreating example configuration...")
    
    # Create a simple configuration file
    config = {
        "model_path": os.path.join(".", "tmp_checkpoint", "best_model.h5"),
        "input_size": 128,
        "confidence_threshold": 0.95,
        "face_margin": 0.3,
        "default_frame_interval": 30
    }
    
    with open('prediction_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Created prediction_config.json")
    
    # Create a simple batch processing example
    batch_example = """# Example batch file for processing multiple files
# Add your file paths here, one per line
# 
# Examples:
# /path/to/image1.jpg
# /path/to/image2.jpg
# /path/to/video1.mp4
# 
# Then run: python 04-predict.py $(cat batch_files.txt)
"""
    
    with open('batch_files_example.txt', 'w') as f:
        f.write(batch_example)
    
    print("✓ Created batch_files_example.txt")

def run_quick_test():
    """Run a quick test to verify everything works"""
    print("\nRunning quick system test...")
    
    try:
        # Try to import required modules
        import tensorflow as tf
        import cv2
        import numpy as np
        from mtcnn import MTCNN
        
        print("✓ All required modules can be imported")
        
        # Check TensorFlow GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("✓ GPU available for TensorFlow")
        else:
            print("⚠ No GPU detected, will use CPU (slower)")
        
        # Test MTCNN initialization
        detector = MTCNN()
        print("✓ MTCNN face detector initialized")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def print_usage_instructions():
    """Print instructions for using the prediction system"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - PREDICTION SYSTEM READY!")
    print("="*60)
    
    print("\nQuick Start:")
    print("  python deploy.py --interactive    # Interactive mode (recommended)")
    print("  python deploy.py --check          # Verify system is ready")
    
    print("\nDirect Usage:")
    print("  python 04-predict.py image.jpg    # Predict single image")
    print("  python 04-predict.py video.mp4    # Predict video")
    print("  python 04-predict.py *.jpg        # Predict multiple files")
    
    print("\nTesting:")
    print("  python test_prediction.py         # Run system tests")
    print("  python example_predict.py         # See usage examples")
    
    print("\nFor help:")
    print("  python 04-predict.py --help       # Command line options")
    print("  python deploy.py --help           # Deployment options")

def main():
    """Main setup function"""
    print("DeepFake Detection - Prediction System Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    print("\nWould you like to install/update dependencies? (y/n): ", end="")
    if input().lower().startswith('y'):
        install_dependencies()
    
    # Check model status
    model_ready = check_model_status()
    
    # Check sample data
    check_sample_data()
    
    # Create example files
    create_prediction_examples()
    
    # Run quick test
    if run_quick_test():
        print("✓ System test passed")
    else:
        print("✗ System test failed - check error messages above")
        return
    
    # Print usage instructions
    if model_ready:
        print_usage_instructions()
    else:
        print("\n" + "="*60)
        print("SETUP INCOMPLETE - MODEL TRAINING REQUIRED")
        print("="*60)
        print("\nTo complete setup:")
        print("1. Prepare your dataset (steps 0-2 in README)")
        print("2. Train the model: python 03-train_cnn.py")
        print("3. Run this setup again: python setup_prediction.py")
        
    print(f"\nSetup completed! Check the files created:")
    print("  - prediction_config.json")
    print("  - batch_files_example.txt")

if __name__ == "__main__":
    main()