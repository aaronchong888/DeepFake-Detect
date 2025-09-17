#!/usr/bin/env python3
"""
DeepFake Detection Deployment Script

This script provides a simple interface to deploy and run the deepfake detection model
for prediction on new images and videos.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def check_model_exists(model_path=None):
    """Check if the trained model exists"""
    if model_path is None:
        model_path = os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
    
    if os.path.exists(model_path):
        print(f"✓ Model found at: {model_path}")
        return True
    else:
        print(f"✗ Model not found at: {model_path}")
        print("Please train the model first using: python 03-train_cnn.py")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import tensorflow
        import cv2
        import mtcnn
        import numpy
        print("✓ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False

def run_prediction(input_files, model_path=None, output_file=None):
    """Run prediction on input files"""
    cmd = ['python', '04-predict.py'] + input_files
    
    if model_path is None:
        model_path = os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
    
    if model_path != os.path.join('.', 'tmp_checkpoint', 'best_model.h5'):
        cmd.extend(['--model', model_path])
    
    if output_file:
        cmd.extend(['--output', output_file])
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running prediction: {e}")
        return False

def interactive_mode():
    """Interactive mode for easy prediction"""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Predict single image/video")
        print("2. Predict multiple files")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            file_path = input("Enter path to image or video file: ").strip()
            if os.path.exists(file_path):
                run_prediction([file_path])
            else:
                print(f"File not found: {file_path}")
        
        elif choice == '2':
            print("Enter file paths (one per line, empty line to finish):")
            files = []
            while True:
                file_path = input().strip()
                if not file_path:
                    break
                if os.path.exists(file_path):
                    files.append(file_path)
                else:
                    print(f"Warning: File not found: {file_path}")
            
            if files:
                output_file = input("Enter output JSON file (optional, press Enter to skip): ").strip()
                output_file = output_file if output_file else None
                run_prediction(files, output_file=output_file)
            else:
                print("No valid files provided.")
        
        elif choice == '3':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    parser = argparse.ArgumentParser(description='DeepFake Detection Deployment Tool')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--check', '-c', action='store_true',
                       help='Check if model and dependencies are ready')
    parser.add_argument('--model', default=None,
                       help='Path to trained model')
    parser.add_argument('--output', '-o', help='Output JSON file for results')
    parser.add_argument('files', nargs='*', help='Files to predict (images or videos)')
    
    args = parser.parse_args()
    
    print("DeepFake Detection Deployment Tool")
    print("="*40)
    
    # Set default model path if not provided
    model_path = args.model if args.model else os.path.join('.', 'tmp_checkpoint', 'best_model.h5')
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    if not check_model_exists(model_path):
        sys.exit(1)
    
    if args.check:
        print("✓ System is ready for prediction!")
        return
    
    if args.interactive:
        interactive_mode()
    elif args.files:
        success = run_prediction(args.files, model_path, args.output)
        if not success:
            sys.exit(1)
    else:
        print("\nUsage examples:")
        print("  python deploy.py --interactive                    # Interactive mode")
        print("  python deploy.py image.jpg                        # Predict single image")
        print("  python deploy.py video.mp4 --output results.json  # Predict video with output")
        print("  python deploy.py *.jpg *.mp4                      # Predict multiple files")
        print("  python deploy.py --check                          # Check system readiness")

if __name__ == "__main__":
    main()