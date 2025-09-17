# DeepFake Detection - Prediction System Guide

This guide explains how to use the prediction system that has been added to the DeepFake Detection project.

## Overview

The prediction system allows you to use your trained deepfake detection model to analyze new images and videos. It includes face detection, preprocessing, and classification with confidence scores.

## Quick Start

### 1. Setup and Verification

```bash
# Setup the prediction environment
python setup_prediction.py

# Check if everything is ready
python deploy.py --check

# Run integration tests
python integration_test.py
```

### 2. Making Predictions

#### Interactive Mode (Recommended for beginners)
```bash
python deploy.py --interactive
```

#### Command Line Usage
```bash
# Single image
python 04-predict.py image.jpg

# Single video
python 04-predict.py video.mp4

# Multiple files with JSON output
python 04-predict.py image1.jpg image2.jpg video.mp4 --output results.json

# Custom model path
python 04-predict.py image.jpg --model /path/to/custom/model.h5
```

#### Deployment Script
```bash
# Interactive mode
python deploy.py --interactive

# Direct file processing
python deploy.py image.jpg video.mp4 --output results.json

# System check
python deploy.py --check
```

## New Files Added

### Core Prediction System
- **`04-predict.py`** - Main prediction script with comprehensive functionality
- **`deploy.py`** - User-friendly deployment script with interactive mode
- **`setup_prediction.py`** - Setup and verification script
- **`integration_test.py`** - Complete system integration test

### Testing and Examples
- **`test_prediction.py`** - System testing with sample videos
- **`example_predict.py`** - Usage examples and integration guide

### Configuration Files (Created by setup)
- **`prediction_config.json`** - Configuration parameters
- **`batch_files_example.txt`** - Example for batch processing

## Features

### Image Analysis
- Automatic face detection using MTCNN
- Face cropping with 30% margins (matching training)
- Preprocessing identical to training pipeline
- Binary classification (Real/Fake) with confidence scores

### Video Analysis
- Frame-by-frame analysis at configurable intervals
- Batch processing of detected faces
- Average prediction across all detected faces
- Detailed frame-by-frame results

### Batch Processing
- Process multiple images and videos at once
- JSON output for integration with other systems
- Progress tracking and error handling

### Error Handling
- Comprehensive error messages
- Graceful handling of missing faces
- Model loading compatibility checks
- Cross-platform path handling

## Technical Details

### Model Compatibility
The prediction system is designed to work with the EfficientNetB0-based model trained by `03-train_cnn.py`. It includes:

- Automatic EfficientNet import handling
- Custom object loading for model compatibility
- Fallback mechanisms for different model formats

### Preprocessing Pipeline
The preprocessing exactly matches the training pipeline:
- Resize to 128x128 pixels
- Normalize to [0,1] range (rescale = 1/255)
- RGB color format
- Face detection with 95% confidence threshold
- 30% margin around detected faces

### Output Format

#### Image Prediction Result
```json
{
  "filename": "image.jpg",
  "face_detected": true,
  "prediction": 0.8234,
  "confidence": 0.8234,
  "classification": "Real"
}
```

#### Video Prediction Result
```json
{
  "filename": "video.mp4",
  "total_frames": 1500,
  "analyzed_frames": 50,
  "faces_detected": 45,
  "average_prediction": 0.7123,
  "classification": "Real",
  "confidence": 0.7123,
  "frame_predictions": [0.8, 0.7, 0.9, ...]
}
```

## Integration Examples

### Python Integration
```python
from predict import DeepFakePredictor

# Initialize predictor
predictor = DeepFakePredictor()

# Predict single image
result = predictor.predict_image('photo.jpg')

if result['face_detected']:
    if result['classification'] == 'Fake':
        print(f"⚠️ Potential deepfake detected! Confidence: {result['confidence']:.2%}")
    else:
        print(f"✅ Image appears real. Confidence: {result['confidence']:.2%}")
```

### Batch Processing
```python
# Process multiple files
files = ['image1.jpg', 'image2.jpg', 'video.mp4']
results = predictor.predict_batch(files, output_file='results.json')

# Filter potential deepfakes
deepfakes = [r for r in results if r.get('classification') == 'Fake']
```

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model file not found at ./tmp_checkpoint/best_model.h5
   ```
   **Solution**: Train the model first using `python 03-train_cnn.py`

2. **EfficientNet import error**
   ```
   Warning: EfficientNet not found. Model loading may fail...
   ```
   **Solution**: Install EfficientNet: `pip install efficientnet`

3. **No faces detected**
   ```
   Result: No face detected
   ```
   **Solution**: Ensure images contain clear, frontal faces. Check image quality and lighting.

4. **MTCNN errors**
   ```
   Error: MTCNN initialization failed
   ```
   **Solution**: Install MTCNN: `pip install mtcnn`

### Performance Tips

1. **For videos**: Use larger frame intervals (e.g., `--frame-interval 60`) for faster processing
2. **For batch processing**: Process files in smaller batches to avoid memory issues
3. **GPU acceleration**: Ensure TensorFlow can access GPU for faster inference

## System Requirements

### Minimum Requirements
- Python 3.6+
- 4GB RAM
- 1GB free disk space

### Recommended Requirements
- Python 3.8+
- 8GB RAM
- GPU with CUDA support
- SSD storage

### Dependencies
All dependencies are listed in `requirements.txt`:
- tensorflow
- opencv-python
- mtcnn
- numpy
- pandas
- efficientnet
- h5py

## Support

### Getting Help
1. Run `python deploy.py --check` to verify system status
2. Run `python integration_test.py` to test all components
3. Check the error messages for specific guidance
4. Review the examples in `example_predict.py`

### Reporting Issues
When reporting issues, please include:
- Python version
- Operating system
- Error messages
- Steps to reproduce
- Output of `python deploy.py --check`

## Next Steps

After successfully setting up the prediction system:

1. **Test with sample data**: Use the provided sample videos to verify functionality
2. **Integrate into your workflow**: Use the Python API for custom applications
3. **Deploy for production**: Consider the deployment script for user-friendly access
4. **Monitor performance**: Track prediction accuracy and processing speed

For more detailed examples and advanced usage, see `example_predict.py` and the updated README.md.