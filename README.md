# DeepFake-Detect

<p align="center"><a href="https://deepfake-detect.com/"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/dfdetect-home.png" width="60%"></a></p>

<p align="center"><a href="https://deepfake-detect.com/">https://deepfake-detect.com/</a></p>

## Description

This project aims to guide developers to train a deep learning-based deepfake detection model from scratch using [Python](https://www.python.org), [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org). The proposed deepfake detector is based on the state-of-the-art EfficientNet structure with some customizations on the network layers, and the sample models provided were trained against a massive and comprehensive set of deepfake datasets. 

The proposed deepfake detection model is also served via a standard web-based interface at [DF-Detect](https://deepfake-detect.com/) to assist both the general Internet users and digital media providers in identifying potential deepfake contents. It is hoped that such approachable solution could remind Internet users to stay vigilant against fake contents, and ultimately help counter the emergence of deepfakes.

### Deepfake Datasets

Due to the nature of deep neural networks being data-driven, it is necessary to acquire massive deepfake datasets with various different synthesis methods in order to achieve promising results. The following deepfake datasets were used in the final model at [DF-Detect](https://deepfake-detect.com/):

- [DeepFake-TIMIT](https://www.idiap.ch/dataset/deepfaketimit)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Google Deep Fake Detection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
- [Celeb-DF](https://github.com/danmohaha/celeb-deepfakeforensics)
- [Facebook Deepfake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)

<p align="center"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/sample_dataset.png" width="80%"></p>

Combining all the datasets from different sources would provide us a total of 134,446 videos with approximately 1,140 unique identities and around 20 deepfake synthesis methods.

<br>

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Keras
- TensorFlow
- EfficientNet for TensorFlow Keras
- OpenCV on Wheels
- MTCNN

### Installation

```
pip install -r requirements.txt
```

### Quick Start for Prediction

If you just want to use the prediction system with a pre-trained model:

```bash
# Setup the prediction environment
python setup_prediction.py

# Check if everything is ready
python deploy.py --check

# Start making predictions (interactive mode)
python deploy.py --interactive
```

**Note**: You'll need a trained model first. If you don't have one, follow the training steps below.

### Usage

#### Step 0 - Convert video frames to individual images

```
python 00-convert_video_to_image.py
```

Extract all the video frames from the acquired deepfake datasets above, saving them as individual images for further processing. In order to cater for different video qualities and to optimize for the image processing performance, the following image resizing strategies were implemented:

- 2x resize for videos with width less than 300 pixels
- 1x resize for videos with width between 300 and 1000 pixels
- 0.5x resize for videos with width between 1000 and 1900 pixels
- 0.33x resize for videos with width greater than 1900 pixels

#### Step 1 - Extract faces from the deepfake images with MTCNN

```
python 01a-crop_faces_with_mtcnn.py
```

Further process the frame images to crop out the facial parts in order to allow the neural network to focus on capturing the facial manipulation artifacts. In cases where there are more than one subject appearing in the same video frame, each detection result is saved separately to provide better variety for the training dataset.

- The pre-trained MTCNN model used is coming from this GitHub repo: https://github.com/ipazc/mtcnn
- Added 30% margins from each side of the detected face bounding box
- Used 95% as the confidence threshold to capture the face images

#### (Optional) Step 1b - Extract faces from the deepfake images with Azure Computer Vision API

In case you do not have a good enough hardware to run MTCNN, or you want to achieve a faster execution time, you may choose to run **01b** instead of **01a** to leverage the [Azure Computer Vision API](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) for facial recognition.

```
python 01b-crop_faces_with_azure-vision-api.py
```

> Replace the missing parts (*API Name* & *API Key*) before running

#### Step 2 - Balance and split datasets into various folders

```
python 02-prepare_fake_real_dataset.py
```

As we observed that the number of fakes are much larger than the number of real faces (due to the fact that one real video is usually used for creating multiple deepfakes), we need to perform a down-sampling on the fake dataset based on the number of real crops, in order to tackle for possible class imbalance issues during the training phase. 

We also need to split the dataset into training, validation and testing sets (for example, in the ratio of 80:10:10) as the final step in the data preparation phase.

#### Step 3 - Model training

```
python 03-train_cnn.py
```

EfficientNet is used as the backbone for the development work. Given that most of the deepfake videos are synthesized using a frame-by-frame approach, we have formulated the deepfake detection task as a binary classification problem such that it would be generally applicable to both video and image contents.

In this code sample, we have adapted the EfficientNet B0 model in several ways: The top input layer is replaced by an input size of 128x128 with a depth of 3, and the last convolutional output from B0 is fed to a global max pooling layer. In addition, 2 additional fully connected layers have been introduced with ReLU activations, followed by a final output layer with Sigmoid activation to serve as a binary classifier. 

Thus, given a colored square image as the network input, we would expect the model to compute an output between 0 and 1 that indicates the probability of the input image being either deepfake (0) or pristine (1).

#### Step 4 - Model prediction and deployment

After training the model, you can use it to make predictions on new images and videos:

##### Option 1: Using the deployment script (Recommended)

```
python deploy.py --interactive
```

This will launch an interactive mode where you can easily select files to analyze.

##### Option 2: Direct prediction script

```
python 04-predict.py image.jpg
python 04-predict.py video.mp4
python 04-predict.py *.jpg --output results.json
```

##### Option 3: Quick deployment check

```
python deploy.py --check
```

This will verify that your model and dependencies are ready for prediction.

##### Prediction Features

- **Image Analysis**: Detects faces in images and classifies them as real or deepfake
- **Video Analysis**: Analyzes video frames at specified intervals to detect deepfakes
- **Batch Processing**: Process multiple files at once
- **JSON Output**: Save detailed results to JSON files for further analysis
- **Confidence Scores**: Get probability scores along with classifications

##### Example Usage

```bash
# Interactive mode (easiest)
python deploy.py --interactive

# Predict a single image
python deploy.py photo.jpg

# Predict multiple files with JSON output
python deploy.py video1.mp4 video2.mp4 image1.jpg --output results.json

# Use custom model path
python 04-predict.py input.jpg --model /path/to/custom/model.h5

# Analyze video with custom frame interval
python 04-predict.py video.mp4 --frame-interval 15
```

The prediction system will:
1. Detect faces in the input using MTCNN
2. Crop and preprocess the detected faces
3. Run the trained model to get predictions
4. Return classification results with confidence scores

## Project Files

### Training Pipeline
- `00-convert_video_to_image.py` - Extract frames from videos
- `01a-crop_faces_with_mtcnn.py` - Extract faces using MTCNN
- `01b-crop_faces_with_azure-vision-api.py` - Extract faces using Azure API
- `02-prepare_fake_real_dataset.py` - Balance and split dataset
- `03-train_cnn.py` - Train the deepfake detection model

### Prediction System
- `04-predict.py` - Main prediction script for images and videos
- `deploy.py` - Easy-to-use deployment script with interactive mode
- `setup_prediction.py` - Setup and verify prediction environment
- `test_prediction.py` - Test the prediction system
- `example_predict.py` - Usage examples and integration guide

### Configuration
- `requirements.txt` - Python dependencies
- `prediction_config.json` - Prediction system configuration (created by setup)
- `batch_files_example.txt` - Example for batch processing (created by setup)

## Authors

* **Aaron Chong** - *Initial work* - [aaronchong888](https://github.com/aaronchong888)
* **Hugo Ng** - *Initial work* - [hugoclong](https://github.com/hugoclong)

See also the list of [contributors](https://github.com/aaronchong888/DeepFake-Detect/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project is built using the following packages and libraries as listed [here](https://github.com/aaronchong888/DeepFake-Detect/network/dependencies)