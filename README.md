# DeepFake-Detect

<p align="center"><a href="http://deepfakedetection.eastasia.cloudapp.azure.com"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/dfdetect-home.png" width="60%"></a></p>

## Description

This project aims to guide developers to train a deep learning-based deepfake detection model from scratch using [Python](https://www.python.org), [Keras](https://keras.io) and [TensorFlow](https://www.tensorflow.org). The proposed deepfake detector is based on the state-of-the-art EfficientNet structure with some customizations on the network layers, and the sample models provided were trained against a massive and comprehensive set of deepfake datasets. 

The proposed deepfake detection model is also served via a standard web-based interface at [DF-Detect](http://deepfakedetection.eastasia.cloudapp.azure.com) to assist both the general Internet users and digital media providers in identifying potential deepfake contents. It is hoped that such approachable solution could remind Internet users to stay vigilant against fake contents, and ultimately help counter the emergence of deepfakes.

### Deepfake Datasets

Due to the nature of deep neural networks being data-driven, it is necessary to acquire massive deepfake datasets with various different synthesis methods in order to achieve promising results. The following deepfake datasets were used in the final model at [DF-Detect](http://deepfakedetection.eastasia.cloudapp.azure.com):

- [DeepFake-TIMIT](https://www.idiap.ch/dataset/deepfaketimit)
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Google Deep Fake Detection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
- [Celeb-DF](https://github.com/danmohaha/celeb-deepfakeforensics)
- [Facebook Deepfake Detection Challenge (DFDC)](https://ai.facebook.com/datasets/dfdc/)

<p align="center"><a href="http://deepfakedetection.eastasia.cloudapp.azure.com"><img alt="" src="https://github.com/aaronchong888/DeepFake-Detect/blob/master/img/sample_dataset.png" width="80%"></a></p>

Combining all the datasets from different generations would provide us a total of 134,446 videos with approximately 1,140 unique identities and around 20 deepfake synthesis methods.

## Getting Started

### Prerequisites

- Python 3

> To be updated

### Installation

```
pip install
```

> To be updated

### Deployment

> To be updated

### Usage

> To be updated

## Authors

* **Aaron Chong** - *Initial work* - [aaronchong888](https://github.com/aaronchong888)
* **Hugo Ng** - *Initial work* - [hugoclong](https://github.com/hugoclong)

See also the list of [contributors](https://github.com/aaronchong888/DeepFake-Detect/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

This project is built using the following packages and libraries as listed [here](https://github.com/aaronchong888/DeepFake-Detect/network/dependencies)