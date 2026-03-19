<div align="center">

# DeepFake-Detect

**Open-source deepfake detection & face forgery detection — train your own model with TensorFlow, Keras & EfficientNet**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.2+-D00000?logo=keras)](https://keras.io/)
[![GitHub stars](https://img.shields.io/github/stars/aaronchong888/DeepFake-Detect?style=social)](https://github.com/aaronchong888/DeepFake-Detect/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aaronchong888/DeepFake-Detect?style=social)](https://github.com/aaronchong888/DeepFake-Detect/network/members)

[**Live Demo**](https://deepfake-detect.com/) · [Report Bug](https://github.com/aaronchong888/DeepFake-Detect/issues) · [Request Feature](https://github.com/aaronchong888/DeepFake-Detect/issues)

</div>

---

## Table of Contents

- [About](#about)
- [Features](#features)
- [Use Cases](#use-cases)
- [Demo](#demo)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Training Datasets](#training-datasets)
- [Project Structure](#project-structure)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Citing](#citing)
- [Authors & License](#authors--license)

---

## About

**DeepFake-Detect** is an open-source pipeline for training **deepfake detection** and **face forgery detection** models from scratch. Built with [Python](https://www.python.org), [Keras](https://keras.io), and [TensorFlow](https://www.tensorflow.org), the detector uses an **EfficientNet** backbone and is trained on major public benchmarks (FaceForensics++, Celeb-DF, DFDC, and others) to recognize synthetic faces and manipulated media.

Try it in your browser: **[DF-Detect](https://deepfake-detect.com/)** — a free web app to detect deepfake images and videos.

---

## Features

- **EfficientNet-based architecture** — State-of-the-art backbone with 128×128 input, global max pooling, and binary classification head (pristine vs deepfake).
- **Multi-dataset training** — Supports five major public benchmarks (FaceForensics++, Celeb-DF, DFDC, DFD, DeepFake-TIMIT) for robustness across ~20 synthesis methods.
- **End-to-end pipeline** — From raw videos to trained model: frame extraction → face cropping (MTCNN or Azure Vision API) → dataset balancing & split → training.
- **Live web demo** — Try the model at [deepfake-detect.com](https://deepfake-detect.com/) without installing anything.

---

## Use Cases

- **Researchers & students** — Reproduce or extend deepfake detection experiments; train on custom datasets.
- **Developers** — Integrate a trained model into apps or pipelines; use the scripts as a starting point.
- **Media & fact-checkers** — Use the [live demo](https://deepfake-detect.com/) to quickly screen images or frames for potential manipulation.
- **Educators** — Teach deep learning and synthetic media detection with a full, runnable pipeline.

---

## Demo

<p align="center">
  <img src="img/demo.gif" alt="DeepFake-Detect demo: upload image and get deepfake detection score" width="85%" />
</p>

<p align="center">
  <strong><a href="https://deepfake-detect.com/">Try the live demo → deepfake-detect.com</a></strong>
</p>

<p align="center">
  <a href="https://deepfake-detect.com/"><img src="img/dfdetect-home.png" alt="DF-Detect deepfake detection web app homepage" width="70%" /></a>
</p>

---

## Quick Start

### Prerequisites

- **Python 3**
- **TensorFlow** & **Keras**
- **EfficientNet** (TensorFlow/Keras), **OpenCV**, **MTCNN** (or Azure Computer Vision for cloud-based face cropping)

### Install & run pipeline

```bash
# Clone and install dependencies
git clone https://github.com/aaronchong888/DeepFake-Detect.git
cd DeepFake-Detect
pip install -r requirements.txt

# Run the full pipeline (after placing your dataset videos as expected by the scripts)
python 00-convert_video_to_image.py    # Extract frames
python 01a-crop_faces_with_mtcnn.py    # Crop faces (or 01b for Azure)
python 02-prepare_fake_real_dataset.py # Balance & split train/val/test
python 03-train_cnn.py                 # Train EfficientNet classifier
```

---

## Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| **0** | `00-convert_video_to_image.py` | Extract frames from videos; resize by width (2× if &lt;300px, 1× for 300–1000px, 0.5× for 1000–1900px, 0.33× if &gt;1900px). |
| **1a** | `01a-crop_faces_with_mtcnn.py` | Crop faces with [MTCNN](https://github.com/ipazc/mtcnn) (30% margin, 95% confidence). Multiple faces per frame saved separately. |
| **1b** | `01b-crop_faces_with_azure-vision-api.py` | Optional: use [Azure Computer Vision API](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) for face cropping (set API name & key in script). |
| **2** | `02-prepare_fake_real_dataset.py` | Down-sample fakes to match real count; split into train/val/test (e.g. 80:10:10). |
| **3** | `03-train_cnn.py` | Train EfficientNet B0 backbone → global max pooling → 2× FC (ReLU) → sigmoid. Input 128×128 RGB; output probability pristine (1) vs deepfake (0). |

---

## Training Datasets

The model is trained on the following public deepfake datasets to cover diverse identities and synthesis methods:

| Dataset | Link |
|---------|------|
| DeepFake-TIMIT | [https://www.idiap.ch/dataset/deepfaketimit](https://www.idiap.ch/dataset/deepfaketimit) |
| FaceForensics++ | [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics) |
| Google DFD | [https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html) |
| Celeb-DF | [https://github.com/danmohaha/celeb-deepfakeforensics](https://github.com/danmohaha/celeb-deepfakeforensics) |
| Facebook DFDC | [https://ai.facebook.com/datasets/dfdc/](https://ai.facebook.com/datasets/dfdc/) |

**Aggregate scale (approximate):** ~134,446 videos · ~1,140 identities · ~20 synthesis methods.

<p align="center">
  <img src="img/sample_dataset.png" alt="DeepFake-Detect training dataset sample: real vs deepfake face images" width="85%" />
</p>

---

## Project Structure

```
DeepFake-Detect/
├── 00-convert_video_to_image.py   # Video → frame images
├── 01a-crop_faces_with_mtcnn.py   # Face cropping (MTCNN)
├── 01b-crop_faces_with_azure-vision-api.py  # Face cropping (Azure)
├── 02-prepare_fake_real_dataset.py           # Balance & split
├── 03-train_cnn.py               # Train EfficientNet classifier
├── requirements.txt
├── img/                          # Screenshots & demo assets
└── README.md
```

---

## FAQ

**How do I detect if an image is a deepfake?**  
Use the [live demo](https://deepfake-detect.com/) or run the trained model on a face crop (128×128). The model outputs a score: higher = more likely pristine, lower = more likely synthetic.

**Can I train on my own deepfake dataset?**  
Yes. Follow the pipeline: put videos in the expected layout, run the scripts in order (frame extraction → face crop → prepare dataset → train). You can mix your data with the public datasets.

**What deepfake methods does this detect?**  
The default model is trained on ~20 methods across FaceForensics++, Celeb-DF, DFDC, DFD, and DeepFake-TIMIT, so it generalizes to many common face-swap and manipulation techniques.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=aaronchong888/DeepFake-Detect&type=Date)](https://star-history.com/#aaronchong888/DeepFake-Detect&Date)

---

## Contributing

Contributions are welcome. Please open an [issue](https://github.com/aaronchong888/DeepFake-Detect/issues) or submit a [pull request](https://github.com/aaronchong888/DeepFake-Detect/pulls).

---

## Citing

If you use DeepFake-Detect in research or a project, please cite:

```bibtex
@software{deepfake_detect,
  title = {DeepFake-Detect: Open-Source Deepfake Detection Pipeline},
  author = {Chong, Aaron and Ng, See Long Hugo},
  year = {2020},
  url = {https://github.com/aaronchong888/DeepFake-Detect},
  note = {Train deepfake detection models with TensorFlow, Keras \& EfficientNet}
}
```

---

## Authors & License

- **[Aaron Chong](https://github.com/aaronchong888)** — *Initial work*
- **[Hugo Ng](https://github.com/hugoclong)** — *Initial work*

See [contributors](https://github.com/aaronchong888/DeepFake-Detect/contributors) for the full list.

**License:** [MIT](LICENSE).

**Acknowledgments:** Dependencies are listed in the [dependency graph](https://github.com/aaronchong888/DeepFake-Detect/network/dependencies).


