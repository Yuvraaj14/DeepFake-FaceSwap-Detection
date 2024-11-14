# DeepFake Detection using CNN-LSTM

This repository provides a comprehensive deep learning-based approach to detecting deepfake videos. The system leverages Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to analyze video frames and identify whether a video is real or manipulated (deepfake). The project consists of multiple stages, including frame extraction, data preprocessing, model training, and evaluation on unseen videos.

## Table of Contents
- [Project Overview](#project-overview)
- [Technologies and Libraries](#technologies-and-libraries)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Folder Structure](#folder-structure)
- [Data Preparation](#data-preparation)
  - [Frame Extraction from Videos](#frame-extraction-from-videos)
  - [Organizing the Data](#organizing-the-data)
- [Model Details](#model-details)
  - [CNN-LSTM Architecture](#cnn-lstm-architecture)
- [Training and Evaluation](#training-and-evaluation)
  - [Training the Model](#training-the-model)
  - [Evaluating the Model](#evaluating-the-model)
- [How to Use](#how-to-use)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Project Overview

Deepfake videos are synthetic media created using AI technologies like GANs (Generative Adversarial Networks) or autoencoders. These videos can manipulate a person's face or actions and are difficult to detect with traditional methods. This project proposes a solution that classifies videos into two categories: **Real** or **Manipulated** (Deepfake) by analyzing sequences of video frames. The solution uses a combination of CNN for feature extraction from individual frames and LSTM to capture temporal dependencies across multiple frames.

### Key Features:
- **Frame Extraction**: Extract frames from videos at a fixed rate to process and analyze.
- **Preprocessing**: Resize, normalize, and transform video frames to prepare for model training.
- **CNN-LSTM Model**: A hybrid architecture combining CNN for spatial feature extraction and LSTM for learning temporal dependencies.
- **Evaluation**: Evaluate the model's performance on unseen videos and report the classification accuracy.

## Technologies and Libraries

This project uses several cutting-edge libraries and technologies to build and evaluate the deepfake detection system:

- **PyTorch**: The primary framework for deep learning, used to build the CNN-LSTM model.
- **OpenCV**: A powerful library used to process video files and extract frames.
- **PIL (Python Imaging Library)**: Used for basic image operations like resizing and format conversion.
- **NumPy**: For numerical operations such as tensor manipulations.
- **scikit-learn**: For evaluating the model's classification performance.

## Getting Started

### Prerequisites

Before you start, you will need:
- **Python 3.7+**: The code is developed and tested on Python 3.7 and later versions.
- **CUDA (Optional)**: If you plan to use GPU acceleration, ensure that you have CUDA set up with PyTorch.

### Setting Up the Environment

You can set up a virtual environment to install the required dependencies. Here's a step-by-step guide:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/DeepFakeDetection.git
   cd DeepFakeDetection

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   
4. **Set up the data**: Follow the instructions in the Data Preparation section.

## Folder Structure
The project follows a structured approach to organizing files and directories. Here's the layout:
```bash
DeepFakeDetection/
├── Data/ # Raw video files 
│ ├── train/ # Training dataset 
│ │ ├── real/ # Real videos 
│ │ └── manipulated/ # Manipulated (Deepfake) videos 
│ └── test/ # Testing dataset 
├── DataFrames/ # Extracted frames from videos 
│ ├── train/ # Frames for training 
│ └── test/ # Frames for testing 
├── CNN_LSTM.py # Model definition (CNN-LSTM architecture) 
├── FrameExtraction.py # Frame extraction script 
├── TestPreprocessing.py # Preprocessing for unseen data 
├── train.py # Script to train the model 
├── requirements.txt # Python dependencies
├── README.md # Project documentation 
└── utils/ # Helper functions for preprocessing and evaluation
```

## Data Preparation
To use the model, you first need to prepare the data. This involves extracting frames from video files and organizing them for training and evaluation.

### Frame Extraction from Videos
The first step is to extract frames from video files. The FrameExtraction.py script is used to convert video files into images (frames) at a specified frame rate (fps). For example, you can extract one frame every second.

Run the following command to extract frames:
```bash
python FrameExtraction.py
```
This will extract frames from both the train/real/, train/manipulated/ folders, as well as from the test set, and store them in the DataFrames/ directory.

## Organizing the Data
After extracting frames, your data will be organized as follows:
```bash
DataFrames/
├── train/
│   ├── real/
│   └── manipulated/
├── test/
│   ├── real/
│   └── manipulated/
```
Each folder contains the frames of video files, where each frame is saved as a .jpg image.

## Model Details

### CNN-LSTM Architecture
The core of this project is the CNN-LSTM model, which is designed to classify whether a video is real or manipulated by analyzing sequential frames. The architecture consists of two main components:

1. **CNN for Feature Extraction**: A pre-trained CNN (such as ResNet50) is used to extract spatial features from individual frames. The CNN learns to identify visual patterns like faces, backgrounds, and motion details.

2. **LSTM for Temporal Modeling**: After extracting features from individual frames, these features are fed into an LSTM network. The LSTM captures the temporal dependencies between frames, which is critical for detecting manipulated videos that may exhibit subtle inconsistencies over time.

The model is trained on sequences of frames, and the final output is a classification indicating whether the video is real or manipulated.

## Training and Evaluation
### Training the Model
To train the model, run the train.py script. This will initialize the CNN-LSTM model, load the training data, and train the model over multiple epochs.
```bash
python train.py
```
This command will:

- Load the training data (video frames).
- Preprocess the frames using the defined transformations (resize, normalize).
- Train the model and save the trained weights to a file for later use.

### Evaluating the Model
To evaluate the trained model, use the test.py script. This script loads the trained model and evaluates it on unseen video data from the test set.
```bash
python test.py
```
The evaluation will provide accuracy metrics, including the confusion matrix and other relevant metrics like precision and recall.

## How to Use
1. **Prepare the Data**: Extract frames from the videos using FrameExtraction.py.
2. **Train the Model**: Use the train.py script to train the model on your dataset.
3. **Evaluate the Model**: Use test.py to evaluate the model on unseen data.

## Contributing
We welcome contributions to improve the performance of this project. Here are some ways you can contribute:

- Improving the model architecture.
- Adding additional data augmentation techniques.
- Contributing to the documentation.
If you'd like to contribute, fork this repository and submit a pull request. Please ensure that your contributions are well-documented, and if you're adding new features, include test cases where applicable.

## Acknowledgements
- PyTorch: For building and training deep learning models.
- OpenCV: For video processing and frame extraction.
- DeepFake Detection Community: For sharing resources and datasets to facilitate research in deepfake detection.
