# Brain Tumor Detection

A deep learning model for detecting and classifying brain tumors from MRI images using PyTorch.

## Features

- Upload MRI images for tumor detection
- Classify tumors into 4 categories: glioma, meningioma, pituitary, or no tumor
- Web interface built with Flask

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python app.py`
4. Open your browser to `http://localhost:5000`

## Model

The model is a CNN architecture trained on brain MRI images.

## Dependencies

- torch
- torchvision
- flask
- numpy
