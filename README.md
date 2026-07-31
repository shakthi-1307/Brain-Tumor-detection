# 🧠 Brain Tumor Detection using Deep Learning

A web-based **Brain Tumor Detection System** built using **PyTorch** and **Flask** that classifies brain MRI images into one of four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The application allows users to upload an MRI scan through a simple web interface and instantly predicts the tumor type along with the model's confidence score.

---

# 📌 Table of Contents

- Overview
- Features
- Project Architecture
- Technology Stack
- Dataset
- Model Architecture
- Folder Structure
- Installation
- Running the Project
- Usage
- Prediction Pipeline
- Model Performance
- Future Improvements
- Troubleshooting
- License

---

# 📖 Overview

Brain tumors are one of the most critical neurological disorders that require early diagnosis for effective treatment.

This project leverages **Convolutional Neural Networks (CNNs)** to automatically classify MRI brain scans into different tumor categories.

The trained model is deployed using **Flask**, allowing users to upload MRI images directly from a browser and receive predictions in real time.

---

# ✨ Features

- Upload MRI brain scan images
- Detect brain tumors using a trained CNN
- Classifies into **4 classes**
  - Glioma
  - Meningioma
  - Pituitary
  - No Tumor
- Displays prediction confidence
- Simple and responsive web interface
- PyTorch-based inference
- Fast prediction on CPU

---

# 🏗️ Project Architecture

```
               MRI Image
                   │
                   ▼
           Image Upload (Flask)
                   │
                   ▼
        Image Preprocessing
    (Resize + Normalize + Tensor)
                   │
                   ▼
          Trained CNN Model
             (PyTorch)
                   │
                   ▼
      Softmax Probability Scores
                   │
                   ▼
     Highest Probability Selected
                   │
                   ▼
 Prediction + Confidence Display
```

---

# 🛠 Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| PyTorch | Deep Learning Framework |
| Flask | Web Framework |
| Torchvision | Image Transformations |
| PIL (Pillow) | Image Processing |
| HTML | Frontend |
| CSS | Styling |
| Base64 | Image Rendering |

---

# 📂 Dataset

The project uses MRI brain scan images organized into four categories.

```
Training/
│
├── glioma
├── meningioma
├── pituitary
└── notumor

Testing/
│
├── glioma
├── meningioma
├── pituitary
└── notumor
```

### Classes

| Label | Description |
|--------|-------------|
| Glioma | Cancer originating from glial cells |
| Meningioma | Tumor formed in the meninges |
| Pituitary | Tumor affecting the pituitary gland |
| No Tumor | Healthy MRI image |

---

# 🧠 CNN Model Architecture

The model is implemented using **PyTorch**.

### Feature Extraction

```
Input Image
      │
Conv2D (16 filters)
      │
ReLU
      │
MaxPooling
      │
Conv2D (32 filters)
      │
ReLU
      │
MaxPooling
      │
Conv2D (64 filters)
      │
ReLU
      │
MaxPooling
      │
Conv2D (128 filters)
      │
ReLU
      │
MaxPooling
```

### Classification

```
Flatten
    │
Fully Connected (512)
    │
ReLU
    │
Dropout (0.5)
    │
Output Layer (4 Classes)
```

---

# 📁 Project Structure

```
Brain-Tumor-detection/

│
├── brain_tumor_app.py          # Flask application
├── brain_tumor_model.pth       # Trained CNN model
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
│
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Brain-Tumor-detection.git

cd Brain-Tumor-detection
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install flask
pip install torch torchvision
pip install pillow
```

or

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Start the Flask server

```bash
python brain_tumor_app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

# 📸 Using the Application

1. Open the application in your browser.
2. Upload an MRI image.
3. Click **Predict**.
4. Wait for the model to process the image.
5. View:

- Predicted Tumor Type
- Prediction Confidence
- Uploaded MRI Image

---

# 🔄 Prediction Pipeline

### Step 1

User uploads MRI image.

↓

### Step 2

Flask receives the image.

↓

### Step 3

Image is converted to RGB.

↓

### Step 4

Image resized to

```
256 × 256
```

↓

### Step 5

Normalize pixel values using ImageNet statistics.

↓

### Step 6

Convert image into Tensor.

↓

### Step 7

CNN extracts features.

↓

### Step 8

Softmax calculates class probabilities.

↓

### Step 9

Highest probability class becomes the prediction.

↓

### Step 10

Result is displayed with confidence.

---

# 🖼 Image Preprocessing

Before prediction, every image undergoes the following transformations:

- Resize to **256 × 256**
- Convert to RGB
- Convert to Tensor
- Normalize using:

```
Mean:
0.485
0.456
0.406

Standard Deviation:
0.229
0.224
0.225
```

These preprocessing steps help improve the model's prediction consistency.

---

# 🎯 Output

Example prediction:

```
Prediction

Pituitary Tumor

Confidence

98.64%
```

---

# 🚀 Future Improvements

- Train using transfer learning (ResNet, EfficientNet, DenseNet)
- Add Grad-CAM visualization for explainability
- Display probability for all classes
- Upload DICOM (.dcm) medical images
- REST API for external applications
- Docker support
- Cloud deployment (AWS, Azure, GCP)
- User authentication and prediction history
- Mobile-responsive UI improvements

---

# ⚠ Known Issue

The model is currently loaded using an absolute Windows path:

```python
model.load_state_dict(torch.load(
    r"D:\brain-tumor-detection\brain_tumor_model.pth",
    map_location=torch.device('cpu')
))
```

Replace it with a relative path for portability:

```python
model.load_state_dict(
    torch.load("brain_tumor_model.pth", map_location=torch.device("cpu"))
)
```

This ensures the project works on any machine without modifying the source code.

---

# 📚 Learning Outcomes

Through this project, the following concepts are demonstrated:

- Deep Learning using CNNs
- Image Classification
- Medical Image Processing
- PyTorch Model Inference
- Flask Web Development
- Image Preprocessing
- Deployment of Machine Learning Models
- End-to-End AI Application Development

---

# 📄 License

This project is intended for **educational and research purposes only** and should **not** be used as a substitute for professional medical diagnosis.

Always consult qualified healthcare professionals for clinical decisions.

---

# 👨‍💻 Author

**Shakthi Chellappan**

- GitHub: https://github.com/shakthi-1307

If you found this project helpful, consider giving it a ⭐ on GitHub.