from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io
import base64
import os
from pathlib import Path

# Ensure the templates folder exists
templates_dir = Path(__file__).parent / 'templates'
templates_dir.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(templates_dir))

# Define the same CNN model architecture
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = BrainTumorCNN(num_classes=4)
model.load_state_dict(torch.load(r"D:\brain-tumor-detection\brain_tumor_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/')
def index():
    # CORRECTED: Use just the template filename, not the full path
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Read and preprocess the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = class_names[predicted.item()]
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100
        
        # Convert image to base64 for display
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        # Format confidence to 2 decimal places before passing to template
        formatted_confidence = "{:.2f}".format(confidence)
        
        # CORRECTED: Use just the template filename, not the full path
        return render_template("result.html", 
                         prediction=prediction, 
                         confidence=formatted_confidence,
                         img_data=img_base64)

if __name__ == '__main__':
    app.run(debug=True)