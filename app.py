import os
import cv2
import torch
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import sys
import subprocess

required_packages = ["streamlit", "torch", "torchvision", "opencv-python", "numpy", "matplotlib", "PIL"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import streamlit as st  # Now import after installing dependencies


# Load CSRNet Model
class CSRNet(torch.nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.frontend = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.backend = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, kernel_size=3, dilation=2, padding=2), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x

# Load Pre-trained Model
model = CSRNet()
model_path = "csrnet.pth"  # Update this path if needed
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
else:
    st.error(f"Model file '{model_path}' not found!")

# Streamlit App Title
st.title("Crowd Counting using CSRNet")

# File Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert PIL Image to OpenCV Format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Transform Image for CSRNet
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    # Model Prediction
    with torch.no_grad():
        output = model(img_tensor)

    density_map = output.squeeze(0).squeeze(0).cpu().numpy()
    predicted_count = np.sum(density_map)

    # Display Predicted Count
    st.subheader(f"Predicted Crowd Count: {int(predicted_count // 10)}")

    # Display Density Map
    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap="jet")
    ax.set_title(f"Density Map (Predicted Count: {int(predicted_count // 10)})")
    plt.colorbar(ax.imshow(density_map, cmap="jet"))
    
    st.pyplot(fig)
