import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from flask import Flask, request, render_template, send_from_directory
import matplotlib.pyplot as plt

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
model.load_state_dict(torch.load("csrnet.pth", map_location="cpu"))
model.eval()

# Initialize Flask App
app = Flask(__name__, static_folder="static")

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file:
            file_path = os.path.join("static", "input.jpg")
            file.save(file_path)
            count = process_image(file_path)
            return render_template("result.html", count=count)

    return render_template("upload.html")

def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((384, 512)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)

    density_map = output.squeeze(0).squeeze(0).cpu().numpy()
    predicted_count = np.sum(density_map)  # Sum of density map
    
    # 🛠 Debugging
    print("Predicted Count:", predicted_count)
    print("Density Map Shape:", density_map.shape)

    # Save Density Map
    plt.imshow(density_map, cmap="jet")
    plt.colorbar()
    plt.title(f"Predicted Count: {int(predicted_count//10)}")
    plt.savefig("static/result.png")
    plt.close()

    return int(predicted_count)

@app.route("/static/<filename>")
def send_file(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
