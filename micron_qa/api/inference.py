import joblib
import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml
import os

# Load config
with open("configs/params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load Tabular Model
tab_model_path = os.path.join(params['paths']['results'], "xgboost_model.pkl")
TAB_MODEL = joblib.load(tab_model_path)

# Load Image Model
img_model_path = os.path.join(params['paths']['results'], "img_model.pth")
IMG_MODEL = torch.load(img_model_path, map_location=torch.device("cpu"), weights_only=False)
IMG_MODEL.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_tabular(features):
    return int(TAB_MODEL.predict([features])[0])

def predict_image(img_file):
    image = Image.open(img_file).convert("L")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = IMG_MODEL(tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

