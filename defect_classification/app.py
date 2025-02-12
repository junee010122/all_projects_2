import streamlit as st
import torch
import numpy as np
from PIL import Image
from utils.general import load_config
from utils.data import MetalDefectDataset
from utils.model import ResNet18
from utils.plots import plot_confusion_matrix, show_predictions
from utils.jmp_analysis import run_jmp_analysis
from utils.tableau_export import export_to_tableau

# Load config
config = load_config()
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# Load trained model
@st.cache_resource
def load_model():
    model = ResNet18(num_classes=config["data"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

st.title("Metal Defect Classification 🔬")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    img_resized = img.resize((config["data"]["image_size"], config["data"]["image_size"]))
    
    # Convert to tensor
    img_tensor = torch.tensor(np.array(img_resized) / 255.0).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    class_names = ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in Scale", "Scratches"]
    predicted_class = class_names[pred]

    st.image(img, caption=f"Prediction: {predicted_class}", use_column_width=True)
    
    # Run JMP Analysis
    jmp_results = run_jmp_analysis(predicted_class)
    st.write(f"JMP Analysis Results: {jmp_results}")

    # Export results to Tableau
    export_status = export_to_tableau(predicted_class)
    st.write(export_status)

