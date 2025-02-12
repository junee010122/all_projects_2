import torch
from utils.general import load_config
from utils.data import get_dataloaders
from utils.model import ResNet18
from utils.jmp_analysis import run_jmp_analysis
from utils.tableau_export import export_to_tableau

# Load config
config = load_config()
device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")

# Load dataset
train_loader, valid_loader, test_loader = get_dataloaders(config["data"]["dataset_path"], config["data"]["batch_size"], config["data"]["image_size"])

# Load trained model
model = ResNet18(num_classes=config["data"]["num_classes"], pretrained=False)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.to(device)
model.eval()

# Run classification & automation
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    for pred in preds.cpu().numpy():
        predicted_class = ["Crazing", "Inclusion", "Patches", "Pitted Surface", "Rolled-in Scale", "Scratches"][pred]
        
        # Run JMP Analysis
        run_jmp_analysis(predicted_class)

        # Export to Tableau
        export_to_tableau(predicted_class)

