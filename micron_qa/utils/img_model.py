# utils/img_model.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


# 1. Custom ResNet18 for 1-channel grayscale input
def ResNet18(num_classes=6):
    model = models.resnet18(weights=None)  # pretrained=False is deprecated
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# 2. Training loop
def train_model(model, data, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, _ = data  # data is a tuple of (train, val, test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1} Summary → Loss: {running_loss:.4f}, Accuracy: {train_acc:.2f}%")

        if val_loader:
            print("Validation Performance:")
            evaluate_model(model, (None, val_loader, None), params)
    
    # Save the trained image model
    torch.save(model, os.path.join(params['paths']['results'], "img_model.pth"))

    return model

def evaluate_model(model, data, params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    _, val_loader, test_loader = data

    loader = test_loader or val_loader  # Use whichever is provided
    if loader is None:
        print("No loader provided for evaluation.")
        return

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("[Evaluation Report]")
    print(confusion_matrix(all_labels, all_preds))
    print(classification_report(all_labels, all_preds, digits=4))

