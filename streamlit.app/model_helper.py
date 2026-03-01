import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import os

trained_model = None

# Internal class names (must match training order)
class_names = [
    'F_Breakage',
    'F_Crushed',
    'F_Normal',
    'R_Breakage',
    'R_Crushed',
    'R_Normal'
]

# Friendly names for UI
label_map = {
    'F_Breakage': 'Front Breakage',
    'F_Crushed': 'Front Crushed',
    'F_Normal': 'Front Normal',
    'R_Breakage': 'Rear Breakage',
    'R_Crushed': 'Rear Crushed',
    'R_Normal': 'Rear Normal'
}


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.model = models.resnet50(weights='DEFAULT')

        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace final layer
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    global trained_model

    if trained_model is None:
        trained_model = CarClassifierResNet()

        # 🔥 SAFE MODEL PATH (works in Streamlit Cloud)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.pth")

        trained_model.load_state_dict(
            torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        )
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        raw_label = class_names[predicted_class]
        friendly_label = label_map[raw_label]

        return friendly_label
