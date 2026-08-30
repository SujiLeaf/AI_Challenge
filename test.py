import os
import torch
import torch.nn as nn
import pandas as pd

from PIL import Image
from torchvision import datasets, transforms, models

# ==========================================
# DEVICE
# ==========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================================
# TRANSFORM
# (HARUS SAMA DENGAN TRAINING)
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# LOAD TRAIN DATASET
# (UNTUK MENGETAHUI JUMLAH CLASS)
# ==========================================
train_dataset = datasets.ImageFolder(
    root="train",
    transform=transform
)

num_classes = len(train_dataset.classes)

print("Classes :", train_dataset.classes)
print("Number of classes :", num_classes)

# Mapping index -> label
idx_to_class = {
    value: key
    for key, value in train_dataset.class_to_idx.items()
}

# ==========================================
# LOAD MODEL
# ==========================================
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    torch.load(
        "resnet18_model.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded successfully!")

# ==========================================
# LOAD SAMPLE SUBMISSION
# ==========================================
submission = pd.read_csv("sample_submission")

print(f"\nTotal test images: {len(submission)}")

# ==========================================
# PREDICTION
# ==========================================
predictions = []

with torch.no_grad():

    for image_name in submission["id"]:

        image_path = os.path.join(
            "test",
            "test",
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(device)

        output = model(image)

        pred = torch.argmax(output, dim=1).item()

        label = idx_to_class[pred]

        predictions.append(label)

# ==========================================
# SAVE SUBMISSION
# ==========================================
submission["label"] = predictions

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nPrediction Finished!")
print("Submission saved as submission.csv")