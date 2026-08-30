import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==========================
# DEVICE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================
# DATA TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# LOAD DATASET
# ==========================
train_dataset = datasets.ImageFolder(
    root="train",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

print("Classes:", train_dataset.classes)
print("Number of classes:", len(train_dataset.classes))

# ==========================
# LOAD PRETRAINED RESNET18
# ==========================
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# Replace the last layer
num_classes = len(train_dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

# ==========================
# LOSS & OPTIMIZER
# ==========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)

# ==========================
# TRAINING
# ==========================
epochs = 10

for epoch in range(epochs):

    model.train()

    average_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        average_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# ==========================
# SAVE MODEL
# ==========================
torch.save(
    model.state_dict(),
    "resnet18_model.pth"
)

print("\nTraining Finished!")
print("Model saved as resnet18_model.pth")