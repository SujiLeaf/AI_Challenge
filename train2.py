import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split
from collections import Counter

# ==========================================
# DEVICE
# ==========================================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device : {device}")

# ==========================================
# TRANSFORM
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
# LOAD DATASET
# ==========================================

dataset = datasets.ImageFolder(

    root="train",

    transform=transform

)

# ==========================================
# CLASS INFORMATION
# ==========================================

print("\n========== DATASET ==========")

print("Class names :")

for idx, name in enumerate(dataset.classes):

    print(f"{idx} -> {name}")

print(f"\nTotal images : {len(dataset)}")

# ==========================================
# STRATIFIED SPLIT
# ==========================================

targets = dataset.targets

indices = list(range(len(dataset)))

train_idx, val_idx = train_test_split(

    indices,

    test_size=0.20,

    stratify=targets,

    random_state=42

)

# ==========================================
# SUBSET
# ==========================================

train_dataset = Subset(

    dataset,

    train_idx

)

val_dataset = Subset(

    dataset,

    val_idx

)

# ==========================================
# DATALOADER
# ==========================================

train_loader = DataLoader(

    train_dataset,

    batch_size=32,

    shuffle=True,

    num_workers=0

)

val_loader = DataLoader(

    val_dataset,

    batch_size=32,

    shuffle=False,

    num_workers=0

)

# ==========================================
# PRINT DATA DISTRIBUTION
# ==========================================

train_labels = [

    dataset.targets[i]

    for i in train_idx

]

val_labels = [

    dataset.targets[i]

    for i in val_idx

]

train_count = Counter(train_labels)

val_count = Counter(val_labels)

print("\n========== TRAIN DATA ==========")

for i, cls in enumerate(dataset.classes):

    print(

        f"{cls:<10}: {train_count[i]}"

    )

print(f"Total : {len(train_dataset)}")

print("\n========== VALIDATION DATA ==========")

for i, cls in enumerate(dataset.classes):

    print(

        f"{cls:<10}: {val_count[i]}"

    )

print(f"Total : {len(val_dataset)}")

print("\nDataset preparation finished.\n")

# ==========================================
# LOAD PRETRAINED RESNET18
# ==========================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# ==========================================
# PARTIAL FREEZING
# ==========================================

# Freeze seluruh layer
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# Ganti classifier
num_classes = len(dataset.classes)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

# FC harus ikut belajar
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()

# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=1e-4,

    weight_decay=1e-4

)

# ==========================================
# LEARNING RATE SCHEDULER
# ==========================================

scheduler = optim.lr_scheduler.StepLR(

    optimizer,

    step_size=5,

    gamma=0.1

)

# ==========================================
# TRAINING
# ==========================================

epochs = 10

best_accuracy = 0

for epoch in range(epochs):

    ####################################
    # TRAIN
    ####################################

    model.train()

    average_loss = 0

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

    train_accuracy = 100 * correct / total
    average_loss = average_loss / len(train_loader)

    ####################################
    # VALIDATION
    ####################################

    model.eval()

    val_loss = 0

    val_correct = 0

    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss / len(val_loader)

    ####################################
    # PRINT
    ####################################

    print(

        f"Epoch [{epoch+1}/{epochs}]"

    )

    print(

        f"Train Loss       : {average_loss:.4f}"

    )

    print(

        f"Train Accuracy   : {train_accuracy:.2f}%"

    )

    print(

        f"Validation Loss  : {val_loss:.4f}"

    )

    print(

        f"Validation Acc   : {val_accuracy:.2f}%"

    )

    print("-" * 40)

    ####################################
    # SAVE BEST MODEL
    ####################################

if val_accuracy > best_accuracy:

    best_accuracy = val_accuracy

    torch.save(
        model.state_dict(),
        "best_resnet18_model.pth"
    )

    print("Best model saved!")
    
    scheduler.step()