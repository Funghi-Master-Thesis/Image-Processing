import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm  

# 1. Define data preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ViT input
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

trainingset = "Data\RIS1_0_TL_20_preset\Trainingset"
validationset = "Data\RIS1_0_TL_20_preset\Validationset"

# 2. Load dataset (CIFAR-10 example, replace with your dataset)
# train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
# test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

train_dataset = datasets.CIFAR10(root=trainingset, train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root=validationset, train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Load the Vision Transformer (ViT) model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=10)  # For CIFAR-10 (10 classes)

# 4. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 5. Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 5  # Set the number of training epochs
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# 6. Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
