import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Datasets and Loaders
def get_loader(train, transform):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=train, num_workers=2)

trainloader = get_loader(True, transform_train)
testloader = get_loader(False, transform_test)

# Model Setup

from fasterkan import FasterKANvolver
net = FasterKANvolver([256, 256, 10]).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001)  # Adjusted learning rate
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # Slight adjustment

EPOCHS = 100

for epoch in range(EPOCHS):
    net.train()
    total_loss = 0
    for inputs, labels in tqdm(trainloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
    
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(trainloader)
    print(f'Epoch {epoch}, Loss: {avg_loss:.6f}', )
    scheduler.step()

    # Evaluation
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * (correct / total)
    print(f'Accuracy on 10,000 test images: {accuracy:.2f} %')