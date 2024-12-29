import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import CIFAR10Net
from utils.dataset import CIFAR10Dataset
from utils.transforms import Transforms
from torchsummary import summary

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {train_loss/(batch_idx+1):.6f} '
                  f'Accuracy: {100.*correct/total:.2f}%')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create datasets and dataloaders
    train_dataset = CIFAR10Dataset(train=True)
    transform = Transforms(means=train_dataset.mean, stds=train_dataset.std)
    
    train_dataset.transform = transform
    test_dataset = CIFAR10Dataset(train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model, optimizer and loss function
    model = CIFAR10Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Print model summary
    summary(model, (3, 32, 32))
    
    # Training loop
    best_acc = 0
    for epoch in range(1, 51):
        train(model, device, train_loader, optimizer, criterion, epoch)
        accuracy = test(model, device, test_loader, criterion)
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == '__main__':
    main() 