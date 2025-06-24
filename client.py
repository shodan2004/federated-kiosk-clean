# client.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import flwr as fl
from flwr.client import NumPyClient

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)    # (3, 32, 32) -> (6, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)     # (6, 28, 28) -> (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, 5)   # (6, 14, 14) -> (16, 10, 10)
                                           # -> after pool: (16, 5, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (6, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (16, 5, 5)
        x = x.view(-1, 16 * 5 * 5)             # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Flower client
class FlowerClient(NumPyClient):
    def __init__(self):
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for epoch in range(1):  # Use more epochs if needed
            for images, labels in trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                self.optimizer.step()
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += F.cross_entropy(outputs, labels, reduction='sum').item()
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
        loss /= len(testloader.dataset)
        accuracy = correct / len(testloader.dataset)
        return float(loss), len(testloader.dataset), {"accuracy": float(accuracy)}

# Run the Flower client
if __name__ == "__main__":
    client = FlowerClient()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client.to_client())
