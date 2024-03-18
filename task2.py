import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.001

train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                  train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=True)

test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                 train=False,
                                                 transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Why = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(device)

        for i in range(x.size(1)):
            h = torch.tanh(self.Wxh(x[:, i, :]) + self.Whh(h))

        output = self.Why(h)
        return output


model = CustomRNN(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch):
    total_step = len(train_loader)
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('训练轮次: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                epoch, i * len(images), len(train_loader.dataset),
                       100. * i / len(train_loader), loss.item()))
def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('\n测试集: 准确率: {}/{} ({:.2f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    test()
    for epoch in range(30):
        train(epoch)
        test()