import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
import torch.nn.functional as F

torch.manual_seed(123)
train_batch_size = 64
test_batch_size = 1000
img_size = 28

def get_dataloader(train=True):
    assert isinstance(train, bool)
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        transform=transforms.ToTensor(),
        download=True
    )
    batch_size = train_batch_size if train else test_batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256,10)

    def forward(self, x):
        x = x.view(-1, 28*28*1)
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)
        return out

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    train_dataloader = get_dataloader(train=True)
    for idx, (x, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('训练轮次: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                epoch, idx * len(x), len(train_dataloader.dataset),
                100. * idx / len(train_dataloader), loss.item()))

def test():
    model.eval()
    test_dataloader = get_dataloader(train=False)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for x, y in test_dataloader:
            output = model(x)
            test_loss += F.cross_entropy(output, y, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    print('\n测试集: 平均损失: {:.4f}, 准确率: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))



if __name__ == '__main__':
    test()
    for epoch in range(30):
        train(epoch)
        test()
