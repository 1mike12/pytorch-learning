import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

RELU = functional.relu
FILTERS = 6  # default 6


class CNN(nn.Module):

    def __init__(self, device, learningRate):
        super(CNN, self).__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)

        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=Tru
        self.conv1 = nn.Conv2d(3, FILTERS, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(FILTERS, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=0.9)
        self.to(device)

    def forward(self, x):
        x = self.pool(RELU(self.conv1(x)))
        x = self.pool(RELU(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = RELU(self.fc1(x))
        x = RELU(self.fc2(x))
        x = self.fc3(x)
        return x

    def runAndGetLoss(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss
