import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

RELU = functional.relu
FILTERS = 32  # default 6
FILTER_DIMENSIONS = 3
PADDING = 1


class CNN(nn.Module):

    def __init__(self, device, learningRate):
        super(CNN, self).__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)

        # (in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=Tru
        self.conv1 = nn.Conv2d(3, FILTERS, FILTER_DIMENSIONS, padding=PADDING)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(FILTERS, FILTERS, FILTER_DIMENSIONS, padding=PADDING)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=0.9)
        self.to(device)

    def forward(self, input):
        out = self.pool(RELU(self.conv1(input)))
        out = self.pool(RELU(self.conv2(out)))
        out = self.pool(RELU(self.conv2(out)))
        out = out.view(-1, out.shape[1] * out.shape[2] * out.shape[3])
        out = RELU(self.fc1(out))
        out = RELU(self.fc2(out))
        out = self.fc3(out)
        return out

    def runAndGetLoss(self, inputs, labels):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss
