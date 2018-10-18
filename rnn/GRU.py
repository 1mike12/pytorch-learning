from __future__ import unicode_literals, print_function, division

import torch.nn as nn
import torch
import torch.optim as optim

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, device, learningRate, momentum, inputSize, hiddenSize, outputSize, batchSize):
        super(Net, self).__init__()

        self.criterion = nn.NLLLoss().to(device)
        self.batchSize = batchSize

        self.gru1 = nn.GRU(input_size=inputSize,
                           hidden_size=hiddenSize,
                           num_layers=1)
        self.fc = nn.Linear(hiddenSize, outputSize)

        self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=momentum)
        self.to(device)

    def forward(self, input, hidden):
        input, hidden = self.gru1(input, hidden)
        output = self.fc(input)
        return output, hidden

    def runAndGetLoss(self, inputs, categoryTensor):
        self.optimizer.zero_grad()
        hidden = self.getHidden()
        for i in range(inputs.size()[0]):
            output, hidden = self(inputs[i], hidden)

        loss = self.criterion(output, categoryTensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def getInitialHidden(self, batchSize):
        weight = next(self.parameters()).data
        return Variable(weight.new(128, batchSize, 1).zero_())

    def getHidden(self):
        return torch.zeros(1, self.hiddenSize)
