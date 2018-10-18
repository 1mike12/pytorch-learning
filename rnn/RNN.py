from __future__ import unicode_literals, print_function, division

import torch.nn as nn
import torch
import torch.optim as optim
import unicodedata


class RNN(nn.Module):

    def __init__(self, device, learningRate, momentum, inputSize, hiddenSize, outputSize, batchSize=None):
        super(RNN, self).__init__()

        self.criterion = nn.NLLLoss().to(device)

        self.hiddenSize = hiddenSize
        self.inputToHidden = nn.Linear(inputSize + hiddenSize, hiddenSize)
        self.inputToOutput = nn.Linear(inputSize + hiddenSize, outputSize)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.SGD(self.parameters(), lr=learningRate, momentum=momentum)
        self.to(device)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.inputToHidden(combined)
        output = self.inputToOutput(combined)
        output = self.softmax(output)
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

    def predict(self, inputs):
        hidden = self.getHidden()
        for i in range(inputs.size()[0]):
            output, hidden = self(inputs[i], hidden)
        return output

    def getHidden(self):
        return torch.zeros(1, self.hiddenSize)
