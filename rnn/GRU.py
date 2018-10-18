from __future__ import unicode_literals, print_function, division

import torch.nn as nn
import torch
import torch.optim as optim
import unicodedata

from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, characters, device, learningRate, momentum, inputSize, hiddenSize, outputSize, batchSize):
        super(Net, self).__init__()
        self.rnn1 = nn.GRU(input_size=characters,
                           hidden_size=hiddenSize,
                           num_layers=1)
        self.fc = nn.Linear(hiddenSize, outputSize)

    def forward(self, input, hidden):
        input, hidden = self.rnn1(input, hidden)
        output = self.fc(input)
        return input, hidden

    def getInitialHidden(self, batchSize):
        weight = next(self.parameters()).data
        return Variable(weight.new(128, batchSize, 1).zero_())

class RNN(nn.Module):

    def __init__(self, characters, device, learningRate, momentum, inputSize, hiddenSize, outputSize):
        super(RNN, self).__init__()
        self.charSet = {}

        for i, char in enumerate(characters):
            self.charSet[char] = i
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
        hidden = torch.zeros(1, self.hiddenSize)
        for i in range(inputs.size()[0]):
            output, hidden = self(inputs[i], hidden)

        loss = self.criterion(output, categoryTensor)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()

    def predict(self, inputs):
        hidden = torch.zeros(1, self.hiddenSize)
        for i in range(inputs.size()[0]):
            output, hidden = self(inputs[i], hidden)
        return output

    @staticmethod
    def unicodeToAscii(charSet, input):
        input = input.lower()
        return ''.join(
            c for c in unicodedata.normalize('NFD', input)
            if unicodedata.category(c) != 'Mn'
            and c in charSet
        )

    # throws KeyError if not in charSet
    def indexForChar(self, char):
        return self.charSet[char]

    def sentenceToTensor(self, sentence):
        """
        :param sentence: any unicode string
        :return: converted to tensor
        """
        sentence = RNN.unicodeToAscii(self.charSet, sentence)
        tensor = torch.zeros(len(sentence), 1, len(self.charSet))
        for i, letter in enumerate(sentence):
            tensor[i][0][self.indexForChar(letter)] = 1
        return tensor
