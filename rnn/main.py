from __future__ import unicode_literals, print_function, division

import math
from io import open
import glob
import os
import torch
import torchvision
import time
import random

from Util import timeSince
from rnn.RNN import RNN
from util.Tensorizer import Tensorizer

CHAR_SET = "abcdefghijklmnopqrstuvwxyz" + ".,';"

RUN_ON_GPU = False

all_categories = []
category_lines = {}
current_loss = 0
all_losses = []

LEARNING_RATE = .002
MOMENTUM = 0
HIDDEN_SIZE = 128

ITERATIONS = 100000
PRINT_EVERY = 5000
PLOT_EVERY = 1000

## setup data
if RUN_ON_GPU:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not torch.cuda.is_available(): raise Exception("trying to run on gpu but no cuda available")
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

tensorizer = Tensorizer(CHAR_SET)


def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [tensorizer.unicodeToAscii(line) for line in lines]


for fileName in glob.glob("./names/*.txt"):
    category = os.path.splitext(os.path.basename(fileName))[0]
    all_categories.append(category)
    lines = readLines(fileName)
    category_lines[category] = lines

start = time.time()

# =============================


model = RNN(DEVICE, LEARNING_RATE, MOMENTUM, len(CHAR_SET), HIDDEN_SIZE, len(all_categories))


def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = tensorizer.sentenceToTensor(line)
    return category, line, category_tensor, line_tensor


def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


for i in range(1, ITERATIONS + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = model.runAndGetLoss(line_tensor, category_tensor)
    current_loss += loss

    # Print iter number, loss, name and guess
    if i % PRINT_EVERY == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else f'✗ {category}'
        percentage = i / ITERATIONS * 100
        lossString = "{:.4f}".format(loss)

        print(f'{i} {int(percentage)}% ({timeSince(start)}) {lossString} {line} | {guess} {correct} ')
        # print(f'%d %d%% (%s) %.4f %s / %s %s' % (i, i / ITERATIONS * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if i % PLOT_EVERY == 0:
        all_losses.append(current_loss / PLOT_EVERY)
        current_loss = 0


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = model.predict(tensorizer.sentenceToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])


predict("O'Flannigan")
predict('Jackson')
predict('Hirohito')
