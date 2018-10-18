import torch

from util.Tensorizer import Tensorizer

CHAR_SET = "abcdefghijklmnopqrstuvwxyz" + ".,;'"

# 3x1x3
tensorizer = Tensorizer("abc")
correctTensor = torch.FloatTensor([[[1, 0, 0]], [[1, 0, 0]], [[0, 1, 0]]])
testTensor = tensorizer.sentenceToTensor("aab")
isEqual = torch.equal(correctTensor, testTensor)
assert isEqual == True
