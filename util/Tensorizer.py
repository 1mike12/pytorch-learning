import torch
import unicodedata


class Tensorizer():
    def __init__(self, characterSet):
        self.character_Index = {}
        self.charSetLength = len(characterSet)
        for i, char in enumerate(characterSet):
            self.character_Index[char] = i

    def unicodeToAscii(self, input):
        input = input.lower()
        return ''.join(
            c for c in unicodedata.normalize('NFD', input)
            if unicodedata.category(c) != 'Mn'
            and c in self.character_Index
        )

    def sentenceToTensor(self, sentence):
        """
        :param sentence: any unicode string
        :return: converted to tensor
        """
        sentence = self.unicodeToAscii(sentence)
        tensor = torch.zeros(len(sentence), 1, self.charSetLength)
        for i, letter in enumerate(sentence):
            tensor[i][0][self.character_Index[letter]] = 1
        return tensor
