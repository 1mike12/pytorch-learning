from rnn.RNN import RNN

CHAR_SET = "abcdefghijklmnopqrstuvwxyz" + ".,;'"
model = RNN(CHAR_SET, None, .001)

assert (model.unicodeToAscii("François")) == "francois"

charSet = model.charSet
assert (charSet["a"]) == 0

d = {'a': 1}
x = d["b"]
z = 0