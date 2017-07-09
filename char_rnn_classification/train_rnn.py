#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification

from torch.autograd import Variable

from char_rnn import data
from char_rnn import model



if __name__ == "__main__":
    train_data = data.read_data()
    print("Read %d classes" % len(train_data))

    rnn = model.RNN(input_size=model.INPUT_SIZE, hidden_size=100, output_size=len(train_data))
    tensor = Variable(model.name_to_tensor("A"))
    hidden = rnn.new_hidden()

    output, hidden = rnn.forward(tensor[0], hidden)
    print(output.data.topk(1))
    pass
