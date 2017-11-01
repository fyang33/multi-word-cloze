import torch, math
import torch.nn as nn
from torch.autograd import Variable


# python train.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48  --model_file test.model
# python train_bi.py --data_file data/hw4_data.bin --optimizer Adam -lr 1e-2 --batch_size 48  --model_file test.model
class Linear(nn.Module):
    def __init__(self, in_size, out_size):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.zeros(in_size, out_size))
        self.b = nn.Parameter(torch.zeros(out_size))
        # reset uniformly
        stdv = 1. / math.sqrt(self.w.size(0))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return torch.matmul(input, self.w) + self.b


class Embedding(nn.Module):
    def __init__(self, in_size, out_size):
        super(Embedding, self).__init__()
        self.C = nn.Parameter(torch.zeros(in_size, out_size))
        # reset uniformly
        stdv = 1. / math.sqrt(self.C.size(0))
        self.C.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return self.C[input.data, :]


class LogSoftmax(nn.Module):
    def forward(self, input):
        X = torch.exp(input - torch.max(input, dim=2, keepdim=True)[0])
        return torch.log(X) - torch.log(torch.sum(X, dim=2, keepdim=True))


class RNN(nn.Module):
    def __init__(self, in_size, out_size, bi_dir=False):
        super(RNN, self).__init__()
        self.hidden_size = out_size
        self.i2h = Linear(in_size, out_size)
        self.h2h = Linear(out_size, out_size)
        self.activation = torch.sigmoid
        self.bi_dir = bi_dir
        if bi_dir:
            self.i2h_back = Linear(in_size, out_size)
            self.h2h_back = Linear(out_size, out_size)

    def forward(self, input):
        T = input.data.shape[0]
        h = [Variable(torch.zeros(input.data.shape[1], self.hidden_size))]
        # for each time step
        for t in xrange(T):
            h.append(self.activation(self.i2h(input[t]) + self.h2h(h[-1])))

        if self.bi_dir:
            h_back = [Variable(torch.zeros(input.data.shape[1], self.hidden_size))]
            for t in xrange(T - 1, -1, -1):
                h_back.append(self.activation(self.i2h_back(input[t]) + self.h2h_back(h_back[-1])))
            # reverse so h_back[-1] is the init state
            h_back = h_back[::-1]
            h = torch.stack(h[:-1], 0)  # shift ignore the last token (end of sent)
            h_back = torch.stack(h_back[1:], 0)  # ignore the first token(start of sent)
            return torch.cat((h, h_back), dim=2)
        else:
            return torch.stack(h[1:], 0)


class RNNLM(nn.Module):
    def __init__(self, vocab_size):
        super(RNNLM, self).__init__()
        self.input_size = vocab_size
        self.embedding_size = 32
        self.hidden_size = 16

        self.layers = nn.ModuleList()
        self.layers.append(Embedding(self.input_size, self.embedding_size))
        self.layers.append(RNN(self.embedding_size, self.hidden_size))
        self.layers.append(Linear(self.hidden_size, self.input_size))
        self.layers.append(LogSoftmax())

    def forward(self, input_batch):
        """
        input shape seq_len, batch_size
        ouput shape sequence_length, batch_size, vocab_size
        """
        output = input_batch
        for layer in self.layers:
            output = layer(output)
        return output


class BiRNNLM(nn.Module):
    def __init__(self, vocab_size):
        self.train = True
        super(BiRNNLM, self).__init__()
        self.input_size = vocab_size
        self.embedding_size = 32
        self.hidden_size = 8

        self.layers = nn.ModuleList()
        self.layers.append(Embedding(self.input_size, self.embedding_size))
        self.layers.append(RNN(self.embedding_size, self.hidden_size, bi_dir=True))
        self.layers.append(Linear(2 * self.hidden_size, self.input_size))
        self.layers.append(LogSoftmax())

    def training(self):
        """
        Turn the module into training mode
        """
        self.train = True

    def evaluate(self):
        """
        Turn the module into evaluate mode
        """
        self.train = False
    def forward(self, input_batch):
        """
        input shape seq_len, batch_size
        ouput shape sequence_length, batch_size, vocab_size
        """
        output = input_batch
        for layer in self.layers:
            output = layer(output)
        return output

class Dropout(nn.Module):
    """
    A dropout layer
    """
    def __init__(self, p = 0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout prob has to be between 0 and 1, but got {}".format(p))
        self.p = p #self.p is the drop rate, if self.p is 0, then it's a identity layer

    def forward(self, input):
        if self.train:
            p = self.p
            self.mask = .bernoulli_(1 - p).div_(1 - p)

            output = self.mask * input
        else:
            output = input
        # Need to take care of training mode and evaluation mode
        return output

    def backward(self, input, gradOutput):
        if self.train:
            self._gradInput = self.mask * _gradOutput
        else:
            self._gradInput = _gradOutput
        return self._gradInput