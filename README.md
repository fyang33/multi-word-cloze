# Part 1
For RNNLM, we use `nn.ModuleList()` to represent a sequential NN.
```
    def forward(self, input_batch):
        """
        input shape seq_len, batch_size
        ouput shape sequence_length, batch_size, vocab_size
        """
        output = input_batch
        for layer in self.layers:
            output = layer(output)
        return output
```
And we have class `Linear` `LogSoftmax` `Embedding` `RNN` which all are the subclass of `nn.Module` and overwrite their `forward()` method respectively.
# Part 2
For bi-directional RNNLM, we just need to modify the RNN layer.
```
class RNNLM(nn.Module):
    def __init__(self, vocab_size, bi_directional=False):
        super(RNNLM, self).__init__()
        self.input_size = vocab_size
        self.embedding_size = 32
        num_dir = 2 if bi_directional else 1
        self.hidden_size = 16

        self.layers = nn.ModuleList()
````

# Part 3
From this part, we implemented the dropout layer
