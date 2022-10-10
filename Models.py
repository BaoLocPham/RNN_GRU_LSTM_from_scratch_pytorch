import torch
from torch import nn
from Cells import RNNCell, GRUCell, LSTMCell


class SimpleRNN(nn.Module):
    def __init__(
            self, vocab_size, embedding_size,
            input_size, hidden_size, num_layers,
            bias, output_size, activation="tanh", device="cpu"):
        super(SimpleRNN, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.activation = activation
        self.rnn_cell_list = nn.ModuleList()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.rnn_cell_list.append(RNNCell(
            self.input_size,
            self.hidden_size,
            self.bias,
            activation
        ))

        for layer in range(1, self.num_layers):
            self.rnn_cell_list.append(RNNCell(
                self.hidden_size,
                self.hidden_size,
                self.bias,
                activation
            ))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hx=None):
        """
        Input:
            - shape (batch_size, sequence_length, input_size)

        Output:
            - shape (batch_size, output_size)
        """
        # print("Input shape", input.shape)
        if hx is None:
            if self.device == "cuda":
                h0 = torch.autograd.Variable(
                                            torch.zeros(self.num_layers,
                                            input.size(0),
                                            self.hidden_size)
                    ).cuda()
            else:
                h0 = torch.autograd.Variable(
                                            torch.zeros(self.num_layers,
                                            input.size(0),
                                            self.hidden_size)
                    )
        else:
            h0 = hx

        outs = []
        hidden = []

        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])

        input = self.embedding(input)

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](
                                                        input[:, t, :],
                                                        hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](
                                                        hidden[layer-1],
                                                        hidden[layer])

                hidden[layer] = hidden_l
            outs.append(hidden_l)
        # take only last time step. Modify seq to seq
        out = outs[-1].squeeze()

        out = self.fc(out)
        out = self.sigmoid(out)
        return out


class SimpleGRU(nn.Module):
    def __init__(
                self, vocab_size, embedding_size,
                input_size, hidden_size, num_layers, bias, output_size,
                device="cpu"):
        super(SimpleGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.device = device

        self.gru_cell_list = nn.ModuleList()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.gru_cell_list.append(GRUCell(
            self.input_size,
            self.hidden_size,
            self.bias,
        ))

        for layer in range(1, self.num_layers):
            self.gru_cell_list.append(GRUCell(
                self.hidden_size,
                self.hidden_size,
                self.bias,
            ))

        self.fc = nn.Linear(self.hidden_size, self.output_size)


        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hx=None):
        """
            Input: of shape (batch_size, sequence_length, input_size)

            Output: of shape (batch_size, output_size)
        """

        if hx is None:
            if self.device == "cuda":
                h0 = torch.autograd.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size)).cuda()
            else:
                h0 = torch.autograd.Variable(torch.zeros(self.num_layers, input.size(0), self.hidden_size))
        else:
            h0 = hx

        outs = []
        hidden = list()

        for layer in range(self.num_layers):
            hidden.append(h0[layer, :, :])


        input = self.embedding(input)

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.gru_cell_list[layer](input[:, t, :], hidden[layer])
                else:
                    hidden_1 = self.gru_cell_list[layer](hidden[layer-1], hidden[layer])

            hidden[layer] = hidden_l

            outs.append(hidden_l)

        # take only one last step.
        out = outs[-1].squeeze()

        out = self.fc(out)

        out = self.sigmoid(out)

        return out


class SimpleLSTM(nn.Module):
    def __init__(
            self, vocab_size, embedding_size,
            input_size, hidden_size, num_layers, bias, output_size, device="cpu"):
        super(SimpleLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bias = bias
        self.device = device

        self.lstm_cell_list = nn.ModuleList()

        # embedding layer is useful to map input into vector representation
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        self.lstm_cell_list.append(LSTMCell(
            self.input_size,
            self.hidden_size,
            self.bias,
            # activation
        ))

        for layer in range(1, self.num_layers):
            self.lstm_cell_list.append(LSTMCell(
                self.hidden_size,
                self.hidden_size,
                self.bias,
                # activation
            ))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Sigmoid layer cz we will have binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hx=None):
        """
            Input: of shape (batch_size, sequence_length, input_size)

            Output: of shape (batch_size, output_size)
        """

        if hx is None:
            if self.device == "cuda":
                h0 = torch.autograd.Variable(
                    torch.zeros(self.num_layers,
                    input.size(0), self.hidden_size)).cuda()
            else:
                h0 = torch.autograd.Variable(
                    torch.zeros(self.num_layers,
                    input.size(0), self.hidden_size))
        else:
            h0 = hx

        outs = []
        hidden = list()

        for layer in range(self.num_layers):
            hidden.append(
                (h0[layer, :, :], h0[layer, :, :])
            )


        input = self.embedding(input)

        for t in range(input.size(1)):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.lstm_cell_list[layer](
                                                        input[:, t, :],
                                                        (hidden[layer][0],
                                                        hidden[layer][1]))
                else:
                    hidden_1 = self.lstm_cell_list[layer](
                                                        hidden[layer-1][0],
                                                        (hidden[layer][0],
                                                        hidden[layer][1]))

            hidden[layer] = hidden_l

            outs.append(hidden_l[0])

        # take only one last step.
        out = outs[-1].squeeze()

        out = self.fc(out)

        out = self.sigmoid(out)

        return out
