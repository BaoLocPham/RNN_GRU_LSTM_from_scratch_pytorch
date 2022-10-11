import torch
from torch import nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh", device="cpu"):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.device = device

        self.x2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(input_size, hidden_size, bias=bias)

    def forward(self, input, hx=None):
        """
        Inputs:
            - input: of shape (batch_size, input_size)
            - hx: of shape (batch_size, hidden_size)
        Ouputs:
            - hy: of shape (batch_size, hidden_size)
        """

        if hx is None:
            if self.device == "cuda":
                hx = torch.autograd.Variable(
                      input.new_zeros(input.size(0), self.hidden_size)).cuda()
            else:
                torch.autograd.Variable(
                      input.new_zeros(input.size(0), self.hidden_size))

        hy = (self.x2h(input) + self.h2h(hx))

        if self.nonlinearity == "tanh":
            hy = torch.tanh(hy)
        else:
            hy = torch.relu(hy)

        return hy


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device="cpu"):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=self.bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=self.bias)

    def forward(self, input, hx=None):
        """
        Inputs:
            input: of shape (batch_size, input_size)
            hx: of shape (batch_size, hidden_size)

        Outputs:
            hy: of shape(batch_size, hidden_size)
        """
        if hx is None:
            if self.device == "cuda":
                hx = torch.autograd.Variable(
                    input.new_zeros(input.size(0), self.hidden_size)).cuda()
            else:
                hx = torch.autograd.Variable(
                    input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        x_reset, x_update, x_new = x_t.chunk(3, 1)
        h_reset, h_update, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate

        return hy


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, device="cpu"):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device

        self.xh = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hh = nn.Linear(hidden_size, hidden_size *4, bias=bias)

    def forward(self, input, hx=None):
        """
        Inputs:
            input: of shape (batch_size, input_size)
            hx: of shape (batch_size, hidden_size)

        Outputs:
            hy: of shape (batch_size, hidden_size)
            cy: of shape (batch_size, hidden_size)
        """
        if hx is None:
            hx = torch.autograd.Variable(
                    input.new_zeros(input.size(0), self.hidden_size))
            hx = (hx, hx)

        hx, cx = hx

        gates = self.xh(input) + self.hh(hx)

        # get gates (i_t, f_t, g_t, o_t)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        cy = cx * f_t + i_t * g_t

        hy = o_t * torch.tanh(cy)

        return (hy, cy)
