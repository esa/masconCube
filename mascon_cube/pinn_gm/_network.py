import torch
from torch import nn

from ._utils import spherical_5d_encoding


class PinnGM(nn.Module):
    def __init__(self, hidden_features, hidden_layers):
        super().__init__()
        self.input_layer = nn.Linear(5, hidden_features)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_features, 1)
        self.activation = nn.GELU()

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights of all layers with Xavier uniform
        nn.init.xavier_uniform_(self.input_layer.weight)
        self.input_layer.bias.data.fill_(0.0)  # Initialize biases to zero

        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.0)  # Initialize biases to zero

        self.output_layer.weight.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)  # Initialize biases to zero

    def forward(self, input):
        """
        input shape: Nx3
        """
        # input = input.clone().detach().requires_grad_(True)  # Allows to take derivative w.r.t. input
        input = spherical_5d_encoding(input)
        y = self.input_layer(input)
        y = self.activation(y)

        x = y  # Initialize x to y before the loop
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i > 0:
                x = x + y
            x = hidden_layer(x)
            x = self.activation(x)

        output = self.output_layer(x)
        return output


if __name__ == "__main__":
    net = PinnGM(32, 8)
    print([torch.Tensor(list(p.size())) for p in net.parameters()])
    print([torch.prod(torch.Tensor(list(p.size()))) for p in net.parameters()])
    print(sum([torch.prod(torch.Tensor(list(p.size()))) for p in net.parameters()]))
