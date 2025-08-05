import torch
from torch import nn

from ._utils import spherical_5d_encoding


class PinnGM(nn.Module):
    def __init__(self, hidden_features=16, hidden_layers=8):
        super().__init__()
        self.encoder_1 = nn.Linear(5, hidden_features)
        self.encoder_2 = nn.Linear(5, hidden_features)
        self.input_layer = nn.Linear(5, hidden_features)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_features, hidden_features) for _ in range(hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_features, 1)
        self.activation = nn.GELU()
        # Trainable bounding conditions
        self.r_ref = nn.Parameter(
            torch.tensor(3.0, dtype=torch.float32), requires_grad=True
        )
        self.k = nn.Parameter(
            torch.tensor(0.1, dtype=torch.float32), requires_grad=True
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights of all layers with Xavier normal
        nn.init.xavier_normal_(self.encoder_1.weight)
        nn.init.xavier_normal_(self.encoder_2.weight)
        nn.init.xavier_normal_(self.input_layer.weight)
        self.encoder_1.bias.data.fill_(0.0)
        self.encoder_2.bias.data.fill_(0.0)
        self.input_layer.bias.data.fill_(0.0)  # Initialize biases to zero

        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0.0)  # Initialize biases to zero

        self.output_layer.weight.data.fill_(0.0)
        self.output_layer.bias.data.fill_(0.0)  # Initialize biases to zero

    def forward(self, input):
        """
        input shape: Nx3
        """
        # input = input.clone().detach().requires_grad_(True)  # Allows to take derivative w.r.t. input
        r = torch.norm(input, dim=1, keepdim=True)
        input = spherical_5d_encoding(input)
        x = self.input_layer(input)
        x = self.activation(x)
        u = self.encoder_1(input)
        u = self.activation(u)
        v = self.encoder_2(input)
        v = self.activation(v)
        ux = u * x
        vx = (1 - x) * v
        x = ux + vx

        for _, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)
            x = self.activation(x)
            ux = u * x
            vx = (1 - x) * v
            x = ux + vx

        u_nn = self.output_layer(x)

        # scale proxy potential into true potential, see section 3.3 of PINN paper
        r_e = input[:, 1].view(-1, 1)
        u_nn = u_nn * r_e
        # enforce boundary conditions, see section 3.4 of PINN paper
        w_bc = (1 + torch.tanh(self.k * (r - self.r_ref))) / 2
        w_nn = 1 - w_bc
        u_bc = 1 / r  # u_bc = mu / r.    mu = M * G = 1 assuming G = 1 and M = 1
        output = w_bc * u_bc + w_nn * u_nn
        return output


if __name__ == "__main__":
    net = PinnGM(32, 8)
    print([torch.Tensor(list(p.size())) for p in net.parameters()])
    print([torch.prod(torch.Tensor(list(p.size()))) for p in net.parameters()])
    print(sum([torch.prod(torch.Tensor(list(p.size()))) for p in net.parameters()]))
