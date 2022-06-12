import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_layer = nn.Linear(args.input_size, args.hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(args.hidden_dim, args.hidden_dim)
            for _ in range(args.n_hidden_layers)
        ])
        self.output_layer = nn.Linear(args.hidden_dim, 20)

    def forward(self, x):
        x = self.input_layer(x).relu()
        for layer in self.hidden_layers:
            x = x + layer(x).relu()

        x = self.output_layer(x)
        return x