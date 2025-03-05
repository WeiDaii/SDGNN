import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim, drop_prob=0.0, acti_fn=nn.ReLU(), bias=False):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias)
        self.dr = nn.Dropout(drop_prob)
        self.ac = acti_fn

    def forward(self, x):
        x = self.fc(x)
        if self.ac == None:
            return x
        output = self.ac(x).clone()

        return output