import torch.nn as nn


class bp_model_2017(nn.Module):
    def __init__(self, in_size, out_size):
        super(bp_model_2017, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.02),
            nn.Linear(64, 32),
            nn.Dropout(0.02),
        )
        self.linear = nn.Linear(32, out_size)
        # self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        x = self.linear(out)
        return x
