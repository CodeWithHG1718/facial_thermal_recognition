"""Small respiration regressors operating on nasal intensity traces."""

from torch import nn


class CNNRateNet(nn.Module):
    """Original temporal CNN retained as the comparison architecture."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(10, 24, 9, padding=4), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(24, 32, 9, padding=4), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(32, 32, 9, padding=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CNNGRURateNet(nn.Module):
    """Convolutions extract local patterns; a GRU integrates their sequence.

    Input: (batch, 10, 200), nine nasal traces and a missing-data mask.
    Encoder: (batch, 32, 50). GRU: 50 steps with 32 features each.
    Output: one respiration rate per window, in breaths/minute.
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(10, 24, 9, padding=4), nn.ReLU(), nn.AvgPool1d(2),
            nn.Conv1d(24, 32, 9, padding=4), nn.ReLU(), nn.AvgPool1d(2),
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        sequence = self.encoder(x).transpose(1, 2)
        _, hidden = self.rnn(sequence)
        return self.head(hidden[-1]).squeeze(-1)
