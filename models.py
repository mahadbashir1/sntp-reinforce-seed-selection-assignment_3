import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    # same small network applied to every participant separately
    def __init__(self):
        super().__init__()
        self.config = {"architecture": "mlp", "input_dim": 5, "hidden_dim": 64}
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, features):
        # features: (N, 5) -> logits: (N,)
        return self.net(features).squeeze(-1)


class TransformerPolicy(nn.Module):
    # each participant is one token, no positional or ID embeddings
    def __init__(self):
        super().__init__()
        self.config = {"architecture": "transformer", "input_dim": 5, "d_model": 64, "nhead": 4,
                       "num_layers": 1, "dim_feedforward": 128, "activation": "relu",
                       "dropout": 0.0, "norm_first": False}
        self.input_layer = nn.Linear(5, 64)
        self.encoder = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                  dropout=0.0, activation="relu",
                                                  batch_first=True, norm_first=False)
        self.output_layer = nn.Linear(64, 1)

    def forward(self, features):
        tokens = self.input_layer(features).unsqueeze(0)  # (1, N, 64)
        tokens = self.encoder(tokens)  # full self-attention over all N tokens, no mask
        return self.output_layer(tokens).squeeze(0).squeeze(-1)  # (N,)


def masked_distribution(logits, mask):
    # selected participants get -inf, so their probability is exactly 0
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return torch.distributions.Categorical(logits=masked_logits)


def greedy_action(policy, features, mask):
    # evaluation: highest probability valid action (argmax returns smaller ID on ties)
    with torch.no_grad():
        dist = masked_distribution(policy(torch.from_numpy(features)), torch.from_numpy(mask))
        return int(torch.argmax(dist.probs))
