import torch
from torch import nn


class GRUOnlyBaseline(nn.Module):
    def __init__(
        self,
        num_features: int = 6,
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.dropout_value = dropout
        self.rnn = nn.GRU(
            input_size=num_features,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.util_head = nn.Linear(rnn_hidden, 1)
        self.load_head = nn.Linear(rnn_hidden, 1)
        self.cong_head = nn.Linear(rnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, E, F]
        batch_size, seq_len, num_edges, num_features = x.shape
        rnn_input = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_edges,
            seq_len,
            num_features,
        )
        _, h_n = self.rnn(rnn_input)
        h = h_n[-1].reshape(batch_size, num_edges, self.rnn_hidden)
        h = self.head_dropout(h)
        util_pred = self.util_head(h).squeeze(-1)
        load_pred = self.load_head(h).squeeze(-1)
        cong_logit = self.cong_head(h).squeeze(-1)
        return util_pred, load_pred, cong_logit


class LSTMOnlyBaseline(nn.Module):
    def __init__(
        self,
        num_features: int = 6,
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.dropout_value = dropout
        self.rnn = nn.LSTM(
            input_size=num_features,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.util_head = nn.Linear(rnn_hidden, 1)
        self.load_head = nn.Linear(rnn_hidden, 1)
        self.cong_head = nn.Linear(rnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, E, F]
        batch_size, seq_len, num_edges, num_features = x.shape
        rnn_input = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_edges,
            seq_len,
            num_features,
        )
        _, (h_n, _) = self.rnn(rnn_input)
        h = h_n[-1].reshape(batch_size, num_edges, self.rnn_hidden)
        h = self.head_dropout(h)
        util_pred = self.util_head(h).squeeze(-1)
        load_pred = self.load_head(h).squeeze(-1)
        cong_logit = self.cong_head(h).squeeze(-1)
        return util_pred, load_pred, cong_logit


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
