import torch
from torch import nn


class GraphConvLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, E, F], adj: [E, E]
        agg = torch.einsum("ij,bjf->bif", adj, x)
        out = self.linear(agg)
        out = self.activation(out)
        return self.dropout(out)


class UGGRU(nn.Module):
    def __init__(
        self,
        num_features: int = 6,
        gcn_hidden: int = 32,
        gru_hidden: int = 64,
        gru_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.gcn_hidden = gcn_hidden
        self.gru_hidden = gru_hidden
        self.gru_layers = gru_layers
        self.dropout_value = dropout

        self.graph_conv = GraphConvLayer(num_features, gcn_hidden, dropout)
        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head_dropout = nn.Dropout(dropout)
        self.util_head = nn.Linear(gru_hidden, 1)
        self.load_head = nn.Linear(gru_hidden, 1)
        self.cong_head = nn.Linear(gru_hidden, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, L, E, F]
        batch_size, seq_len, num_edges, _ = x.shape
        g_steps = []
        for step in range(seq_len):
            g_steps.append(self.graph_conv(x[:, step, :, :], adj))

        g_seq = torch.stack(g_steps, dim=1)  # [B, L, E, gcn_hidden]
        gru_input = g_seq.permute(0, 2, 1, 3).reshape(
            batch_size * num_edges,
            seq_len,
            self.gcn_hidden,
        )
        _, h_n = self.gru(gru_input)
        h = h_n[-1].reshape(batch_size, num_edges, self.gru_hidden)
        h = self.head_dropout(h)

        util_pred = self.util_head(h).squeeze(-1)
        load_pred = self.load_head(h).squeeze(-1)
        cong_logit = self.cong_head(h).squeeze(-1)
        return util_pred, load_pred, cong_logit


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
