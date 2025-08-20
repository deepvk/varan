import numpy as np
import torch
from scipy import stats
from torch import nn


class PriorDistributionConstructor:
    def __init__(self, config, batch_size, encoder_layers):
        self.prior_distribution = config.prior_distribution
        self.config = config
        self.batch_size = batch_size
        self.n_encoder_layers = encoder_layers

    def get_geometric_distribution(self):
        pmf = [
            (1 - self.config.p_geometric_pmf) ** i * self.config.p_geometric_pmf
            for i in range(self.n_encoder_layers)
        ]
        if sum(pmf) > 1:
            pmf[-1] -= sum(pmf) - 1
        elif sum(pmf) < 1:
            pmf[-1] += 1 - sum(pmf)
        return pmf

    def get_chi2_distribution(self):
        lower_bound = stats.ncx2.ppf(0.01, self.config.chi2_df, self.config.chi2_nc)
        upper_bound = stats.ncx2.ppf(0.99, self.config.chi2_df, self.config.chi2_nc)
        x, step = np.linspace(
            lower_bound, upper_bound, self.n_encoder_layers, retstep=True
        )
        p = stats.ncx2.pdf(x, self.config.chi2_df, self.config.chi2_nc)
        probas = p * step
        probas_sum = sum(probas)
        probas[0] -= probas_sum - 1
        return probas

    def __call__(self):
        if self.prior_distribution == "geometric":
            distribution = self.get_geometric_distribution()
        elif self.prior_distribution == "geometric_reversed":
            distribution = list(reversed(self.get_geometric_distribution()))
        elif self.prior_distribution == "uniform":
            distribution = [1 / self.n_encoder_layers] * self.n_encoder_layers
        elif self.prior_distribution == "chi2":
            distribution = self.get_chi2_distribution()
        elif self.prior_distribution == "chi2_reversed":
            distribution = list(reversed(self.get_chi2_distribution()))
        else:
            raise AttributeError(
                f"Unknown prior distribution: {self.prior_distribution}"
            )
        return (
            torch.tensor(distribution)
            .repeat(self.batch_size)
            .reshape(self.batch_size, self.n_encoder_layers)
        )


class AttentionDistributionPredictor(nn.Module):
    def __init__(self, hid_dim, num_heads):
        super().__init__()
        self.QKV = nn.Linear(hid_dim, hid_dim * 3, bias=False)
        self.hid_dim = hid_dim
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hid_dim, num_heads=num_heads, batch_first=True
        )
        self.logit_layer = nn.Linear(hid_dim, 1)

    def forward(self, x):
        # x.shape = [bs, n_layers, hid_dim]
        q, k, v = torch.split(self.QKV(x), self.hid_dim, dim=2)
        # q.shape = k.shape = v.shape = [bs, n_layers, hid_dim]
        attn_output, _ = self.self_attention(q, k, v)
        # logits.shape = [bs, n_layers]
        logits = self.logit_layer(attn_output).squeeze(-1)
        return logits
