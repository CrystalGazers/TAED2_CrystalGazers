import math
import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Class to implement self attention mechanism in Neural Network."""
    def __init__(self, embed_dim, bias=True, num_heads=1):
        super().__init__()
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, bias=bias, batch_first=True)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset k, v, q and weights."""
        # Empirically observed the convergence to be much better with the scaled initialization
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    def forward(self, x):
        """Forward step in the self-attention module."""
        # x shape is (B, W, E)
        q = self.q_proj(x)
        # q shape is (B, W, E)
        k = self.k_proj(x)
        # k shape is (B, W, E)
        v = self.v_proj(x)
        # k shape is (B, W, E)
        y, _ = self.multihead_attn(q, k, v)
        # y shape is (B, W, E)
        y = self.out_proj(y)
        # y shape is (B, W, E)
        return y


class TransformerLayer(nn.Module):
    """Class that implements a transformer layer."""
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1, num_heads=1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads=num_heads)
        # Implementation of Feed-forward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        """Forward step in the transformer layer."""
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Predictor(nn.Module):
    """Class to predict a word given a context window around it."""
    def __init__(self, num_embeddings, embedding_dim, context_words=6, num_heads=1):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        self.att1 = TransformerLayer(embedding_dim, num_heads=num_heads)
        self.att2 = TransformerLayer(embedding_dim, num_heads=num_heads)
        self.position_embedding = nn.Parameter(torch.Tensor(context_words, embedding_dim))
        nn.init.xavier_uniform_(self.position_embedding)

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, inp):
        """Forward step for the prediction."""
        # input shape is (B, W)
        e = self.emb(inp)
        # e shape is (B, W, E)
        u = e + self.position_embedding
        # u shape is (B, W, E)
        v1 = self.att1(u)
        # v1 shape is (B, W, E)
        v2 = self.att2(v1)
        # v2 shape is (B, W, E)
        x = v2.sum(dim=1)
        # x shape is (B, E)
        y = self.lin(x)
        # y shape is (B, V)
        return y
        