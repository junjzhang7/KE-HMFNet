import math

import torch
from torch import nn


class SAB(nn.Module):

    def __init__(self, hidden_size, nheads=2, norm=True) -> None:
        super().__init__()
        self.num_attention_heads = nheads
        self.attention_head_size = int(hidden_size / nheads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_linear = nn.Linear(hidden_size, hidden_size)
        self.key_linear = nn.Linear(hidden_size, hidden_size)
        self.value_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = norm

        self.ln0 = nn.LayerNorm(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.out_linear = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query, context=None, mask=None):
        if context == None:
            context = query

        query_layer = self.transpose_for_scores(query)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(context)
        value_layer = self.transpose_for_scores(context)

        attn_weights = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attn_weights = attn_weights / math.sqrt(self.attention_head_size)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        context_layer = torch.matmul(attn_weights, value_layer)

        output_layer = query_layer + context_layer
        output_layer = output_layer.permute(0, 2, 1, 3).contiguous()

        new_output_shape = output_layer.size()[:-2] + (self.all_head_size, )
        output_layer = output_layer.view(*new_output_shape)

        if self.norm:
            output_layer = self.ln0(output_layer)

        output_layer = output_layer + nn.functional.relu(
            self.out_linear(output_layer))

        if self.norm:
            output_layer = self.ln1(output_layer)

        return output_layer, attn_weights
