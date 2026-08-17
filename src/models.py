"""UniFi time-aware DNN (mTAN encoder + GRU + MLP classifier)."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombineEmbed(nn.Module):
    """Fuse CSI values with the time encoding into one content-aware key
    (Eq. 14-15): project the observed values into the time-embedding space and
    add the time encoding."""

    def __init__(self, num_channels=3, d_model=128):
        super().__init__()
        self.value_embed = nn.Linear(num_channels, d_model)
        self.time_embed = nn.Linear(1, d_model)  # kept for checkpoint parity

    def forward(self, x, time_enc):
        # x: (batch, seq_len, num_channels), time_enc: (batch, seq_len, d_model)
        value_emb = self.value_embed(x)
        return value_emb + time_enc


class multiTimeAttention(nn.Module):
    
    def __init__(self, input_dim, nhidden=16, 
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time), 
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])
        self.norm = nn.LayerNorm(nhidden)
        # self.dropout = nn.Dropout(0.1)

        
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -1e9)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn
    
    
    def forward(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        # print(value[0,0,:])
        # print(query.size(), key.size(), value.size(), mask.size())
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        # print(query.size(), key.size(), value.size(), mask.size())
        x, _ = self.attention(query, key, value, mask, dropout)
        # print(x.size())
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        # print(x.size())
        # return self.linears[-1](x)
        x = self.linears[-1](x)
        x = self.norm(x)
        # x = self.dropout(x)
        return x


class enc_mtan_classif_csi(nn.Module):
    """UniFi DNN: time-aware attention over irregular CSI -> GRU -> MLP head.

    Two design choices from the paper are baked in:

    * the missingness mask is **not** fed as a value feature (CSI missingness is
      not motion related); it is still used to mask attention scores.
    * attention keys are content-aware, built from CSI + time rather than time
      alone (Eq. 14-15).
    """

    def __init__(self, input_dim, query, nhidden, n_labels,
                 embed_time=16, num_heads=1, learn_emb=True, freq=100.,
                 device='cuda'):
        super(enc_mtan_classif_csi, self).__init__()
        assert embed_time % num_heads == 0
        self.freq = freq
        self.embed_time = embed_time
        self.learn_emb = learn_emb
        self.dim = input_dim
        self.device = device
        self.nhidden = nhidden
        self.query = query
        self.value_embed = nn.Linear(input_dim, embed_time)
        self.att = multiTimeAttention(input_dim, nhidden, embed_time, num_heads)
        self.combine_embed = CombineEmbed(input_dim, embed_time)

        self.classifier = nn.Sequential(
            nn.Linear(nhidden, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, n_labels))
        self.enc = nn.GRU(nhidden, nhidden)

        if learn_emb:
            self.periodic = nn.Linear(1, embed_time-1)
            self.linear = nn.Linear(1, 1)
    
    
    def learn_time_embedding(self, tt):
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)
       
        
    def time_embedding(self, pos, d_model):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model)
        position = pos.unsqueeze(2)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(np.log(self.freq) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe
    
       
    def forward(self, x, time_steps):
        time_steps = time_steps.cpu()
        # x is [values | mask]: keep the mask for attention masking only
        mask = x[:, :, self.dim:]
        x = x[:, :, :self.dim]
        if self.learn_emb:
            key_time_embed = self.learn_time_embedding(time_steps).to(self.device)
            query_time_embed = self.learn_time_embedding(self.query.unsqueeze(0)).to(self.device)
        else:
            key_time_embed = self.time_embedding(time_steps, self.embed_time).to(self.device)
            query_time_embed = self.time_embedding(self.query.unsqueeze(0), self.embed_time).to(self.device)
        combined_embeddings = self.combine_embed(x, key_time_embed)
        out = self.att(query_time_embed, combined_embeddings, x, mask)

        out = out.permute(1, 0, 2)
        _, out = self.enc(out)
        return self.classifier(out.squeeze(0))
