import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class SelfAttn(nn.Module):
    def __init__(self, emb_dim,return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim
        
        # Linear projections for Query , Key, Vale, bias needs to be false
        self.query_proj = nn.Linear(em_dim, em_dim, bias=False)
        self.key_proj = nn.Linear(em_dim, em_dim, bias=False)
        self.val_proj = nn.Linear(em_dim, em_dim, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
    def forward(self, inputs,causal_mask=None):
        # inputs: [B, Seq_len, D]
        Query = self.query_proj(inputs)  # [B, Seq_len, D]
        Key = self.key_proj(inputs)    # [B, seq_len, D]
        Val = self.val_proj(inputs)    # [B, seq_len, D]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, Seq_len, seq_len]
        
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Seq_len, Seq_len]

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, Val)  # [B, T, D]
        
        if return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output  



class HeadAttn(nn.Module):
    def __init__(self, emb_dim=256,head_size=16,drop_fact=0.0,causal_mask=False,return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim
        
        # Linear projections for Query , Key, Vale, bias needs to be false
        self.query_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.key_proj = nn.Linear(emb_dim, head_size, bias=False)
        self.val_proj = nn.Linear(emb_dim, head_size, bias=False)

        self.scale = 1.0 / math.sqrt(emb_dim)
        self.causal_mask =causal_mask
        self.dropout = nn.Dropout(drop_fact)
        self.return_attn_weights=return_attn_weights
    def forward(self, inputs):
        # inputs: [B, Seq_len, D]
        B,seq_len,D=inputs.shape
        Query = self.query_proj(inputs)  # [B, Seq_len, D]
        Key = self.key_proj(inputs)    # [B, seq_len, D]
        Val = self.val_proj(inputs)    # [B, seq_len, D]

        # Compute attention scores
        scores = torch.matmul(Query, Key.transpose(-2, -1)) * self.scale  # [B, Seq_len, seq_len]
        
        if self.causal_mask:
            causal_tril = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=inputs.device))
            scores = scores.masked_fill(causal_tril == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, Seq_len, Seq_len]

        # drop in attention weights
        
        attn_weights = self.dropout(attn_weights)
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, Val)  # [B, T, D]
        
        if self.return_attn_weights:
            return attn_output, attn_weights
        else:
            return attn_output 

class MultiHeadAttn(nn.Module):
    def __init__(self, emb_dim=256,num_heads=8,drop_fact=0.0,causal_mask=False,return_attn_weights=False):
        super().__init__()
        self.em_dim = emb_dim
        self.num_heads=num_heads
        # TODO write a assert or raise error if emb_dim is not divisible by num_heads

        self.head_size=emb_dim//num_heads
        self.heads = nn.ModuleList([HeadAttn(emb_dim,head_size=self.head_size,drop_fact=drop_fact,causal_mask=causal_mask,return_attn_weights=return_attn_weights) for _ in range(num_heads)])
        
        self.proj = nn.Linear(emb_dim,emb_dim)
        self.dropout = nn.Dropout(drop_fact)
    def forward(self, inputs):
        # inputs: [B, Seq_len, D]
        head_outputs = [head(inputs) for head in self.heads]  # list of [B, Seq_len, head_size]
        
        # Concatenate head outputs
        concat_output = torch.cat(head_outputs, dim=-1)  # [B, Seq_len, D]
        
        # Final linear projection
        output = self.proj(concat_output)  # [B, Seq_len, D]
        output = self.dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim=256, num_heads=8, mlp_dim=512, drop_fact=0.0,causal_mask=False):
        super().__init__()
        self.attn = MultiHeadAttn(emb_dim=emb_dim,num_heads=num_heads,drop_fact=drop_fact,causal_mask=causal_mask)
        # after input
        self.norm1 = nn.LayerNorm(emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(drop_fact)
        )
        # before ffn
        self.norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, inputs,causal_mask=False):
        # Self-attention block
        attn_output = self.attn(self.norm1(inputs))
        x = inputs + attn_output  # Residual connection
        
        # Feed-forward block
        mlp_output = self.mlp(self.norm2(x))
        output = x + mlp_output  # Residual connection
        
        return output

class DecoderTransformer(nn.Module):
    def __init__(self, num_layers=16, emb_dim=1024, num_heads=32, mlp_dim=512, drop_fact=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, drop_fact=drop_fact,causal_mask=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x