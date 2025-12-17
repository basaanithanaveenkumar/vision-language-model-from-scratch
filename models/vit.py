from torch import nn
import torch
from models.transformer import SelfAttn, HeadAttn , MultiHeadAttn,  DecoderTransformer, TransformerBlock
class PatchEmb(nn.Module):
    def __init__(self, img_size=224, p_size=16, in_chans=3, emb_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.p_size = p_size
        self.num_pat = (img_size // p_size) * (img_size // p_size)

        self.proj = nn.Conv2d(in_chans, emb_dim, kernel_size=p_size, stride=p_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x=self.proj(x)
        x=x.flatten(2).transpose(1, 2)  # [B, num_pat, emb_dim]
        return x

class VisTransformer(nn.Module):
    def __init__(self, img_size=224, p_size=16, in_chans=3, emb_dim=1024, num_layers=6, num_heads=8, mlp_dim=2048, drop_fact=0.0):
        super().__init__()
        self.patch_emb = PatchEmb(img_size=img_size, p_size=p_size, in_chans=in_chans, emb_dim=emb_dim)
        # learnable positional embeddings
        self.num_pat = (img_size // p_size) * (img_size // p_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_pat, emb_dim))
        self.dropout = nn.Dropout(drop_fact)
        self.lay_norm = nn.LayerNorm(emb_dim)
        self.transformer = nn.ModuleList([
            TransformerBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, drop_fact=drop_fact,causal_mask=False)
            for _ in range(num_layers)])      
        
    def forward(self, x):
        x = self.patch_emb(x)  # [B, num_pat, emb_dim]
        x = x + self.pos_embed  # Add positional embeddings
        x = self.dropout(x)

        for block in self.transformer:
            x = block(x)  # Apply each TransformerBlock sequentially # [B, num_pat, emb_dim]
        x = self.lay_norm(x)
        return x
        