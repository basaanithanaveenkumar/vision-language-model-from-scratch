import torch
import torch.nn as nn

from positional_embeddings import SinusoidalPositionalEmbedding
from vision_encoder import VisionEncoder
from image_proj import ImageProjector
from lm_head import LMHead


class BasicVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, vision_model_name='vit_small_patch16_dinov3_qkvb.lvd1689m'):
        super().__init__()
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeds = SinusoidalPositionalEmbedding(embed_dim)
        self.vision_encoder = VisionEncoder(model_name=vision_model_name, freeze=True, output_dim=embed_dim)
        self.image_projector = ImageProjector(vision_dim=embed_dim, llm_dim=embed_dim)
        self.lm_head = LMHead(hidden_size=embed_dim, vocab_size=vocab_size)
        # write the decoder only transformer
        # Decoder-only transformer (masked self-attention)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.0,
            activation='gelu',
            batch_first=False  # Important: batch dimension first
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=24
        )


    def forward(self, images, input_ids):
        img_features = self.vision_encoder(images)
        img_proj = self.image_projector(img_features)
        
        text_embeds = self.token_embeds(input_ids)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)

        transformer_inp= combined_embeds+self.positional_embeds(combined_embeds)

        transformer_out = self.transformer(
            tgt=transformer_inp,
            memory=transformer_inp,)
        final_out=self.lm_head(transformer_out)
        return final_out