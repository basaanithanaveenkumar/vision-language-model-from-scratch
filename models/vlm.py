import torch
import torch.nn as nn

from models.positional_embeddings import SinusoidalPositionalEmbedding
from models.vision_encoder import VisionEncoder
from models.open_clipencoder import OpenCLIPEncoder
from models.image_proj import ImageProjector
from models.lm_head import LMHead


class BasicVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, vision_model_name='vit_base_patch16_224'):
        super().__init__()
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeds = SinusoidalPositionalEmbedding(embed_dim)
        #self.vision_encoder = VisionEncoder(model_name=vision_model_name, freeze=True, output_dim=embed_dim)
        self.vision_encoder = OpenCLIPEncoder(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', freeze=True, output_dim=embed_dim)
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
            batch_first=True  # Important: batch dimension first
        )
        self.transformer = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=12
        )


    def forward(self, images, input_ids):
        img_features = self.vision_encoder(images)
        if img_features.dim() == 2:
            img_features = img_features.unsqueeze(1)  # [B, 1, D]
        img_proj = self.image_projector(img_features)
        # if img_proj.dim() == 2:
        #     img_proj = img_proj.unsqueeze(1)  # [B, 1, D]
        B, num_img_tokens, D = img_proj.size()
        
        text_embeds = self.token_embeds(input_ids)
        
        # Combine image and text embeddings
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)

        transformer_inp= combined_embeds+self.positional_embeds(combined_embeds)

        num_tokens = num_img_tokens + input_ids.size(1)
        txt_mask  = torch.triu(torch.ones(num_tokens, num_tokens, device=transformer_inp.device, dtype=torch.bool), diagonal=1)
        # Image tokens are always valid (no padding)
        #img_mask = torch.ones(B, num_img_tokens, dtype=torch.bool, device=images.device)
        
        # Concatenate masks: [image_mask, text_mask]
        #full_mask = torch.cat([img_mask, txt_mask], dim=1)  # (B, total_len)
        transformer_out = self.transformer(
            tgt=transformer_inp,
            memory=transformer_inp,
            tgt_mask=~txt_mask,  # PyTorch expects True=ignore, so invert
            memory_mask=~txt_mask
            )
        # TODO for the masking need to check the tgt_is_causel and memory_is_causal
        final_out=self.lm_head(transformer_out)
        return final_out


# write the code to check the forward pass
if __name__ == "__main__":
    model = BasicVLM(vocab_size=30522, embed_dim=512).cuda()
    batch_size = 2
    num_img_tokens = 1
    num_txt_tokens = 10
    images = torch.randn(batch_size, 3, 224, 224).cuda()
    input_ids = torch.randint(0, 30522, (batch_size, num_txt_tokens)).cuda()
    attention_mask = torch.ones(batch_size, num_txt_tokens).cuda()
    
    outputs = model(images, input_ids)
    print(outputs[:,-1].shape)
    print("Output shape:", outputs.shape)  # Expected: [batch_size, num_img_tokens + num_txt_tokens, vocab_size]