from models.vit import VisTransformer
from models.transformer import DecoderTransformer
from models.lm_head import LMHead
from models.image_proj import ImageProjector
import torch
import torch.nn as nn

from models.positional_embeddings import SinusoidalPositionalEmbedding
class HaloVLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=512):
        super().__init__()
        self.vis_enc = VisTransformer(img_size=224, p_size=16, in_chans=3, emb_dim=emb_dim, num_layers=6, num_heads=16, mlp_dim=512, drop_fact=0.0)
        self.decoder_transformer = DecoderTransformer(num_layers=16, emb_dim=emb_dim, num_heads=32, mlp_dim=1024, drop_fact=0.0)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(5000, emb_dim)
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = LMHead(hidden_size=emb_dim, vocab_size=vocab_size)
        self.image_projector = ImageProjector(vision_dim=emb_dim, llm_dim=emb_dim)
    

    def forward(self,images,input_ids,attention_mask):
        B = input_ids.size(0)
        device = input_ids.device
        seq_len = input_ids.size(1)
        img_features = self.vis_enc(images)
        img_proj = self.image_projector(img_features)
        #print(img_proj.shape, "image projection shape")
        B, num_img_tokens, D = img_proj.size()
        num_total_tokens = num_img_tokens + seq_len
        text_embeds = self.token_emb(input_ids)
        combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
        pos_emb = self.pos_embed(torch.arange(combined_embeds.size(1),device=device)).unsqueeze(0).repeat(B, 1, 1)
        combined_embeds = combined_embeds + pos_emb
        transformer_out=self.decoder_transformer(combined_embeds)
        final_out=self.lm_head(transformer_out)
        return final_out

# write the code to test the forward pass of the model
import torch
import torch.nn as nn

# Test the forward pass
def test_forward_pass():
    # Model hyperparameters
    vocab_size = 32000  # typical tokenizer vocab size
    emb_dim = 512
    batch_size = 2
    seq_len = 77
    img_size = 224
    
    # Initialize model
    model = BasicVLM(vocab_size=vocab_size, emb_dim=emb_dim)
    model.eval()  # Set to evaluation mode
    
    # Create dummy inputs
    dummy_images = torch.randn(batch_size, 3, img_size, img_size)
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len)
    
    print(f"Input shapes:")
    print(f"  Images: {dummy_images.shape}")
    print(f"  Input IDs: {dummy_input_ids.shape}")
    print(f"  Attention mask: {dummy_attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():  # Disable gradient computation for testing
        output = model(dummy_images, dummy_input_ids, dummy_attention_mask)
    
    print(f"\nForward pass successful!")
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: (batch_size, num_img_tokens + seq_len, vocab_size)")
    
    # Additional checks
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[-1] == vocab_size, "Vocab size mismatch"
    
    print(f"\nOutput statistics:")
    print(f"  Min: {output.min().item():.4f}")
    print(f"  Max: {output.max().item():.4f}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")
    
    return True

# Run the test
if __name__ == "__main__":
    test_forward_pass()
