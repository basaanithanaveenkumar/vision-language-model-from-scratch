import torch
import torch.nn as nn

from models.positional_embeddings import SinusoidalPositionalEmbedding
from models.vision_encoder import VisionEncoder
from models.open_clipencoder import OpenCLIPEncoder
from models.image_proj import ImageProjector
from models.lm_head import LMHead


class BasicVLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=1024, vision_model_name='vit_base_patch16_224'):
        super().__init__()
        self.token_embeds = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeds = SinusoidalPositionalEmbedding(embed_dim)
        #self.vision_encoder = VisionEncoder(model_name=vision_model_name, freeze=True, output_dim=embed_dim)
        self.vision_encoder = OpenCLIPEncoder(model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', freeze=False, output_dim=embed_dim)
        self.image_projector = ImageProjector(vision_dim=embed_dim, llm_dim=embed_dim)
        self.lm_head = LMHead(hidden_size=embed_dim, vocab_size=vocab_size)
        # write the decoder only transformer
        # Decoder-only transformer (masked self-attention)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=64,
            dim_feedforward=1024,
            dropout=0.0,
            activation='gelu',
            batch_first=True  # Important: batch dimension first
        )
        self.transformer = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=12
        )


    def forward(self, images, input_ids,attention_mask):
        B = input_ids.size(0)
        seq_len = input_ids.size(1)
        num_img_tokens = 1  # Assuming single image token
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
        combined_embeds = combined_embeds + self.positional_embeds(combined_embeds)

        if attention_mask is None:
            # If no mask provided, assume all tokens are valid
            attention_mask = torch.ones(B, seq_len, device=input_ids.device, dtype=torch.bool)
        else:
            # Convert to bool (input might be float or int)
            attention_mask = attention_mask.to(dtype=torch.bool)
        
        # Create combined mask (image token is never padding)
        img_mask = torch.ones(B, num_img_tokens, device=input_ids.device, dtype=torch.bool)
        combined_mask = torch.cat([img_mask, attention_mask], dim=1)  # [B, 1+seq_len]
        
        # Create causal mask for self-attention
        num_total_tokens = num_img_tokens + seq_len
        attn_mask = torch.zeros(
            num_total_tokens, num_total_tokens,
            device=combined_embeds.device,
            dtype=torch.bool
        )
        
        # Text tokens use causal attention (can't attend to future positions)
        # Image tokens can attend to everything
        text_start = num_img_tokens
        text_seq_len = seq_len
        
        # Create causal pattern: upper triangular (True = mask out)
        text_causal = torch.triu(
            torch.ones(text_seq_len+1, text_seq_len+1, device=combined_embeds.device, dtype=torch.bool),
            diagonal=1  # diagonal=1 means mask out the diagonal and above
        )
        
        # Apply causal mask only to text-text interactions
        attn_mask = text_causal
        
        key_padding_mask = ~combined_mask  
        
        transformer_out = self.transformer(
            tgt=combined_embeds,              
            memory=combined_embeds,           
            tgt_mask=attn_mask,               
            tgt_key_padding_mask=key_padding_mask,  
        )
        final_out = self.lm_head(transformer_out)  
        
        return final_out


        # transformer_inp= combined_embeds+self.positional_embeds(combined_embeds)

        # num_tokens = num_img_tokens + input_ids.size(1)
        # # Create causal mask: allows image tokens and previous text tokens
        # # Shape: [seq_len, seq_len]
        # full_mask = torch.zeros(num_tokens, num_tokens, device=transformer_inp.device, dtype=torch.bool)
        # # Image tokens can attend to all positions, text tokens use causal attention
        # text_start = num_img_tokens
        # text_text_mask = torch.triu(torch.ones(input_ids.size(1), input_ids.size(1), device=transformer_inp.device, dtype=torch.bool), diagonal=1)
        # full_mask[text_start:, text_start:] = text_text_mask
        
        # transformer_out = self.transformer(
        #     tgt=transformer_inp,
        #     memory=transformer_inp,
        #     tgt_mask=full_mask,
        #     memory_mask=None
        #     )
        # # TODO for the masking need to check the tgt_is_causel and memory_is_causal
        # final_out=self.lm_head(transformer_out)
        # return final_out


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

    import sys
    
    def debug_vlm_gradients():
        """Comprehensive debugging for VLM gradient flow"""
        print("\n" + "="*80)
        print("VLM GRADIENT FLOW DEBUGGING")
        print("="*80 + "\n")
        
        model = BasicVLM(vocab_size=30522, embed_dim=512).cuda()
        batch_size = 2
        num_img_tokens = 1
        num_txt_tokens = 10
        images = torch.randn(batch_size, 3, 224, 224).cuda()
        input_ids = torch.randint(0, 30522, (batch_size, num_txt_tokens)).cuda()
        attention_mask = torch.ones(batch_size, num_txt_tokens).cuda()
        
        print("1. INPUT SHAPES")
        print("-" * 80)
        print(f"  Images:        {images.shape}")
        print(f"  Input IDs:     {input_ids.shape}")
        print(f"  Attention mask: {attention_mask.shape}\n")
        
        # Forward pass with debugging
        print("2. FORWARD PASS DEBUGGING")
        print("-" * 80)
        
        try:
            # Step 1: Vision encoder
            print("  [1/7] Vision encoder...")
            img_features = model.vision_encoder(images)
            print(f"        Image features shape: {img_features.shape}")
            print(f"        Image features dtype: {img_features.dtype}")
            print(f"        Image features requires_grad: {img_features.requires_grad}")
            
            if img_features.dim() == 2:
                img_features = img_features.unsqueeze(1)
            print(f"        After unsqueeze: {img_features.shape}\n")
            
            # Step 2: Image projection
            print("  [2/7] Image projector...")
            img_proj = model.image_projector(img_features)
            print(f"        Projected shape: {img_proj.shape}")
            print(f"        Projected dtype: {img_proj.dtype}")
            print(f"        Projected requires_grad: {img_proj.requires_grad}\n")
            
            # Step 3: Text embeddings
            print("  [3/7] Token embeddings...")
            text_embeds = model.token_embeds(input_ids)
            print(f"        Text embeddings shape: {text_embeds.shape}")
            print(f"        Text embeddings dtype: {text_embeds.dtype}")
            print(f"        Text embeddings requires_grad: {text_embeds.requires_grad}\n")
            
            # Step 4: Concatenation
            print("  [4/7] Concatenation...")
            combined_embeds = torch.cat([img_proj, text_embeds], dim=1)
            print(f"        Combined shape: {combined_embeds.shape}")
            print(f"        Combined requires_grad: {combined_embeds.requires_grad}\n")
            
            # Step 5: Positional embeddings
            print("  [5/7] Positional embeddings...")
            pos_embeds = model.positional_embeds(combined_embeds)
            print(f"        Positional embeds shape: {pos_embeds.shape}")
            print(f"        Positional embeds requires_grad: {pos_embeds.requires_grad}")
            combined_embeds = combined_embeds + pos_embeds
            print(f"        After adding pos embeds: {combined_embeds.shape}")
            print(f"        Requires_grad: {combined_embeds.requires_grad}\n")
            
            # Step 6: Create attention mask
            print("  [6/7] Creating attention mask...")
            text_seq_len = input_ids.size(1)
            text_causal = torch.triu(
                torch.ones(text_seq_len+1, text_seq_len+1, device=combined_embeds.device, dtype=torch.bool),
                diagonal=1
            )
            attn_mask = text_causal
            print(f"        Attention mask shape: {attn_mask.shape}")
            print(f"        Attention mask dtype: {attn_mask.dtype}\n")
            
            # Step 7: Transformer
            print("  [7/7] Transformer...")
            transformer_out = model.transformer(
                tgt=combined_embeds,
                memory=combined_embeds,
                tgt_mask=attn_mask
            )
            print(f"        Transformer output shape: {transformer_out.shape}")
            print(f"        Transformer output requires_grad: {transformer_out.requires_grad}\n")
            
            # Step 8: LM Head
            print("  [8/8] LM Head...")
            final_out = model.lm_head(transformer_out)
            print(f"        Final output shape: {final_out.shape}")
            print(f"        Final output requires_grad: {final_out.requires_grad}\n")
            
        except Exception as e:
            print(f"  ✗ FORWARD PASS ERROR: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Backward pass
        print("3. BACKWARD PASS DEBUGGING")
        print("-" * 80)
        
        try:
            # Create a dummy loss
            loss = final_out.mean()
            print(f"  Loss value: {loss.item():.6f}")
            print(f"  Loss requires_grad: {loss.requires_grad}\n")
            
            print("  Computing backward pass...")
            loss.backward()
            print("  ✓ Backward pass completed\n")
            
        except Exception as e:
            print(f"  ✗ BACKWARD PASS ERROR: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Check gradients
        print("4. GRADIENT ANALYSIS")
        print("-" * 80)
        
        params_with_grad = []
        params_without_grad = []
        params_frozen = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                params_frozen.append(name)
            elif param.grad is None:
                params_without_grad.append((name, param.shape, param.numel()))
            else:
                grad_norm = param.grad.data.norm(2).item()
                params_with_grad.append((name, param.shape, grad_norm))
        
        print(f"\n  ✓ Parameters WITH gradients: {len(params_with_grad)}")
        if params_with_grad:
            print(f"    {'Layer':<50} {'Shape':<20} {'Grad Norm':<15}")
            print(f"    {'-'*85}")
            for name, shape, grad_norm in params_with_grad[:10]:
                print(f"    {name:<50} {str(shape):<20} {grad_norm:<15.6e}")
            if len(params_with_grad) > 10:
                print(f"    ... and {len(params_with_grad) - 10} more")
        
        print(f"\n  ✗ Parameters WITHOUT gradients (requires_grad=True): {len(params_without_grad)}")
        if params_without_grad:
            print(f"    {'Layer':<50} {'Shape':<20} {'Num Params':<15}")
            print(f"    {'-'*85}")
            for name, shape, num_params in params_without_grad:
                print(f"    {name:<50} {str(shape):<20} {num_params:<15,}")
        
        print(f"\n  ⊗ Frozen parameters (requires_grad=False): {len(params_frozen)}")
        if params_frozen:
            print(f"    {', '.join(params_frozen[:5])}")
            if len(params_frozen) > 5:
                print(f"    ... and {len(params_frozen) - 5} more")
        
        # Check individual components
        print("\n5. COMPONENT-WISE GRADIENT CHECK")
        print("-" * 80)
        
        components = [
            ("Token Embeddings", model.token_embeds),
            ("Positional Embeddings", model.positional_embeds),
            ("Vision Encoder", model.vision_encoder),
            ("Image Projector", model.image_projector),
            ("Transformer", model.transformer),
            ("LM Head", model.lm_head)
        ]
        
        for comp_name, component in components:
            has_grad = False
            total_params = 0
            grad_params = 0
            
            for param in component.parameters():
                total_params += param.numel()
                if param.requires_grad and param.grad is not None:
                    grad_params += param.numel()
                    has_grad = True
            
            status = "✓" if has_grad else "✗"
            print(f"  {status} {comp_name:<30} Grad params: {grad_params:>10,} / {total_params:>10,}")
        
        # Summary
        print("\n6. SUMMARY")
        print("-" * 80)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        params_with_grad_count = sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and p.grad is not None)
        
        print(f"  Total parameters:          {total_params:>15,}")
        print(f"  Trainable parameters:      {trainable_params:>15,}")
        print(f"  Parameters with gradients: {params_with_grad_count:>15,}")
        print(f"  Parameters without grads:  {trainable_params - params_with_grad_count:>15,}")
        
        if len(params_without_grad) > 0:
            print(f"\n  ⚠️  WARNING: {len(params_without_grad)} parameter groups have no gradients!")
            print("     Possible causes:")
            print("     1. Vision encoder is frozen (not participating in backward pass)")
            print("     2. Attention mask issues in transformer")
            print("     3. Gradient flow blocked by detached tensors")
            print("     4. Loss doesn't depend on certain parameters")
        else:
            print(f"\n  ✓ All trainable parameters have gradients!")
        
        print("\n" + "="*80 + "\n")
        
        return {
            'params_with_grad': len(params_with_grad),
            'params_without_grad': len(params_without_grad),
            'params_frozen': len(params_frozen),
            'total_params': total_params
        }
    
    
    debug_vlm_gradients()