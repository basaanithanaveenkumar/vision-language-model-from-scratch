import torch
import torch.nn as nn
import timm


class SigLIPEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_siglip_224.webli', 
                 freeze=True, output_dim=None):
        super().__init__()
        
        # Load pretrained SigLIP model
        # Common variants: 
        # - 'vit_base_patch16_siglip_224.webli' (SigLIP-B)
        # - 'vit_large_patch16_siglip_384.webli' (SigLIP-L)
        # - 'vit_so400m_patch14_siglip_384.webli' (SigLIP-SO400M)
        self.model = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Freeze weights if needed
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        self.feature_dim = self.model.embed_dim
        
        # Optional projection layer
        self.projection = None
        if output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
            self.feature_dim = output_dim
    
    def forward(self, x):
        """
        Input: [batch, 3, H, W]
        Output: [batch, feature_dim]
        """
        # Extract features
        features = self.model.forward_features(x)
        
        # Average over patches (for ViT) -> [batch, feature_dim]
        features = features.mean(dim=1)
        
        # Apply projection if exists
        if self.projection:
            features = self.projection(features)
        
        return features

class CLIPEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_clip_224.openai_ft_in1k', 
                 freeze=True, output_dim=None):
        super().__init__()
        
        # Load pretrained CLIP model (fine-tuned variants available in timm)
        # Common variants:
        # - 'vit_base_patch16_clip_224.openai_ft_in1k' (CLIP-B/16, fine-tuned)
        # - 'vit_base_patch16_clip_384.openai_ft_in1k' (CLIP-B/16, 384px)
        # - 'vit_large_patch14_clip_336.openai' (CLIP-L/14@336px)
        self.model = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Freeze weights if needed
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Get feature dimension
        self.feature_dim = self.model.embed_dim
        
        # Optional projection layer
        self.projection = None
        if output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
            self.feature_dim = output_dim
    
    def forward(self, x):
        """
        Input: [batch, 3, H, W]
        Output: [batch, feature_dim]
        """
        # Extract features
        features = self.model.forward_features(x)
        
        # Average over patches (for ViT) -> [batch, feature_dim]
        features = features.mean(dim=1)
        
        # Apply projection if exists
        if self.projection:
            features = self.projection(features)
        
        return features



# Usage examples
if __name__ == "__main__":
    # Test SigLIP encoder
    # siglip_encoder = SigLIPEncoder(
    #     model_name='vit_base_patch16_siglip_224.webli',
    #     freeze=True,
    #     output_dim=512
    # ).cuda()
    
    # Test CLIP encoder
    clip_encoder = CLIPEncoder(
        model_name='vit_base_patch16_clip_224.openai_ft_in1k',
        freeze=True,
        output_dim=512
    ).cuda()
    
    # Test with random image
    batch = torch.randn(4, 3, 224, 224).cuda()
    
    siglip_features = siglip_encoder(batch)
    clip_features = clip_encoder(batch)
    
    print(f"SigLIP feature shape: {siglip_features.shape}")
    print(f"CLIP feature shape: {clip_features.shape}")
