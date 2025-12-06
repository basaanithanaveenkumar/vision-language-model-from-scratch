import torch
import torch.nn as nn
import timm

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_small_patch16_dinov3_qkvb.lvd1689m', 
                 freeze=True, output_dim=None):
        super().__init__()
        
        # Load pretrained model
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

# Usage example
if __name__ == "__main__":
    # Create encoder
    encoder = VisionEncoder(
        model_name='vit_small_patch16_dinov3_qkvb.lvd1689m',
        freeze=True,
        output_dim=512
    ).cuda()
    
    # Test with random image
    batch = torch.randn(4, 3, 224, 224).cuda()
    features = encoder(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {encoder.feature_dim}")