import torch
import torch.nn as nn
import open_clip


class OpenCLIPEncoder(nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k',
                 freeze=True, output_dim=512):
        super().__init__()
        
        # Load OpenCLIP model (vision + text)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained
        )
        
        # Extract just the vision encoder
        self.vision_encoder = self.model.visual
        
        # Freeze vision encoder if needed
        if freeze:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.vision_encoder.parameters():
                param.requires_grad = True
        
        # Get feature dimension
        self.feature_dim = self.vision_encoder.output_dim
        
        # Optional projection layer
        self.projection = None
        if output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
            self.feature_dim = output_dim
    
    def forward(self, x):
        """
        Input: [batch, 3, H, W] (images should be preprocessed with self.preprocess)
        Output: [batch, feature_dim]
        """
        # Extract visual features
        features = self.vision_encoder(x)
        
        # Apply projection if exists
        if self.projection:
            features = self.projection(features)
        
        return features


# Usage example
if __name__ == "__main__":
    # Create encoder
    encoder = OpenCLIPEncoder(
        model_name='ViT-B-32',
        pretrained='laion2b_s34b_b79k',
        freeze=True,
        output_dim=512
    ).cuda()
    
    # Test with random image
    batch = torch.randn(4, 3, 224, 224).cuda()
    features = encoder(batch)
    print(features.shape)
    print(f"Input shape: {batch.shape}")
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {encoder.feature_dim}")
