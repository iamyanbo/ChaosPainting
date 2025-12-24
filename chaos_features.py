import torch
import torch.nn as nn
import torchvision.models as models

class ChaosActivation(nn.Module):
    """
    Chaos as an Activation Function.
    Applies N steps of chaotic iteration map.
    Configuration: Tuned for High Accuracy (Last Point, Mild Noise).
    """
    def __init__(self, channels, steps=2, init_skew=0.4, learnable=True, dropout_std=0.01):
        super().__init__()
        self.steps = steps
        self.dropout_std = dropout_std
        if learnable:
            self.skew_b = nn.Parameter(torch.ones(1, channels, 1, 1) * init_skew)
        else:
            self.register_buffer('skew_b', torch.tensor(init_skew))
            
    def forward(self, x, spatial_skew=None):
        # x: (Batch, Channels, H, W)
        current_x = x
        
        # Determine Skew Parameter 'b'
        if spatial_skew is not None:
             # Hyper-Chaos Mode: Skew is a spatial map (B, C, H, W)
             b = spatial_skew
        else:
             # Standard Mode: Skew is a learned scalar per channel (1, C, 1, 1)
             b = self.skew_b
        
        # ChaosDropout (Only applies to learned scalars or base skew)
        if self.training and self.dropout_std > 0 and spatial_skew is None:
            noise = torch.randn_like(b) * self.dropout_std
            b = b + noise
        
        # Clamp b to safe range
        b_clamped = torch.clamp(b, 0.01, 0.99)
        
        for _ in range(self.steps):
            cond = (current_x < b_clamped)
            
            # Skew Tent Map
            next_x = torch.where(
                cond,
                current_x / b_clamped,
                (1.0 - current_x) / (1.0 - b_clamped)
            )
            
            current_x = next_x
            
        # Return Last State (High Frequency Features)
        return current_x

class ChaosTextureNet(nn.Module):
    """
    Extracts Chaotic Texture Features from an image.
    Uses a ResNet backbone backbone, but replaces/augments activations with Chaos.
    Outputs a dense 16-channel feature map encoding 'Texture Physics'.
    """
    def __init__(self, backbone='resnet50', pretrained=True, hyper_mode=False):
        super().__init__()
        print(f"Initializing ChaosTextureNet with {backbone} (Hyper-Mode: {hyper_mode})...")
        self.hyper_mode = hyper_mode
        
        # Load Backbone
        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
            
        # We use early layers for texture (high res)
        # Layer 0: Conv1 -> MaxPool
        self.layer0 = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        # Layer 1: ResBlock 1 (256 ch)
        self.layer1 = base.layer1
        
        # Compression Layer
        self.compress = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.Sigmoid() # Map to [0,1] for Chaos
        )
        
        # Chaos Activation
        self.chaos = ChaosActivation(channels=16, steps=2, init_skew=0.4, learnable=True, dropout_std=0.01)
        
        # Hyper-Chaos: Spatial Skew Predictor
        if self.hyper_mode:
            # Predicts a (B, 16, H, W) skew map from the original features
            self.skew_predictor = nn.Sequential(
                nn.Conv2d(256, 16, kernel_size=1), # Independent parallel head
                nn.BatchNorm2d(16),
                nn.Sigmoid() # Skew must be in [0, 1]
            )
        
    def forward(self, x):
        # x: (Batch, 3, H, W)
        x = self.layer0(x) # -> (Batch, 64, H/4, W/4)
        feat_base = self.layer1(x) # -> (Batch, 256, H/4, W/4)
        
        # 1. Prepare Initial Condition (x0)
        x0 = self.compress(feat_base) # -> (Batch, 16, H/4, W/4)
        
        # 2. Determine Skew
        spatial_skew = None
        if self.hyper_mode:
            spatial_skew = self.skew_predictor(feat_base) # -> (Batch, 16, H/4, W/4)
            # Add small epsilon to avoid 0.0 or 1.0 (handled by clamp inside, but safe practice)
        
        # 3. Run Chaos Dynamics
        chaos_feats = self.chaos(x0, spatial_skew=spatial_skew)
        
        # Upsample back to H, W if needed? 
        # Usually Painters work better with dense maps.
        # But H/4 is standard for semantic seg (stride 4).
        # We will upsample to input size to match the 'pixel-wise' painting expectation
        # or handle it in the painter.
        # DeepLab usually outputs H, W.
        return chaos_feats
