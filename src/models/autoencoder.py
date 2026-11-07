import torch
import torch.nn as nn

class BetterAutoencoder(nn.Module):
    """
    U-Net style autoencoder with skip connections for better detail preservation.
    
    FIXED VERSION: More robust to different input sizes by using adaptive pooling
    and ensuring dimensions match during concatenation.
    
    Architecture:
    - Encoder: 4 downsampling blocks (128 -> 64 -> 32 -> 16 -> 8)
    - Bottleneck: 256 channels at 8x8 resolution
    - Decoder: 4 upsampling blocks with skip connections from encoder
    - Skip connections preserve fine details that would be lost in compression
    """
    def __init__(self):
        super().__init__()
        
        # ============ ENCODER (with skip connections saved) ============
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),      # /2
            nn.ReLU(inplace=True)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),     # /4
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),    # /8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),   # /16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ============ BOTTLENECK ============
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # ============ DECODER (with skip connections) ============
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass with skip connections and size matching.
        
        Args:
            x: Input tensor [B, 3, H, W] - H and W should be divisible by 16
        
        Returns:
            Reconstructed tensor [B, 3, H, W]
        """
        # ============ ENCODER (save features for skip connections) ============
        e1 = self.enc1(x)      # [B, 32, H/2, W/2]
        e2 = self.enc2(e1)     # [B, 64, H/4, W/4]
        e3 = self.enc3(e2)     # [B, 128, H/8, W/8]
        e4 = self.enc4(e3)     # [B, 256, H/16, W/16]
        
        # ============ BOTTLENECK ============
        b = self.bottleneck(e4)  # [B, 256, H/16, W/16]
        
        # ============ DECODER (with skip connections and size matching) ============
        d4 = self.dec4(b)  # [B, 128, H/8, W/8]
        
        # Match sizes before concatenation (in case of rounding issues)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # [B, 64, H/4, W/4]
        
        if d3.shape[2:] != e2.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B, 32, H/2, W/2]
        
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # [B, 16, H, W]
        
        # Final output - match original input size
        out = self.final(d1)  # [B, 3, H, W]
        
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


# Lightweight version with same robustness
class LightweightAutoencoder(nn.Module):
    """
    Lighter version with fewer channels and size matching.
    """
    def __init__(self):
        super().__init__()
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(8, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        b = self.bottleneck(e4)
        
        d4 = self.dec4(b)
        if d4.shape[2:] != e3.shape[2:]:
            d4 = nn.functional.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        if d3.shape[2:] != e2.shape[2:]:
            d3 = nn.functional.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        if d2.shape[2:] != e1.shape[2:]:
            d2 = nn.functional.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        out = self.final(d1)
        if out.shape[2:] != x.shape[2:]:
            out = nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out