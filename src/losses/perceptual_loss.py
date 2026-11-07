import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    Combined loss function for sharper autoencoder reconstruction.
    
    Components:
    1. MSE Loss: Pixel-wise accuracy
    2. Edge Loss: Preserves sharp transitions and prevents blurring
    3. Color Loss: Maintains color consistency
    
    The edge loss measures gradients (rate of change between pixels) to ensure
    the network preserves sharp edges instead of regressing to blurry averages.
    """
    def __init__(self, mse_weight=1.0, edge_weight=0.2, color_weight=0.1):
        """
        Args:
            mse_weight: Weight for pixel-wise MSE loss
            edge_weight: Weight for edge preservation loss (0.1-0.3 recommended)
            color_weight: Weight for perceptual color loss
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        """
        Calculate combined perceptual loss.
        
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
        
        Returns:
            Combined loss scalar
        """
        # 1. MSE Loss - pixel-wise accuracy
        mse_loss = self.mse(pred, target)
        
        # 2. Edge Loss - gradient preservation for sharpness
        # Horizontal gradients (rate of change left-to-right)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        
        # Vertical gradients (rate of change top-to-bottom)
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # L1 loss on gradients (less sensitive to outliers than L2)
        edge_loss = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        
        # 3. Perceptual Color Loss - maintain color distribution
        # This helps preserve overall color balance
        pred_mean = pred.mean(dim=[2, 3])  # [B, 3] - average color per channel
        target_mean = target.mean(dim=[2, 3])
        color_loss = self.mse(pred_mean, target_mean)
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.edge_weight * edge_loss +
            self.color_weight * color_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'edge': edge_loss.item(),
            'color': color_loss.item(),
            'total': total_loss.item()
        }


class VGGPerceptualLoss(nn.Module):
    """
    Advanced perceptual loss using pretrained VGG16 features.
    
    This is the "professional" version - uses a pretrained network to compare
    high-level features instead of raw pixels. Results in more perceptually
    similar reconstructions.
    
    Note: Requires torchvision and downloads VGG16 weights (~500MB) on first use.
    Training will be slower but quality is better.
    """
    def __init__(self, 
                 mse_weight=1.0, 
                 vgg_weight=0.1, 
                 edge_weight=0.1,
                 requires_grad=False):
        """
        Args:
            mse_weight: Weight for pixel-wise MSE loss
            vgg_weight: Weight for VGG perceptual loss (0.05-0.2 recommended)
            edge_weight: Weight for edge loss
            requires_grad: If True, VGG features are trainable (usually False)
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.vgg_weight = vgg_weight
        self.edge_weight = edge_weight
        
        try:
            import torchvision.models as models
            
            # Load pretrained VGG16 and extract feature layers
            vgg = models.vgg16(pretrained=True).features
            
            # Use layers up to relu3_3 (layer 16) for mid-level features
            # These capture edges, textures, and patterns
            self.vgg_features = nn.Sequential(*list(vgg.children())[:16])
            
            # Freeze VGG parameters (we don't want to train VGG)
            if not requires_grad:
                for param in self.vgg_features.parameters():
                    param.requires_grad = False
            
            self.vgg_features.eval()
            self.vgg_available = True
            
        except ImportError:
            print("Warning: torchvision not available, VGG loss disabled")
            self.vgg_available = False
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize_for_vgg(self, x):
        """Normalize image for VGG (expects ImageNet normalization)"""
        return (x - self.vgg_mean) / self.vgg_std
    
    def forward(self, pred, target):
        """
        Calculate VGG perceptual loss.
        
        Args:
            pred: Predicted image [B, 3, H, W] in [0, 1]
            target: Target image [B, 3, H, W] in [0, 1]
        
        Returns:
            Combined loss scalar and loss dictionary
        """
        # 1. MSE Loss
        mse_loss = self.mse(pred, target)
        
        # 2. Edge Loss
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        edge_loss = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        
        # 3. VGG Perceptual Loss (if available)
        vgg_loss = torch.tensor(0.0, device=pred.device)
        if self.vgg_available and self.vgg_weight > 0:
            # Normalize inputs for VGG
            pred_norm = self.normalize_for_vgg(pred)
            target_norm = self.normalize_for_vgg(target)
            
            # Extract features
            with torch.set_grad_enabled(self.training):
                pred_features = self.vgg_features(pred_norm)
                target_features = self.vgg_features(target_norm)
            
            # Compare features
            vgg_loss = self.mse(pred_features, target_features)
        
        # Combine losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.edge_weight * edge_loss +
            self.vgg_weight * vgg_loss
        )
        
        return total_loss, {
            'mse': mse_loss.item(),
            'edge': edge_loss.item(),
            'vgg': vgg_loss.item() if self.vgg_available else 0.0,
            'total': total_loss.item()
        }


class EdgeLoss(nn.Module):
    """
    Standalone edge loss for quick testing.
    Focuses purely on gradient preservation.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        # Compute gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        # Sum of L1 losses on gradients
        loss = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)
        return loss


# Quick reference for choosing loss function:
"""
LOSS FUNCTION SELECTION GUIDE:

1. PerceptualLoss (RECOMMENDED for most cases):
   - Fast training
   - Good edge preservation
   - Balanced quality/speed
   Use: criterion = PerceptualLoss(mse_weight=1.0, edge_weight=0.2, color_weight=0.1)

2. VGGPerceptualLoss (BEST QUALITY, slower):
   - Professional-grade results
   - Best for complex images
   - Requires torchvision
   - ~30% slower training
   Use: criterion = VGGPerceptualLoss(mse_weight=1.0, vgg_weight=0.1, edge_weight=0.1)

3. EdgeLoss (TESTING):
   - Pure edge preservation
   - Good for debugging
   Use: criterion = EdgeLoss()

4. MSELoss (BASELINE):
   - Original (blurry results)
   - Fastest training
   Use: criterion = nn.MSELoss()
"""