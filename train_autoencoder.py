import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from datetime import datetime

# Import improved modules
from src.models.autoencoder import BetterAutoencoder, LightweightAutoencoder
from src.losses.perceptual_loss import PerceptualLoss, VGGPerceptualLoss

# ============ CONFIGURATION ============

# Training parameters
EPOCHS = int(os.getenv("EPOCHS", 200))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
LR = float(os.getenv("LR", 1e-3))
SAVE_PATH = os.getenv("MODEL_PATH", "models/autoencoder_model.pth")

# Model selection
USE_LIGHTWEIGHT = os.getenv("USE_LIGHTWEIGHT", "false").lower() == "true"

# Loss function selection
# Options: "perceptual", "vgg", "mse"
LOSS_TYPE = os.getenv("LOSS_TYPE", "perceptual")

# Learning rate schedule
USE_SCHEDULER = True
SCHEDULER_PATIENCE = 15
SCHEDULER_FACTOR = 0.5

# Logging
LOG_INTERVAL = 10  # Print loss every N epochs
SAVE_BEST = True   # Save best model based on validation loss

print("=" * 60)
print("🎨 AI Bead Mapper - Training with Perceptual Loss")
print("=" * 60)
print(f"Configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LR}")
print(f"  Loss Type: {LOSS_TYPE}")
print(f"  Model: {'Lightweight' if USE_LIGHTWEIGHT else 'Full'}")
print(f"  Save Path: {SAVE_PATH}")
print("=" * 60)

# ============ DATASET ============

# Data augmentation for better generalization
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),  # Random flips
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # Slight color variation
    transforms.ToTensor()
])

transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

print("\n📦 Loading STL10 dataset...")
train_set = datasets.STL10(
    root="./data", 
    split="train", 
    download=True, 
    transform=transform_train
)
val_set = datasets.STL10(
    root="./data", 
    split="test",  # Use test split for validation
    download=True, 
    transform=transform_val
)

train_loader = DataLoader(
    train_set, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_set, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=4,
    pin_memory=True
)

print(f"  Training samples: {len(train_set)}")
print(f"  Validation samples: {len(val_set)}")

# ============ MODEL & OPTIMIZER ============

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

# Select model
if USE_LIGHTWEIGHT:
    model = LightweightAutoencoder().to(device)
    print("  Model: Lightweight (faster training)")
else:
    model = BetterAutoencoder().to(device)
    print("  Model: Full (better quality)")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# Select loss function
if LOSS_TYPE == "vgg":
    criterion = VGGPerceptualLoss(mse_weight=1.0, vgg_weight=0.1, edge_weight=0.1)
    print("  Loss: VGG Perceptual Loss (best quality)")
elif LOSS_TYPE == "perceptual":
    criterion = PerceptualLoss(mse_weight=1.0, edge_weight=0.2, color_weight=0.1)
    print("  Loss: Perceptual Loss with Edge Preservation (recommended)")
else:
    criterion = torch.nn.MSELoss()
    print("  Loss: MSE (baseline)")

criterion = criterion.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Learning rate scheduler
scheduler = None
if USE_SCHEDULER:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR
    )
    print(f"  Scheduler: ReduceLROnPlateau (patience={SCHEDULER_PATIENCE})")

# ============ TRAINING FUNCTIONS ============

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'mse': 0.0, 'edge': 0.0, 'color': 0.0, 'vgg': 0.0}
    
    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y = model(x)
        
        # Calculate loss
        if LOSS_TYPE in ["perceptual", "vgg"]:
            loss, components = criterion(y, x)
            for key, val in components.items():
                if key in loss_components:
                    loss_components[key] += val
        else:
            loss = criterion(y, x)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Average components
    for key in loss_components:
        loss_components[key] /= len(loader)
    
    return avg_loss, loss_components


def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    loss_components = {'mse': 0.0, 'edge': 0.0, 'color': 0.0, 'vgg': 0.0}
    
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            y = model(x)
            
            if LOSS_TYPE in ["perceptual", "vgg"]:
                loss, components = criterion(y, x)
                for key, val in components.items():
                    if key in loss_components:
                        loss_components[key] += val
            else:
                loss = criterion(y, x)
            
            total_loss += loss.item() * x.size(0)
    
    avg_loss = total_loss / len(loader.dataset)
    
    for key in loss_components:
        loss_components[key] /= len(loader)
    
    return avg_loss, loss_components


# ============ TRAINING LOOP ============

print("\n" + "=" * 60)
print("🚀 Starting Training")
print("=" * 60)

best_val_loss = float('inf')
start_time = datetime.now()

for epoch in range(1, EPOCHS + 1):
    # Train
    train_loss, train_components = train_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    
    # Validate
    val_loss, val_components = validate(model, val_loader, criterion, device)
    
    # Learning rate scheduling
    if scheduler is not None:
        scheduler.step(val_loss)
    
    # Save best model
    if SAVE_BEST and val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH.replace('.pth', '_best.pth'))
    
    # Logging
    if epoch % LOG_INTERVAL == 0 or epoch == 1:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        if LOSS_TYPE in ["perceptual", "vgg"]:
            print(f"  Components (Val):")
            print(f"    MSE: {val_components['mse']:.4f}")
            print(f"    Edge: {val_components['edge']:.4f}")
            if 'color' in val_components and val_components['color'] > 0:
                print(f"    Color: {val_components['color']:.4f}")
            if 'vgg' in val_components and val_components['vgg'] > 0:
                print(f"    VGG: {val_components['vgg']:.4f}")

# ============ SAVE FINAL MODEL ============

torch.save(model.state_dict(), SAVE_PATH)
print("\n" + "=" * 60)
print(f"✅ Training Complete!")
print(f"  Final model saved: {SAVE_PATH}")
if SAVE_BEST:
    print(f"  Best model saved: {SAVE_PATH.replace('.pth', '_best.pth')}")
    print(f"  Best validation loss: {best_val_loss:.4f}")

elapsed = datetime.now() - start_time
print(f"  Training time: {elapsed}")
print("=" * 60)

# ============ TRAINING TIPS ============
print("\n💡 Next Steps:")
print("  1. Test the model: python test_model.py")
print("  2. Run the app: streamlit run app.py")
print("  3. Compare with old model to see improvements!")
print("\n  To use best model in app, update model path to:")
print(f"     {SAVE_PATH.replace('.pth', '_best.pth')}")