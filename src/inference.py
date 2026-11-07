import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.autoencoder import BetterAutoencoder
from src.color_mapping import BeadPalette

# Model loading

def load_model(model_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = BetterAutoencoder().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

# Passing through the autoencoder for soft "color quantization"

def autoencode_image(img_pil, model, device):
    """
    Pass image through autoencoder.
    
    FIXED: Always resize to exactly 128x128 to ensure proper tensor dimensions
    for skip connections in U-Net architecture.
    """
    # Store original size for later restoration
    original_size = img_pil.size
    
    # CRITICAL: Resize to EXACTLY 128x128 (not just smaller dimension)
    # This ensures all encoder/decoder layers have matching dimensions
    tfm = transforms.Compose([
        transforms.Resize((128, 128)),  # Force both dimensions to 128
        transforms.ToTensor()
    ])
    
    x = tfm(img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        y = model(x).clamp(0, 1)
    
    # Convert back to PIL image
    y_np = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    result = Image.fromarray(y_np)
    
    # Resize back to original size (or close to it)
    # This preserves the aspect ratio in the final output
    result = result.resize(original_size, Image.LANCZOS)
    
    return result

# Splitting into a grid and averaging the color in each block (bead size simulation)

def grid_average(pil_img: Image.Image, grid_w: int, grid_h: int):
    # More accurate averaging while preserving proportions
    w, h = pil_img.size
    W = (w // grid_w) * grid_w
    H = (h // grid_h) * grid_h
    pil_img = pil_img.crop((0, 0, W, H))
    arr = np.array(pil_img, dtype=np.float32)
    out = np.zeros_like(arr)

    for y in range(0, H, grid_h):
        for x in range(0, W, grid_w):
            block = arr[y:y+grid_h, x:x+grid_w]
            mean = block.reshape(-1, 3).mean(axis=0)
            out[y:y+grid_h, x:x+grid_w] = mean

    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

# Mapping visualization: replace each block color with the nearest one from the palette

def apply_palette_grid(pil_img: Image.Image, grid_w: int, grid_h: int, palette_df):
    if palette_df.empty:
        raise ValueError("Palette is empty. Please select at least one brand.")

    # The palette is built from the FILTERED dataframe
    palette_obj = BeadPalette(df=palette_df)

    w, h = pil_img.size
    W = (w // grid_w) * grid_w
    H = (h // grid_h) * grid_h
    arr = np.array(pil_img)[:H, :W]
    res = np.zeros_like(arr)
    used_indices = []

    for y in range(0, H, grid_h):
        for x in range(0, W, grid_w):
            block = arr[y:y+grid_h, x:x+grid_w]
            mean = block.reshape(-1, 3).mean(axis=0)
            matched, _, idxs = palette_obj.nearest(np.array([mean]))
            row = matched.iloc[0]
            res[y:y+grid_h, x:x+grid_w] = [row["r"], row["g"], row["b"]]
            used_indices.append(int(idxs[0]))  # indices are compatible with palette_df

    return Image.fromarray(res.astype(np.uint8)), np.array(used_indices, dtype=int)
