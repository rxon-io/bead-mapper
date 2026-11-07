import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from skimage.color import rgb2lab

# Transfer palette and image to LAB for more natural distances

def rgb_array_to_lab(rgb_arr):
    rgb_norm = (rgb_arr / 255.0).reshape(-1, 1, 3)
    return rgb2lab(rgb_norm).reshape(-1, 3)

class BeadPalette:
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None):
        if df is None:
            df = pd.read_csv(csv_path)
        # it is important to reset the index so that used_indices match df.iloc
        self.df = df.reset_index(drop=True)
        self.rgb = self.df[["r","g","b"]].to_numpy(dtype=np.float32)
        self.lab = rgb_array_to_lab(self.rgb)
        self.kdt = KDTree(self.lab)

    def nearest(self, rgb_pixels: np.ndarray):
        lab = rgb_array_to_lab(rgb_pixels.astype(np.float32))
        dist, idx = self.kdt.query(lab, k=1)
        idx = idx.flatten()
        matched = self.df.iloc[idx].reset_index(drop=True)
        return matched, dist.flatten(), idx  # <-- we also return the indexes relative to df
