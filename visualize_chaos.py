import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ==========================================
# 1. Chaos Logic (Skew Tent Map)
# ==========================================
class ChaosActivation(nn.Module):
    def __init__(self, channels, steps=5, init_skew=0.4):
        super().__init__()
        self.steps = steps
        # Define 16 different skew values to show diversity
        # ranging from 0.1 to 0.9
        self.skews = torch.linspace(0.1, 0.9, channels).view(1, channels, 1, 1)
            
    def forward(self, x):
        # x: (Batch, Channels, H, W)
        current_x = x
        b = self.skews.to(x.device)
        
        # Iteration Loop
        for _ in range(self.steps):
            cond = (current_x < b)
            next_x = torch.where(
                cond,
                current_x / b,
                (1.0 - current_x) / (1.0 - b)
            )
            current_x = next_x
            
        return current_x

# ==========================================
# 2. Synthetic Data Generation
# ==========================================
def generate_synthetic_image(h=256, w=256):
    """
    Creates an image with:
    1. Smooth Gradient (Low Freq)
    2. Strong Edge (High Freq)
    3. Noise Texture (Chaos Freq)
    """
    img = np.zeros((h, w), dtype=np.float32)
    
    # Gradient
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xv, yv = np.meshgrid(x, y)
    img += xv * 0.5
    
    # Circle (Edge)
    mask = (xv - 0.5)**2 + (yv - 0.5)**2 < 0.2
    img[mask] = 0.8
    
    # Texture / Noise (The stuff Chaos loves)
    noise = np.random.randn(h, w) * 0.05
    img += noise
    
    return np.clip(img, 0, 1)

# ==========================================
# 3. Main Visualization
# ==========================================
def main():
    # Setup
    out_dir = 'holographic_chaos/assets'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    print("Loading Real Input Image...")
    # Use a real image found in the previous step
    # C:\Users\yanbo\Downloads\Papers\uncertainty_painting\vis_results\vis_0001\0005.png
    input_path = r"C:\Users\yanbo\Downloads\Papers\uncertainty_painting\vis_results\vis_0001\0005.png"
    
    if os.path.exists(input_path):
        raw_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        raw_img = cv2.resize(raw_img, (512, 512))
        raw_img = raw_img.astype(np.float32) / 255.0
    else:
        print("Warning: Real image not found, falling back to synthetic (but better)")
        raw_img = generate_synthetic_image(512, 512)

    # Prepare Tensor: replicate to 16 channels to simulate feature map
    input_tensor = torch.from_numpy(raw_img).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    input_tensor = input_tensor.repeat(1, 16, 1, 1) # (1, 16, H, W)
    
    # Initialize Chaos
    # CRITICAL FIX: Reduce steps from 10 to 3. 
    # At step 10, the map becomes purely random noise (ergodic property).
    # At to step 3, it highlights edges and texture variance.
    model = ChaosActivation(channels=16, steps=3) 
    
    print("Running Skew Tent Map Dynamics (Steps=3)...")
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # ==========================================
    # 4. Plotting
    # ==========================================
    print("Plotting results...")
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # Plot Input
    ax_in = axes[0, 0]
    ax_in.imshow(raw_img, cmap='gray')
    ax_in.set_title("Input (Real Image)", fontsize=10)
    ax_in.axis('off')
    
    # Plot 16 Chaos Channels
    flat_axes = axes.flatten()
    
    for i in range(16):
        ax = flat_axes[i+1] # Shift by 1
        
        # Get channel i
        feat = output_tensor[0, i, :, :].numpy()
        skew_val = model.skews[0, i, 0, 0].item()
        
        ax.imshow(feat, cmap='inferno') # Use heat map
        ax.set_title(f"Chaos Ch-{i}\n(Skew b={skew_val:.2f})", fontsize=8)
        ax.axis('off')
        
    for i in range(17, 20):
        flat_axes[i].axis('off')
        
    out_path = os.path.join(out_dir, 'chaos_vis_new.png')
    plt.suptitle("Holographic Chaos (Step 3): Structure Preserved, Texture Enhanced", fontsize=16)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
