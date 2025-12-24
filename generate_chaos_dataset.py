import sys
import os
import numpy as np
import argparse
from tqdm import tqdm
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.painting import FeaturePainter
from src.projection import Calibration

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Full Chaotic Painted KITTI Dataset')
    parser.add_argument('--data_root', type=str, default='data', help='Root directory of KITTI data')
    parser.add_argument('--output_root', type=str, default=None, help='Root directory for output (defaults to data_root if not set)')
    parser.add_argument('--splits', type=str, nargs='+', default=['training', 'testing'], help='Data splits to process')
    parser.add_argument('--hyper', action='store_true', help='Use Hyper-Chaos mode (Spatially Adaptive)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check CUDA
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Initialize Painter ONCE
    # This loads the DeepLab and ChaosNet models
    painter = FeaturePainter(hyper_mode=args.hyper)
    
    for split in args.splits:
        base_dir = os.path.join(args.data_root, split)
        image_dir = os.path.join(base_dir, 'image_2')
        velo_dir = os.path.join(base_dir, 'velodyne')
        calib_dir = os.path.join(base_dir, 'calib')
        
        # Output directory
        # Use output_root if specified, otherwise use data_root
        out_base = os.path.join(args.output_root, split) if args.output_root else base_dir
        if args.hyper:
            output_dir = os.path.join(out_base, 'velodyne_hyper_chaos')
        else:
            output_dir = os.path.join(out_base, 'velodyne_chaos')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Processing {split} set...")
        print(f"Input: {velo_dir}")
        print(f"Output: {output_dir}")
        
        if not os.path.exists(image_dir):
            print(f"Skipping {split}: Image directory not found at {image_dir}")
            continue

        files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        for filename in tqdm(files):
            idx = filename.split('.')[0]
            
            image_path = os.path.join(image_dir, filename)
            velo_path = os.path.join(velo_dir, f"{idx}.bin")
            calib_path = os.path.join(calib_dir, f"{idx}.txt")
            output_path = os.path.join(output_dir, f"{idx}.bin")
            
            # Skip if already exists? (Optional, but good for resuming)
            # if os.path.exists(output_path): continue
            
            if not os.path.exists(velo_path): continue
            if not os.path.exists(calib_path): continue
                
            # Load Data
            calib = Calibration(calib_path)
            lidar_points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
            
            # Paint with CHAOS
            try:
                painted_points = painter.paint(image_path, lidar_points, calib, mode='chaos')
                
                # Check shape correctness once in a while or always? 
                # Doing it always is cheap enough for shape check
                if painted_points.shape[1] != 20:
                    print(f"Error: {idx} has shape {painted_points.shape}")
                
                # Save
                painted_points.astype(np.float32).tofile(output_path)
                
            except Exception as e:
                print(f"Error processing {idx}: {e}")
                import traceback
                traceback.print_exc()

    print("Full Dataset Generation Complete.")

if __name__ == "__main__":
    main()
