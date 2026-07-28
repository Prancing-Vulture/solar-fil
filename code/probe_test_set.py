import os
import json

import sys
sys.path.append(os.path.dirname(__file__))
from analysis import path as BASE_DATASET_PATH

def probe_test():
    print("=" * 60)
    print("PROBING TEST SET AND COMPETITION STRUCTURE")
    print("=" * 60)
    
    comp_dir = os.path.join(BASE_DATASET_PATH, "MAGFiLO_1.0_Kaggle_2026")
    test_dir = os.path.join(comp_dir, "test")
    test_img_dir = os.path.join(test_dir, "test_images")
    
    print("Listing files recursively in competition dir:")
    for root, dirs, files in os.walk(BASE_DATASET_PATH):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, BASE_DATASET_PATH)
            if not f.endswith('.jpeg') and not f.endswith('.jpg'):
                print(f"  File: {rel_path} ({os.path.getsize(full_path)} bytes)")
                
    test_files = os.listdir(test_img_dir) if os.path.exists(test_img_dir) else []
    print(f"\nTotal test image files on disk: {len(test_files)}")
    print(f"First 5 test files: {test_files[:5]}")
    
if __name__ == "__main__":
    probe_test()
