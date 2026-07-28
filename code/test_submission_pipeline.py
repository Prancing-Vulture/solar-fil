import os
import json
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

def test_rle_and_polygon():
    print("Testing RLE and Polygon conversion utilities...")
    
    # Create synthetic binary mask
    h, w = 256, 256
    dummy_mask = np.zeros((h, w), dtype=np.uint8)
    dummy_mask[50:150, 50:150] = 1 # Square filament
    
    # Fortran order for pycocotools mask encoding
    rle = mask_utils.encode(np.asfortranarray(dummy_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    area = float(mask_utils.area(rle))
    bbox = [float(x) for x in mask_utils.toBbox(rle)] # [x, y, w, h]
    
    print(f"RLE Area: {area}")
    print(f"RLE BBox: {bbox}")
    
    # Test opencv polygon contour extraction
    import cv2
    contours, _ = cv2.findContours(dummy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found contours count: {len(contours)}")
    
    poly_pts = contours[0].flatten().tolist()
    print(f"Polygon points count: {len(poly_pts)} (Sample: {poly_pts[:6]})")
    
    print("Test passed successfully!")

if __name__ == "__main__":
    test_rle_and_polygon()
