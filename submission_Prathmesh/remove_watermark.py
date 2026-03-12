import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

def detect_clean_image(img_lab: np.ndarray, threshold: float = 3.0) -> bool:
    """Detect if the image is already clean based on color variance.
    
    If the a* and b* channels have low standard deviation, the image is 
    likely grayscale/neutral (no colored watermark).
    """
    a_channel = img_lab[:, :, 1]
    b_channel = img_lab[:, :, 2]
    
    a_std = np.std(a_channel)
    b_std = np.std(b_channel)
    
    # Typical watermarks (blue/red) will have high std in a or b
    return (a_std < threshold) and (b_std < threshold)

def remove_watermark_improved(img_path: str) -> Optional[np.ndarray]:
    """Improved watermark removal using LAB colorspace and adaptive masking."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # 1. Convert to LAB for better color separation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 2. Check if clean
    if detect_clean_image(lab, threshold=2.5):
        # Already clean, just enhance contrast slightly
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        return l_enhanced

    # 3. Create Masks for Colored Watermarks
    # Blue/Cyan watermarks (b < 128)
    # Red/Magenta watermarks (a > 128)
    
    # Adaptive threshold for blue
    b_inv = 255 - b
    _, blue_mask = cv2.threshold(b_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive threshold for red
    _, red_mask = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(blue_mask, red_mask)
    
    # Refine mask: only target LIGHT areas (watermark is usually light)
    # Background/Watermark is high L, Text is low L.
    # We use a threshold that targets the upper half of the luminance
    _, light_mask = cv2.threshold(l, 160, 255, cv2.THRESH_BINARY)
    final_wm_mask = cv2.bitwise_and(combined_mask, light_mask)
    
    # 4. Remove Watermark from L Channel
    # Push masked pixels to background white
    l_cleaned = l.copy()
    l_cleaned[final_wm_mask > 0] = 255
    
    # 5. Contrast Enhancement
    # Apply CLAHE to sharpen text/diagram lines
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_cleaned)
    
    # 6. Edge-Preserving Smoothing
    l_smoothed = cv2.bilateralFilter(l_enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    
    # 7. Final Normalization & Contrast Stretch
    # Push light pixels to pure white to ensure background cleanliness
    _, binary_bg = cv2.threshold(l_smoothed, 220, 255, cv2.THRESH_BINARY)
    l_final = l_smoothed.copy()
    l_final[binary_bg > 0] = 255
    
    # Contrast stretch for dark pixels
    p5 = np.percentile(l_final, 5)
    denom = 255.0 - p5
    if denom > 0:
        l_final = np.clip((l_final.astype(float) - p5) * (255.0 / denom), 0, 255).astype(np.uint8)
    
    return l_final

def process_input(input_path_str: str, output_path_str: str):
    """Process either a single image or a directory of images."""
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)

    if input_path.is_file():
        # Single file mode
        print(f"Processing single file: {input_path}")
        result = remove_watermark_improved(str(input_path))
        if result is not None:
            # If output_path is a directory, name the file within it
            if output_path.suffix == "":
                output_path.mkdir(parents=True, exist_ok=True)
                final_output = output_path / input_path.with_suffix(".png").name
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                final_output = output_path
            
            cv2.imwrite(str(final_output), result)
            print(f"  Saved to: {final_output}")
        else:
            print(f"  FAILED to process {input_path}")
            
    elif input_path.is_dir():
        # Directory mode
        output_path.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png"}
        images = [f for f in sorted(input_path.iterdir()) if f.suffix.lower() in exts]

        if not images:
            print(f"No images found in {input_path}")
            return

        print(f"Processing {len(images)} images from {input_path}...")
        processed = 0
        failed = 0

        for i, img_path in enumerate(images):
            result = remove_watermark_improved(str(img_path))
            out_file = output_path / img_path.with_suffix(".png").name

            if result is not None:
                cv2.imwrite(str(out_file), result)
                processed += 1
            else:
                failed += 1
                print(f"  FAILED: {img_path.name}")

            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                print(f"  Progress: {i + 1}/{len(images)} ({(i+1)/len(images)*100:.0f}%)")

        print(f"\nDone: {processed} cleaned, {failed} failed")
        print(f"Output: {output_path}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_watermark.py <input_path> <output_path>")
        print("Example (Single file): python remove_watermark.py img.jpg result.png")
        print("Example (Directory):   python remove_watermark.py samples/watermarked output/")
        sys.exit(1)
    
    input_p = sys.argv[1]
    output_p = sys.argv[2]
    process_input(input_p, output_p)
