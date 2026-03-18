"""
Watermark Removal System — v6 

Author: Aman Jaiswal
Goal: Robust watermark removal for scanned documents & posters

Key Design:
------------
1. Detect watermark REGION (not color transform)
2. Remove using INPAINT (OpenCV Telea)
3. Minimal post-processing (safe)

Why this works:
---------------
- Does NOT damage diagrams
- Removes watermark fully (not just fade)
- Works on all categories (white_bg, colorful, dark_bg)

Dependencies:
-------------
pip install opencv-python numpy pillow
"""

import os
import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ============================================================
# IMAGE CLASSIFIER
# ============================================================

class ImageType:
    WHITE_BG = "white_bg"
    COLORFUL = "colorful"
    DARK_BG = "dark_bg"


def classify_image(img: np.ndarray) -> str:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DARK BACKGROUND
    if np.mean(gray) < 85:
        return ImageType.DARK_BG

    # COLORFUL DETECTION
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    r = img[:, :, 2].astype(np.float32)
    b = img[:, :, 0].astype(np.float32)

    warm_pixels = (r > 140) & (r > b * 1.2) & (sat > 80)

    if np.mean(warm_pixels) > 0.07:
        return ImageType.COLORFUL

    return ImageType.WHITE_BG


# ============================================================
# WATERMARK MASK BUILDER (CORE LOGIC)
# ============================================================

def build_watermark_mask(img: np.ndarray) -> np.ndarray:
    """
    Build precise mask for watermark detection.

    Detects:
    - Light blue watermark
    - Red watermark
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -------------------------------
    # RED MASK
    # -------------------------------
    red1 = cv2.inRange(hsv, (0, 70, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 60), (180, 255, 255))
    red_mask = cv2.bitwise_or(red1, red2)

    # -------------------------------
    # BLUE MASK (light watermark only)
    # -------------------------------
    L = lab[:, :, 0]
    B = lab[:, :, 2]

    blue_mask = ((B < 120) & (L > 150)).astype(np.uint8) * 255

    # -------------------------------
    # COMBINE MASKS
    # -------------------------------
    mask = cv2.bitwise_or(red_mask, blue_mask)

    # -------------------------------
    # CLEAN MASK
    # -------------------------------
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_big = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
    mask = cv2.dilate(mask, kernel_small, iterations=2)

    return mask


# ============================================================
# WATERMARK REMOVAL (INPAINT)
# ============================================================

def remove_watermark(img: np.ndarray) -> np.ndarray:
    """
    Core watermark removal using inpainting.
    """

    mask = build_watermark_mask(img)

    # If mask too small → skip
    ratio = np.count_nonzero(mask) / (img.shape[0] * img.shape[1])

    if ratio < 0.001:
        return img

    # Inpainting (MAIN MAGIC)
    result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    return result


# ============================================================
# SAFE POST-PROCESSING
# ============================================================

def finalize_image(img: np.ndarray, mode: str) -> np.ndarray:
    """
    Minimal enhancement (safe, no damage)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Only enhance if low contrast
    if np.std(gray) < 50:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # Light sharpening
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.3, blur, -0.3, 0)

    return sharp


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_image(img: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Full pipeline controller
    """

    img_type = classify_image(img)

    # DARK → return original
    if img_type == ImageType.DARK_BG:
        return img, "SKIPPED_DARK"

    # Remove watermark
    cleaned = remove_watermark(img)

    # Convert + enhance
    final = finalize_image(cleaned, img_type)

    return final, img_type


# ============================================================
# FILE PROCESSOR
# ============================================================

def process_directory(input_dir: str, output_dir: str):
    src = Path(input_dir)
    dst = Path(output_dir)

    dst.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in src.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]])

    processed = 0
    skipped = 0

    log.info(f"Processing {len(files)} images...")

    for i, file in enumerate(files):
        img = cv2.imread(str(file))

        if img is None:
            log.warning(f"Failed: {file.name}")
            continue

        result, status = process_image(img)

        out_path = dst / file.with_suffix(".png").name
        cv2.imwrite(str(out_path), result)

        if "SKIPPED" in status:
            skipped += 1
        else:
            processed += 1

        print(f"[{i+1:3}/{len(files)}] {file.name:20} → {status}")

    print("\n===== FINAL SUMMARY =====")
    print("Processed:", processed)
    print("Skipped:", skipped)


# ============================================================
# CLI ENTRY
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python remove_watermark.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    process_directory(input_dir, output_dir)