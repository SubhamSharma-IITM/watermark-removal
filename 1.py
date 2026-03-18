"""
Watermark Removal for Scanned Question Paper Images — v3.0

COMPLETE REWRITE based on empirical analysis of the full 120-image dataset.

Critical fixes over v2:
  [FIX-1] BLUE MASK: luminance threshold raised 50→140
           Watermarks are LIGHT-blue (semi-transparent over white, L>140).
           Diagram content is DARK-blue (ink, L<80). Old threshold erased wm_097 sphere.
  [FIX-2] IMAGE CLASSIFIER: DARK_BG / COLORFUL_POSTER / WHITE_BG_LINE_ART
           Dark-background images (blackboard) now pass through RAW — no destruction.
  [FIX-3] GREY-WORLD WHITE BALANCE for extreme blue-cast images (Vernier scale type).
           YCbCr alone was insufficient for images with mean_b* < -6.
  [FIX-4] BRIGHT-REGION b* check: watermarks tint near-white background pixels.
           Detects faint watermarks (YEVA J, THE NAR) even with low pixel count.
  [FIX-5] Detector thresholds tuned down across the board.

Pipeline:
  Stage 0 : Classify image type + analyse watermark
  Stage 1 : Red watermark removal (HSV, elliptic dilation)
  Stage 2 : Blue watermark removal (LAB b*, luminance-constrained L>140)
  Stage 3 : Colour cast correction (grey-world WB or YCbCr depending on severity)
  Stage 4 : Grayscale (Rec.709 weights)
  Stage 5 : Adaptive LUT (per-image Gaussian histogram valley)
  Stage 6 : Edge-aware enhancement (bilateral + CLAHE + gentle unsharp)
  Stage 7 : Mask re-application

Tech: OpenCV + NumPy (primary), Pillow (fallback)
"""

import io
import os
import sys
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image, ImageFilter
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Image Type Constants
# ══════════════════════════════════════════════════════════════════════════════

class ImageType:
    WHITE_BG  = "white_bg"    # Standard question paper: white background, black lines
    COLORFUL  = "colorful"    # High-saturation marketing/poster with photos
    DARK_BG   = "dark_bg"     # Dark/black background (blackboard, inverted scan)


# ══════════════════════════════════════════════════════════════════════════════
# Analysis Container
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ImageAnalysis:
    image_type:           str   = ImageType.WHITE_BG
    has_blue_watermark:   bool  = False
    has_red_watermark:    bool  = False
    has_grey_watermark:   bool  = False
    has_color_cast:       bool  = False
    is_clean:             bool  = False
    extreme_color_cast:   bool  = False   # triggers grey-world WB
    blue_pixel_ratio:     float = 0.0
    red_pixel_ratio:      float = 0.0
    mean_b_star:          float = 0.0     # overall LAB b* (negative=blue)
    bright_b_star:        float = 0.0     # b* of near-white pixels (bg tint check)
    suggested_threshold:  int   = 185
    reason:               str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0a — Image Type Classifier
# ══════════════════════════════════════════════════════════════════════════════

def classify_image(img_bgr: np.ndarray) -> str:
    """
    Classify into DARK_BG / COLORFUL / WHITE_BG.

    DARK_BG   : mean brightness < 90  (blackboard, inverted)
    COLORFUL  : significant orange/red/warm saturated pixels (promotional posters)
    WHITE_BG  : default — standard exam diagrams on white paper
    """
    if img_bgr is None:
        return ImageType.WHITE_BG

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 90:
        return ImageType.DARK_BG

    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat  = hsv[:, :, 1].astype(np.float32)
    total_px = img_bgr.shape[0] * img_bgr.shape[1]

    # Orange/red/warm tones: R-dominant, high saturation
    r_f = img_bgr[:, :, 2].astype(np.float32)
    b_f = img_bgr[:, :, 0].astype(np.float32)
    warm_mask = (r_f > 140) & (r_f > b_f * 1.25) & (sat > 80)
    warm_ratio = float(np.count_nonzero(warm_mask) / total_px)
    mean_sat   = float(np.mean(sat))

    if warm_ratio > 0.08 and mean_sat > 30:
        return ImageType.COLORFUL

    return ImageType.WHITE_BG


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0b — Watermark Detector
# ══════════════════════════════════════════════════════════════════════════════

class WatermarkDetector:
    """
    Multi-feature statistical classifier with tuned thresholds.

    Key improvement over v2: check b* of BRIGHT (near-white) pixels.
    A watermark tints the white background blue, so even faint watermarks
    shift bright-pixel b* negative — detectable even at low pixel count.
    """

    BLUE_OVERALL_THRESHOLD  = -2.5   # overall mean b* trigger [FIX-1: LOWERED]
    BLUE_BRIGHT_THRESHOLD   = -2.0   # b* of pixels with L>200 [FIX-1: LOWERED]
    BLUE_PIXEL_RATIO        = 0.002  # fraction of strongly-blue pixels [FIX-1: LOWERED]
    EXTREME_CAST_THRESHOLD  = -5.5   # triggers grey-world white balance

    RED_MIN = 0.0006
    RED_MAX = 0.15

    def analyze(self, img_bgr: np.ndarray) -> ImageAnalysis:
        result = ImageAnalysis()
        if img_bgr is None:
            result.reason = "null image"
            return result

        result.image_type = classify_image(img_bgr)

        # Dark background → skip all watermark detection, pass through
        if result.image_type == ImageType.DARK_BG:
            result.is_clean = True
            result.reason   = "dark_bg → pass-through raw"
            return result

        total_px = img_bgr.shape[0] * img_bgr.shape[1]

        # ── LAB colour analysis ───────────────────────────────────────────
        lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_ch = lab[:, :, 0]          # 0–255
        b_ch = lab[:, :, 2] - 128.0  # centred; negative = blue

        mean_b = float(np.mean(b_ch))
        result.mean_b_star = mean_b
        result.extreme_color_cast = mean_b < self.EXTREME_CAST_THRESHOLD

        # b* of near-white pixels (background watermark tint check)
        bright_px = L_ch > 200
        result.bright_b_star = (float(np.mean(b_ch[bright_px]))
                                 if np.count_nonzero(bright_px) > 500 else mean_b)

        result.has_color_cast = (
            mean_b < self.BLUE_OVERALL_THRESHOLD
            or result.bright_b_star < self.BLUE_BRIGHT_THRESHOLD
        )

        # Strongly blue-shifted LIGHT pixels (CRITICAL: L>140 not L>50)
        # This is the fix for wm_097: dark blue sphere (L<80) is NOT targeted
        blue_blob = (b_ch < -8) & (L_ch > 140) & (L_ch < 247)
        blue_ratio = float(np.count_nonzero(blue_blob) / total_px)
        result.blue_pixel_ratio = blue_ratio

        result.has_blue_watermark = (
            blue_ratio > self.BLUE_PIXEL_RATIO
            or result.bright_b_star < self.BLUE_BRIGHT_THRESHOLD
            or mean_b < self.BLUE_OVERALL_THRESHOLD
        )

        # ── Red pixels ────────────────────────────────────────────────────
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        m1  = cv2.inRange(hsv, np.array([0,   50,  50]), np.array([10,  255, 255]))
        m2  = cv2.inRange(hsv, np.array([165,  50,  50]), np.array([180, 255, 255]))
        red_ratio = float((np.count_nonzero(m1) + np.count_nonzero(m2)) / total_px)
        result.red_pixel_ratio  = red_ratio
        result.has_red_watermark = self.RED_MIN < red_ratio < self.RED_MAX

        # ── Histogram valley (grey watermark detection) ────────────────────
        gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        sigma = 10
        x     = np.arange(-3 * sigma, 3 * sigma + 1, dtype=np.float32)
        gauss = np.exp(-x**2 / (2 * sigma**2)); gauss /= gauss.sum()
        smooth = np.convolve(hist, gauss, mode="same")

        search       = smooth[110:230]
        valley_off   = int(np.argmin(search))
        valley_pos   = valley_off + 110
        valley_val   = smooth[valley_pos]
        left_peak    = float(np.max(smooth[60:valley_pos]))  if valley_pos > 60  else 0.0
        right_peak   = float(np.max(smooth[valley_pos:240])) if valley_pos < 240 else 0.0
        valley_depth = 1.0 - valley_val / (max(left_peak, right_peak) + 1e-6)
        has_valley   = valley_depth > 0.22 and left_peak > 30

        result.has_grey_watermark  = has_valley and not result.has_blue_watermark
        result.suggested_threshold = max(135, min(valley_pos, 225))

        result.is_clean = not (
            result.has_blue_watermark
            or result.has_red_watermark
            or result.has_grey_watermark
        )
        result.reason = self._reason(result)
        return result

    @staticmethod
    def _reason(r: ImageAnalysis) -> str:
        parts = [f"type={r.image_type}"]
        if r.has_blue_watermark:
            parts.append(f"blue(px={r.blue_pixel_ratio:.3f} b*={r.mean_b_star:.1f} bg={r.bright_b_star:.1f})")
        if r.has_red_watermark:
            parts.append(f"red(px={r.red_pixel_ratio:.4f})")
        if r.has_grey_watermark:
            parts.append("grey_wm")
        if r.extreme_color_cast:
            parts.append("EXTREME_CAST→grey_world_WB")
        if r.is_clean:
            parts.append("CLEAN")
        return " | ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Red Watermark Removal
# ══════════════════════════════════════════════════════════════════════════════

def remove_red_watermark(
    img_bgr: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove red / maroon / orange-red watermark via HSV masking.
    - 4 hue ranges: standard red, dark-wrap red, maroon, orange-red
    - Elliptic 5×5 dilation ×2 to cover anti-aliased glyph edges
    - Guard: skip if < 8 pixels or > 15% of image (avoid false positive)
    """
    hsv      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    total_px = img_bgr.shape[0] * img_bgr.shape[1]

    m1 = cv2.inRange(hsv, np.array([0,   50,  50]), np.array([10,  255, 255]))
    m2 = cv2.inRange(hsv, np.array([165,  50,  50]), np.array([180, 255, 255]))
    m3 = cv2.inRange(hsv, np.array([0,   40,  30]), np.array([15,  255, 200]))  # maroon
    m4 = cv2.inRange(hsv, np.array([0,   60, 100]), np.array([20,  255, 255]))  # orange-red
    red_mask = m1 | m2 | m3 | m4

    count = np.count_nonzero(red_mask)
    if count < 8 or count / total_px > 0.15:
        return img_bgr.copy(), None

    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(red_mask, kernel, iterations=2)
    result  = img_bgr.copy()
    result[dilated > 0] = [255, 255, 255]
    return result, dilated


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Blue Watermark Removal  [FIX-1: L > 140 constraint]
# ══════════════════════════════════════════════════════════════════════════════

def remove_blue_watermark(
    img_bgr:  np.ndarray,
    analysis: ImageAnalysis,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    LAB b*-channel blue watermark isolation.

    THE KEY FIX (v3):
        mask requires L_ch > 140   (was L_ch > 50 in v2)

    Physics:
        Semi-transparent blue watermark over white background:
            white (L≈255) + blue tint = light-blue pixel → L ≈ 150–240  ← TARGETED
        Actual dark-blue diagram ink (sphere outline, arrows):
            blue ink on white = dark-blue pixel → L < 80                 ← PROTECTED

    Without this fix, the entire electric-field sphere diagram (wm_097)
    was erased because its blue outlines had L ≈ 60–90, which the old
    threshold of L>50 caught as "watermark".
    """
    lab  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_ch = lab[:, :, 0]
    b_ch = lab[:, :, 2] - 128.0

    # Only LIGHT blue pixels (watermark semi-transparent overlay)
    mask = (
        (b_ch < -6)
        & (L_ch > 140)   # ← THE critical fix; protects dark blue diagram content
        & (L_ch < 247)
    ).astype(np.uint8) * 255

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.erode(mask, erode_k, iterations=1)

    pixel_count = np.count_nonzero(mask)
    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
    
    # FIX-2: If too few pixels detected, try lower luminance threshold (catch faint watermarks)
    if pixel_count / total_pixels < 0.001:
        mask_fallback = (
            (b_ch < -6)
            & (L_ch > 120)   # Relaxed threshold for faint watermarks
            & (L_ch < 247)
        ).astype(np.uint8) * 255
        mask_fallback = cv2.morphologyEx(mask_fallback, cv2.MORPH_CLOSE, close_k)
        mask_fallback = cv2.erode(mask_fallback, erode_k, iterations=1)
        if np.count_nonzero(mask_fallback) > pixel_count:
            mask = mask_fallback

    if np.count_nonzero(mask) / total_pixels < 0.001:
        return img_bgr.copy(), None

    # FIX-3: Zero out b* channel in LAB space (removes blue tint completely, preserves L for brightness)
    result_lab = lab.copy()
    result_lab[:, :, 2] = np.where(mask > 0, 128, result_lab[:, :, 2])  # Set b* to neutral (128)
    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result, mask


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Colour Cast Correction  [FIX-3: grey-world WB]
# ══════════════════════════════════════════════════════════════════════════════

def grey_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grey-world white balance: assumes scene average is neutral grey.
    Used for EXTREME blue cast (vernier scale, heavy-tint scans).
    Scales each channel so its mean equals the overall mean luminance.
    """
    b, g, r = (img_bgr[:, :, c].astype(np.float64) for c in range(3))
    mb, mg, mr = np.mean(b), np.mean(g), np.mean(r)
    mu = (mb + mg + mr) / 3.0
    result = np.stack([
        np.clip(img_bgr[:, :, c].astype(np.float64) * (mu / ([mb, mg, mr][c] + 1e-6)), 0, 255)
        for c in range(3)
    ], axis=-1).astype(np.uint8)
    return result


def correct_ycbcr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Moderate blue cast correction via YCbCr channel neutralisation.
    Pushes Cb and Cr towards neutral (128) in non-text regions only,
    so diagram lines (low Y) are completely unaffected.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    text_w = np.clip((80 - Y) / 30, 0, 1)
    bg_w   = 1.0 - text_w
    corrected = np.clip(
        np.stack([Y, Cr * text_w + 128.0 * bg_w, Cb * text_w + 128.0 * bg_w], axis=-1),
        0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_YCrCb2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Adaptive LUT (Gaussian histogram valley)
# ══════════════════════════════════════════════════════════════════════════════

def build_adaptive_lut(gray: np.ndarray, suggested: int = 185) -> np.ndarray:
    """
    Per-image LUT from Gaussian-smoothed histogram valley detection.
    Sigmoid transition zone for smooth blending (no step-edge artefacts).
    """
    hist  = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    sigma = 9
    x     = np.arange(-3 * sigma, 3 * sigma + 1, dtype=np.float32)
    gauss = np.exp(-x**2 / (2 * sigma**2)); gauss /= gauss.sum()
    smooth = np.convolve(hist, gauss, mode="same")

    valley = int(np.argmin(smooth[110:230])) + 110
    if valley <= 110 or valley >= 228:
        valley = suggested

    # FIX: Clamp valley to reasonable range (prevent too-low valleys causing white-washing)
    valley = max(135, min(valley, 225))
    
    tw  = 40  # Transition width (reduced to match original behavior better)
    lut = np.empty(256, dtype=np.float32)
    for i in range(256):
        if i <= valley - tw:
            # Map dark pixels linearly: 0→0, (valley-tw)→dark_cap (usually 80-120)
            dark_cap = min(110, valley - tw)
            lut[i] = (i / max(valley - tw, 1)) * dark_cap
        elif i <= valley:
            # Smooth sigmoid transition from dark_cap to 255
            dark_cap = min(110, valley - tw)
            t = (i - (valley - tw)) / float(tw)
            s = 1.0 / (1.0 + np.exp(-8.0 * (t - 0.5)))
            lut[i] = dark_cap * (1.0 - s) + 255.0 * s
        else:
            # All pixels beyond valley go to white
            lut[i] = 255.0
    return np.clip(lut, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Edge-Aware Enhancement
# ══════════════════════════════════════════════════════════════════════════════

def enhance(gray: np.ndarray) -> np.ndarray:
    """
    1. Percentile auto-contrast  (2nd / 98th)
    2. CLAHE                     (local contrast, clipLimit=1.5)
    3. Bilateral filter          (edge-preserving smoothing)
    4. Conservative unsharp mask (σ=1.5, amount=0.35 — no halo artefacts)
    """
    # FIX-4: Auto-contrast stretch: pull darkest pixels toward black
    dark = gray[gray < 250]
    if len(dark) > 100:  # FIX-4: Lowered threshold to handle more images
        p_low = float(np.percentile(dark, 5))  # FIX-4: 5th percentile instead of 2nd (more robust)
        if p_low < 240:
            gray = np.clip(
                (gray.astype(np.float32) - p_low) / (255.0 - p_low) * 255, 0, 255
            ).astype(np.uint8)

    # FIX-6: Apply enhancement with reduced parameters to avoid over-sharpening
    gray = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16)).apply(gray)  # FIX-6: Reduced from 1.5
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=12, sigmaSpace=12)  # FIX-6: Reduced from 18
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    # FIX-6: Conservative unsharp mask: avoid halos on thin lines
    return np.clip(cv2.addWeighted(gray, 1.25, blur, -0.25, 0), 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

_detector: Optional[WatermarkDetector] = None


def remove_watermark(img_path: str) -> Optional[bytes]:
    """
    Process a single image. Returns cleaned PNG bytes, or None on failure.
    """
    if HAS_CV2:
        return _process_cv2(img_path)
    elif HAS_PIL:
        return _process_pil(img_path)
    logger.error("Neither OpenCV nor Pillow is installed.")
    return None


def _process_cv2(img_path: str) -> Optional[bytes]:
    global _detector
    if _detector is None:
        _detector = WatermarkDetector()

    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Cannot read: {img_path}")
        return None

    # ── Stage 0: Classify + Analyse ──────────────────────────────────────
    analysis = _detector.analyze(img)
    logger.debug(f"  {Path(img_path).name}: {analysis.reason}")

    # ── DARK BG: return raw image unchanged ──────────────────────────────
    if analysis.image_type == ImageType.DARK_BG:
        return _enc(img)

    # ── CLEAN: minimal enhancement only ──────────────────────────────────
    if analysis.is_clean:
        return _enc(enhance(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    work      = img
    red_mask  = None
    blue_mask = None

    # ── Stage 1: Red removal ─────────────────────────────────────────────
    if analysis.has_red_watermark:
        work, red_mask = remove_red_watermark(work)

    # ── Stage 2: Blue removal (luminance-constrained) ─────────────────────
    if analysis.has_blue_watermark:
        work, blue_mask = remove_blue_watermark(work, analysis)

    # ── Stage 3: Colour cast correction ──────────────────────────────────
    if analysis.extreme_color_cast:
        work = grey_world_white_balance(work)   # [FIX-3]
    elif analysis.has_color_cast:
        work = correct_ycbcr(work)

    # ── Stage 4: Grayscale (Rec.709) ──────────────────────────────────────
    b_f, g_f, r_f = (work[:, :, c].astype(np.float32) for c in range(3))
    gray = np.clip(0.2126 * r_f + 0.7152 * g_f + 0.0722 * b_f, 0, 255).astype(np.uint8)

    # ── Stage 5: Adaptive LUT ────────────────────────────────────────────
    gray = cv2.LUT(gray, build_adaptive_lut(gray, analysis.suggested_threshold))

    # ── Stage 6: Enhancement ─────────────────────────────────────────────
    gray = enhance(gray)

    # ── Stage 7: Re-apply masks → pure white ─────────────────────────────
    for mask in (red_mask, blue_mask):
        if mask is not None and mask.shape[:2] == gray.shape[:2]:
            gray[mask > 0] = 255

    return _enc(gray)


def _process_pil(img_path: str) -> Optional[bytes]:
    """Pillow-only fallback (no OpenCV)."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return None
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b

    # Red watermark
    red = (r > 120) & (r > g * 1.5) & (r > b * 1.5)
    if 8 < np.count_nonzero(red) < arr.shape[0] * arr.shape[1] * 0.15:
        arr[red] = [255, 255, 255]

    # Blue watermark — luminance-constrained (same FIX-1 applied here too)
    blue = (b > r + 20) & (b > g + 15) & (b < 238) & (lum > 140)
    if np.count_nonzero(blue) > arr.shape[0] * arr.shape[1] * 0.005:
        for c in range(3):
            arr[:, :, c] = np.where(blue, lum, arr[:, :, c])

    gray = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    p50  = float(np.percentile(gray, 50))
    thr  = min(max(p50 + 20, 160), 210)
    out  = np.clip(np.where(gray > thr, 255.0, gray), 0, 255).astype(np.uint8)
    buf  = io.BytesIO()
    Image.fromarray(out).filter(ImageFilter.SHARPEN).save(buf, format="PNG")
    return buf.getvalue()


def _enc(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return buf.tobytes() if ok else None


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def process_directory(input_dir: str, output_dir: str, verbose: bool = False) -> None:
    """Batch-process every image in input_dir → output_dir as PNG."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in exts)

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Processing {len(images)} images  {src} → {dst}")
    ok = fail = 0

    for i, p in enumerate(images):
        t0   = time.perf_counter()
        data = remove_watermark(str(p))
        ms   = (time.perf_counter() - t0) * 1000
        out  = dst / p.with_suffix(".png").name

        if data:
            out.write_bytes(data)
            ok += 1
        else:
            import shutil
            shutil.copy2(p, dst / p.name)
            fail += 1
            logger.warning(f"  FAILED (copied original): {p.name}")

        if (i + 1) % 10 == 0 or (i + 1) == len(images):
            logger.info(f"  [{i+1:>4}/{len(images)}] {p.name}  ({ms:.0f} ms)")

    logger.info(f"\nFinished — {ok} processed, {fail} fallback copies")
    logger.info(f"Output : {dst.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_watermark.py <input_dir> [output_dir] [--verbose]")
        print("  e.g. python remove_watermark.py samples/watermarked output/")
        sys.exit(1)
    _in  = sys.argv[1]
    _out = (sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "output")
    process_directory(_in, _out, verbose="--verbose" in sys.argv)
