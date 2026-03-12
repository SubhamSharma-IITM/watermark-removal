"""Watermark Removal v2 for scanned question paper images.

Key improvements over v1:
  - LAB colorspace b*-channel analysis: cleanly isolates blue watermarks from
    black diagram content without throwing away chromatic information
  - Per-image classification (none / blue / red / grey / mixed): clean images
    get minimal enhancement instead of aggressive processing
  - Confidence-map blending: smooth, artifact-free blue watermark suppression
    via a pixel-level certainty score rather than a hard threshold
  - Otsu-blended adaptive threshold: handles grey watermarks better than the
    fixed [160, 210] range in v1
  - CLAHE local contrast + bilateral filter + unsharp mask: replaces the
    halo-prone 3x3 Laplacian kernel with an edge-preserving pipeline

Usage:
    python src/remove_watermark_v2.py samples/watermarked output_v2/
    python src/remove_watermark_v2.py samples/watermarked output_v2/ --verbose
    python src/remove_watermark_v2.py samples/clean output_v2_clean/  # mostly passthrough
"""

import os
import sys
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


# ── Image Analysis / Classification ──────────────────────────────────────────


@dataclass
class ImageAnalysis:
    """Per-image watermark classification result."""

    wm_type: str  # "none" | "blue" | "blue_sat" | "red" | "grey" | "mixed"
    has_red: bool
    has_blue: bool  # combined: has_blue_lab OR has_blue_saturated
    has_blue_lab: bool  # semi-transparent light-blue cast (LAB b* signal)
    has_blue_saturated: bool  # opaque/saturated blue logo (HSV hue signal)
    has_grey: bool
    red_ratio: float  # fraction of pixels that are saturated-red
    blue_fraction: float  # fraction of bright pixels with strong blue shift (LAB)
    blue_hsv_ratio: float  # fraction of all pixels that are saturated blue/cyan (HSV)
    cool_fraction: float  # fraction of bright pixels with mild cool cast (b < 124)
    grey_mass: float  # histogram mass in mid-luminance range [80, 200]
    mean_b_shift: float  # mean b* shift of bright pixels (positive = more blue)


def analyze_image(img_bgr: np.ndarray) -> ImageAnalysis:
    """Classify watermark type using multi-channel statistics.

    Four independent signals are checked:

    Red        — HSV hue masking: saturated red wraps around H=0/180.
                 Ratio must be 0.05%–15% (present but not a red diagram).

    Blue (LAB) — LAB b* channel: light semi-transparent blue watermarks have
                 b < 116 (standard b* < -12) in bright pixels (L > 100).
                 Catches light blue tints and semi-transparent watermarks.

    Blue (HSV) — Strict HSV hue masking for saturated blue logos only
                 (H=95-130, S>=90, V<=220).  This catches opaque blue graphics
                 while avoiding low-saturation sky-blue page tints that caused
                 over-processing in the previous revision.

    Cool-cast  — Mild blue cast in bright pixels (fraction of b < 124), used
                 as a fallback to route very light blue-tinted pages to the
                 grey/adaptive path instead of clean passthrough.

    Grey       — Luminance histogram: grey watermarks create histogram mass in
                 the mid-tone range L=[80,200] above 5% of total pixels.

    The signals are combined into: none / blue / blue_sat / red / grey / mixed.
    """
    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # ── Red detection (HSV) ──────────────────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_r1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask_r2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    red_mask = mask_r1 | mask_r2
    red_ratio = float(np.count_nonzero(red_mask)) / total_pixels
    # Present as a watermark (not a red diagram): between 0.05% and 15%
    has_red = 0.0005 < red_ratio < 0.15

    # ── Blue detection (LAB b* channel) ──────────────────────────────────
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L_chan = lab[:, :, 0].astype(np.float32)  # [0, 255]
    b_chan = lab[:, :, 2].astype(np.float32)  # [0, 255], 128 = neutral

    # Only examine bright pixels: dark pixels are diagram content, not watermark
    bright_mask = L_chan > 100
    bright_count = int(np.count_nonzero(bright_mask))

    if bright_count > 0:
        b_bright = b_chan[bright_mask]
        # b < 116 => standard b* < -12 => noticeably blue-shifted
        blue_fraction = float(np.sum(b_bright < 116)) / bright_count
        # b < 124 => mild cool cast in bright regions
        cool_fraction = float(np.sum(b_bright < 124)) / bright_count
        mean_b_shift = float(np.mean(128.0 - b_bright))  # positive = bluer
    else:
        blue_fraction = 0.0
        cool_fraction = 0.0
        mean_b_shift = 0.0

    # LAB blue: at least 6% of bright pixels blue-shifted, with meaningful mean shift
    has_blue_lab = blue_fraction > 0.06 and mean_b_shift > 3.0

    # ── Saturated blue detection (HSV hue) ───────────────────────────────
    # Strict mask for opaque/saturated blue logo shapes.
    # - H in [95,130] focuses on blue core (avoids broad cyan/light-blue tints)
    # - S >= 90 avoids low-saturation page casts
    # - V <= 220 avoids near-white sky-blue regions
    blue_hsv_mask = cv2.inRange(
        hsv,
        np.array([95, 90, 20]),
        np.array([130, 255, 220]),
    )
    blue_hsv_ratio = float(np.count_nonzero(blue_hsv_mask)) / total_pixels
    # Present as a watermark: between 2% (not tiny noise blobs) and 45%
    has_blue_saturated = 0.02 < blue_hsv_ratio < 0.45

    # Mild cool cast fallback (for very light blue-tinted watermark backgrounds)
    has_cool_cast = cool_fraction > 0.30 and mean_b_shift > 1.0

    has_blue = has_blue_lab or has_blue_saturated

    # ── Grey detection (luminance histogram) ──────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (total_pixels + 1e-6)
    # Grey watermarks create mass in mid-tone range
    grey_mass = float(np.sum(hist_norm[80:200]))
    # Grey only when not already flagged strong blue.
    # Also route mild cool-cast pages here so they don't get "none" passthrough.
    has_grey = (grey_mass > 0.05 or has_cool_cast) and not has_blue

    # ── Classify ──────────────────────────────────────────────────────────
    if has_red and (has_blue or has_grey):
        wm_type = "mixed"
    elif has_blue_saturated and not has_blue_lab:
        wm_type = "blue_sat"  # opaque saturated blue logo (HSV path)
    elif has_blue:
        wm_type = "blue"  # semi-transparent light-blue cast (LAB path)
    elif has_red:
        wm_type = "red"
    elif has_grey:
        wm_type = "grey"
    else:
        wm_type = "none"

    return ImageAnalysis(
        wm_type=wm_type,
        has_red=has_red,
        has_blue=has_blue,
        has_blue_lab=has_blue_lab,
        has_blue_saturated=has_blue_saturated,
        has_grey=has_grey,
        red_ratio=red_ratio,
        blue_fraction=blue_fraction,
        blue_hsv_ratio=blue_hsv_ratio,
        cool_fraction=cool_fraction,
        grey_mass=grey_mass,
        mean_b_shift=mean_b_shift,
    )


# ── Red watermark removal ─────────────────────────────────────────────────────


def _remove_red_pixels(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove red/maroon watermark pixels via HSV color masking.

    Returns (cleaned_bgr, dilated_mask).  The dilated mask is kept and applied
    AFTER grayscale processing to force those regions back to white, preventing
    the LUT and sharpening steps from creating dark ring artefacts around
    formerly-red letters.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    # Also catch dark maroon/brown-red variations
    mask3 = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([15, 255, 200]))
    red_mask = mask1 | mask2 | mask3

    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(red_mask, dilate_kernel, iterations=3)

    result = img_bgr.copy()
    result[dilated_mask > 0] = [255, 255, 255]
    return result, dilated_mask


# ── Saturated blue watermark removal — HSV masking ──────────────────────────


def _remove_blue_pixels(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Remove opaque/saturated blue-cyan watermark pixels via HSV hue masking.

    This handles watermarks that are dark, fully-saturated blue or cyan logos
    (e.g., a navy book/torch graphic) which have low L values in LAB and are
    therefore invisible to the LAB-based confidence map.  The approach mirrors
    _remove_red_pixels exactly, just targeting the blue/cyan hue band.

    HSV hue in OpenCV (0–180):
        Cyan  : H ≈  85–95
        Blue  : H ≈  95–130
    We use H=[95,130], S>=90, V<=220 to isolate saturated blue logos while
    avoiding broad low-saturation light-blue page backgrounds.

    Returns (cleaned_bgr, dilated_mask).  The dilated mask is re-applied AFTER
    the enhancement pipeline to force those regions to pure white, preventing
    sharpening from creating dark halo rings around formerly-blue logo shapes.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(
        hsv,
        np.array([95, 90, 20]),
        np.array([130, 255, 220]),
    )

    dilate_kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(blue_mask, dilate_kernel, iterations=3)

    result = img_bgr.copy()
    result[dilated_mask > 0] = [255, 255, 255]
    return result, dilated_mask


# ── Blue watermark removal — LAB confidence-map method ───────────────────────


def _remove_blue_watermark_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Remove blue watermarks using LAB colorspace confidence mapping.

    Core concept
    ------------
    In CIE L*a*b* (OpenCV encoding: L in [0,255], b in [0,255] with 128=neutral):

        Blue watermark pixel  : L > 120 (bright),   b < 116 (b* < -12, blue)
        Black diagram content : L < 80  (dark),      b ≈ 128 (neutral)
        White background      : L > 220 (very bright), b ≈ 128 (neutral)

    The mask (L > 100) AND (b < 116) is almost exclusively watermark pixels.
    We turn this observation into a smooth confidence score (0..1) rather than
    a hard threshold, so the transition from watermark to content is gradual and
    produces no visible boundary artefacts.

    Confidence formula
    ------------------
        b_shift          = 128 - b_channel          # positive => bluer
        blue_excess      = max(0, b_shift - 8)       # ignore scan noise (< 8 units)
        lightness_weight = clip((L - 60) / 80, 0, 1) # gates dark pixels → 0
        confidence       = clip(blue_excess / 35, 0, 1) * lightness_weight

    Output correction
    -----------------
        corrected_L = L + confidence * (255 - L) * 1.2

    At confidence=1 this pushes the pixel cleanly to 255 (white).
    At confidence=0 (dark pixels) the L value is unchanged.
    The amplifier 1.2 ensures full suppression even at slightly sub-1 confidence.

    Returns grayscale uint8 (from the corrected L channel).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L_f = lab[:, :, 0].astype(np.float32)  # [0, 255]
    b_f = lab[:, :, 2].astype(np.float32)  # [0, 255], 128 = neutral

    b_shift = 128.0 - b_f  # positive = blue-shifted
    blue_excess = np.maximum(0.0, b_shift - 8.0)  # noise floor at 8 units

    # Protect dark diagram content: ramp 0 → 1 over L=[60, 140]
    lightness_weight = np.clip((L_f - 60.0) / 80.0, 0.0, 1.0)

    # Confidence reaches 1.0 at b_shift ≥ 43 (standard b* ≈ -33, strong blue)
    confidence = np.clip(blue_excess / 35.0, 0.0, 1.0) * lightness_weight

    # Push watermark pixels toward white; amplifier 1.2 ensures full coverage
    corrected_L = L_f + confidence * (255.0 - L_f) * 1.2
    corrected_L = np.clip(corrected_L, 0.0, 255.0)

    return corrected_L.astype(np.uint8)


# ── Grey watermark removal — Otsu-blended adaptive LUT ───────────────────────


def _remove_grey_watermark_adaptive(img_bgr: np.ndarray) -> np.ndarray:
    """Remove grey watermarks using an Otsu-blended adaptive threshold LUT.

    v1 limitation
    -------------
    The fixed threshold range [160, 210] is a single heuristic.  When the
    watermark is lighter (threshold needs to be higher) or the diagram content
    sits at a higher luminance level (threshold needs to be lower), v1 either
    leaves visible residue or clips diagram detail.

    v2 fix
    ------
    Two per-image anchors are computed and blended:

        otsu_val  — Otsu's method finds the optimal binary split between two
                    histogram modes (dark text / light background+watermark).
                    This is the principled upper bound on the threshold.

        p75       — 75th-percentile brightness.  Watermark pixels are typically
                    in the upper quartile of brightness, so this is a
                    conservative lower bound on where watermark starts.

        threshold = clip((otsu_val + p75) / 2, 140, 230)

    The clamp [140, 230] is a safety rail: below 140 would erase diagram
    content; above 230 would leave watermark residue in nearly all images.

    LUT shape
    ---------
    A smooth ease-in-out ramp (Hermite: 3t² - 2t³) over a 50-unit transition
    zone avoids the sharp discontinuity that can create a visible "waterline".
    Above the threshold: white (255).
    In the transition zone: ramp from dark-value to white.
    Below: mild contrast boost (×0.85 avoids harsh black-clamping of greys).

    Returns grayscale uint8.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p75 = float(np.percentile(gray, 75))

    raw_threshold = (float(otsu_val) + p75) / 2.0
    threshold = float(np.clip(raw_threshold, 140, 230))
    transition_width = 50.0

    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        fi = float(i)
        if fi >= threshold:
            lut[i] = 255
        elif fi >= threshold - transition_width:
            # Hermite ease-in-out: smooth S-curve, no derivative discontinuity
            t = (fi - (threshold - transition_width)) / transition_width
            t_smooth = t * t * (3.0 - 2.0 * t)
            dark_val = max(0, int(fi * 0.85))
            lut[i] = int(dark_val * (1.0 - t_smooth) + 255.0 * t_smooth)
        else:
            # Dark content: mild contrast lift only
            lut[i] = max(0, int(fi * 0.85))

    return cv2.LUT(gray, lut)


# ── Output enhancement pipeline ───────────────────────────────────────────────


def _text_guided_sharpen(gray: np.ndarray) -> np.ndarray:
    """Sharpen mainly text/line strokes while leaving background calmer.

    Strategy:
    - Build a dark-stroke mask via Otsu inverse threshold (text/diagram lines).
    - Slightly dilate mask so anti-aliased stroke edges are included.
    - Apply unsharp mask globally, but blend it only on masked stroke regions.

    This avoids the common problem of sharpening bright background artefacts.
    """
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.3)
    unsharp = cv2.addWeighted(gray, 1.85, blurred, -0.85, 0)
    unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)

    _, stroke_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    stroke_mask = cv2.dilate(stroke_mask, np.ones((2, 2), np.uint8), iterations=1)

    out = gray.copy()
    out[stroke_mask > 0] = unsharp[stroke_mask > 0]
    return out


def _remove_tiny_light_residue(gray: np.ndarray) -> np.ndarray:
    """Remove tiny isolated light-grey residue left by watermark suppression.

    Targets small, disconnected, light components that are not near strong text
    edges. This is designed to remove tiny watermark fragments (dots/islands)
    without touching actual dark content.
    """
    # Candidate residue: light but not pure-white
    candidate = ((gray >= 170) & (gray <= 245)).astype(np.uint8) * 255

    # Protect text/diagram neighborhoods from cleanup.
    edges = cv2.Canny(gray, 70, 150)
    edge_protect = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    candidate[edge_protect > 0] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )

    out = gray.copy()
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        if area < 2 or area > 90:
            continue
        if max(width, height) > 30:
            continue

        comp = labels == label
        mean_val = float(np.mean(gray[comp]))
        if mean_val < 182:
            continue

        out[comp] = 255

    return out


def _enhance_output(gray: np.ndarray, is_clean: bool = False) -> np.ndarray:
    """Post-processing enhancement pipeline.

    For watermarked images (is_clean=False)
    ----------------------------------------
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
       clipLimit=2.5, tileGridSize=(8,8)
       Unlike a global LUT, CLAHE processes the image in 8×8 pixel tiles and
       normalizes contrast locally.  This handles uneven scan illumination
       (darker corners, brighter center) that a global curve cannot fix.
       clipLimit prevents over-amplification of noise in uniform flat regions.

    2. Bilateral filter — edge-preserving denoising before sharpening
       d=5, sigmaColor=30, sigmaSpace=30
       Gaussian blur (used naively before unsharp mask) smooths edges, which
       hurts thin diagram lines.  The bilateral filter weights each neighbour
       by both spatial distance AND colorimetric similarity — pixels across a
       strong edge (black line on white) have very different values, so they
       contribute almost nothing to the blend.  Result: noise is removed but
       diagram lines and text edges are preserved.

     3. Text-guided unsharp mask
         Apply stronger sharpening only on dark stroke regions (text/lines), not
         on bright background. This improves readability while avoiding background
         artefacts.

    4. Background lift
       Push pixels above max(p98 × 0.90, 220) to pure white 255.
       Eliminates residual grey wash from scan background without touching
       diagram content (which should be under 200 after the above steps).

     5. Tiny-residue cleanup
         Remove very small disconnected light-grey islands (common leftover from
         watermark suppression), while protecting edges near real content.

    For clean images (is_clean=True)
    ---------------------------------
    Light CLAHE only (clipLimit=1.5) — normalize contrast gently without
    any sharpening that might introduce artefacts on already-clean content.

    Returns enhanced grayscale uint8.
    """
    if is_clean:
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    # Step 1: CLAHE local contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 2: Bilateral filter — edge-preserving smoothing
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=30, sigmaSpace=30)

    # Step 3: Sharpen text/diagram strokes, keep background stable
    sharpened = _text_guided_sharpen(enhanced)

    # Step 4: Background lift — force near-white pixels to pure white
    p98 = float(np.percentile(sharpened, 98))
    bg_threshold = max(p98 * 0.90, 220.0)
    sharpened[sharpened > bg_threshold] = 255

    # Step 5: Clean tiny light residue from watermark remnants
    sharpened = _remove_tiny_light_residue(sharpened)
    sharpened[sharpened > 245] = 255

    return sharpened


# ── Main orchestrator ─────────────────────────────────────────────────────────


def remove_watermark_v2(
    img_path: str,
    verbose: bool = False,
) -> Optional[bytes]:
    """Remove watermark from a scanned question paper image.

    Full pipeline
    -------------
    1. Read image and classify watermark type via analyze_image().
    2. If clean (type=none): apply minimal CLAHE enhancement and return.
    3. If red or mixed: remove red pixels via HSV masking, save dilated mask.
    4. Route to the appropriate removal method:
         - has_blue  → LAB confidence-map method (_remove_blue_watermark_lab)
         - grey-only → adaptive Otsu LUT method (_remove_grey_watermark_adaptive)
    5. Apply enhancement pipeline (CLAHE → bilateral → unsharp → bg-lift).
    6. Re-apply dilated red mask → force formerly-red areas to white (prevents
       sharpening from creating dark ring artefacts around removed red text).

    Returns PNG-encoded bytes, or None on read failure.
    """
    if not HAS_CV2:
        print("ERROR: OpenCV not installed. Run: pip install opencv-python")
        return None

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  ERROR: Cannot read image: {img_path}")
        return None

    # ── Step 1: Classify ─────────────────────────────────────────────────
    analysis = analyze_image(img_bgr)

    if verbose:
        print(
            f"    type={analysis.wm_type:8s} "
            f"blue_frac={analysis.blue_fraction:.3f} "
            f"b_shift={analysis.mean_b_shift:+.1f} "
            f"blue_hsv={analysis.blue_hsv_ratio:.3f} "
            f"cool_frac={analysis.cool_fraction:.3f} "
            f"grey_mass={analysis.grey_mass:.3f} "
            f"red={analysis.red_ratio:.4f}"
        )

    # ── Step 2: Clean passthrough ─────────────────────────────────────────
    if analysis.wm_type == "none":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = _enhance_output(gray, is_clean=True)
        success, buffer = cv2.imencode(".png", enhanced)
        return buffer.tobytes() if success else None

    # ── Step 3: Colour watermark pixel removal (HSV masking) ───────────────
    red_mask = None
    blue_mask = None
    working_img = img_bgr

    if analysis.has_red:
        working_img, red_mask = _remove_red_pixels(working_img)

    if analysis.has_blue_saturated and not analysis.has_blue_lab:
        # Opaque/saturated blue logo: erase via HSV masking BEFORE grayscale.
        # This handles dark navy/teal graphics whose L < 100 makes them
        # invisible to the LAB confidence-map method.
        working_img, blue_mask = _remove_blue_pixels(working_img)

    # ── Step 4: Watermark removal for remaining chromatic cast ─────────────
    if analysis.has_blue_lab:
        # Semi-transparent light-blue tint: LAB b* confidence map
        gray = _remove_blue_watermark_lab(working_img)
    elif analysis.has_grey:
        # Grey watermark cast: adaptive Otsu LUT
        gray = _remove_grey_watermark_adaptive(working_img)
    else:
        # Saturated blue was already fully handled by HSV masking above;
        # just convert to greyscale for the enhancement pipeline.
        gray = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)

    # ── Step 5: Enhancement pipeline ──────────────────────────────────────
    enhanced = _enhance_output(gray, is_clean=False)

    # ── Step 6: Re-apply colour masks ─────────────────────────────────────
    # Sharpening can create dark halo rings around formerly-coloured regions.
    # Force them back to white to prevent ghost outlines.
    if red_mask is not None and red_mask.shape == enhanced.shape:
        enhanced[red_mask > 0] = 255
    if blue_mask is not None and blue_mask.shape == enhanced.shape:
        enhanced[blue_mask > 0] = 255

    success, buffer = cv2.imencode(".png", enhanced)
    return buffer.tobytes() if success else None


# ── Batch processing (CLI) ────────────────────────────────────────────────────


def process_directory(
    input_dir: str,
    output_dir: str,
    verbose: bool = False,
) -> None:
    """Process all images in input_dir, write cleaned PNGs to output_dir."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([f for f in input_path.iterdir() if f.suffix.lower() in exts])

    if not images:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(images)} images  ->  {output_path}")
    if verbose:
        print(f"  {'#':>5}  {'filename':<25}  classification + stats")
        print(f"  {'-'*5}  {'-'*25}  {'-'*50}")

    type_counts: dict = {}
    failed = 0
    t0 = time.perf_counter()

    for i, img_path in enumerate(images):
        if verbose:
            print(f"  [{i+1:>4}/{len(images)}] {img_path.name:<25}", end="  ")

        # Classify first so we can track type counts independently of processing
        img = cv2.imread(str(img_path))
        wm_type = analyze_image(img).wm_type if img is not None else "failed"
        type_counts[wm_type] = type_counts.get(wm_type, 0) + 1

        cleaned_bytes = remove_watermark_v2(str(img_path), verbose=verbose)
        out_file = output_path / img_path.with_suffix(".png").name

        if cleaned_bytes:
            with open(out_file, "wb") as f:
                f.write(cleaned_bytes)
        else:
            import shutil

            shutil.copy2(img_path, output_path / img_path.name)
            failed += 1
            print(f"  FAILED: {img_path.name} (copied original)")

        if not verbose and ((i + 1) % 20 == 0 or (i + 1) == len(images)):
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  Progress: {i+1}/{len(images)}  ({rate:.1f} img/s)")

    elapsed = time.perf_counter() - t0
    per_image_ms = elapsed / len(images) * 1000

    print(f"\n{'─'*55}")
    print(f"  Done in {elapsed:.1f}s  ({per_image_ms:.0f} ms/image)")
    print(f"  Watermark type counts: {type_counts}")
    if failed:
        print(f"  Failures (original copied): {failed}")
    print(f"  Output: {output_path}")
    print(f"{'─'*55}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Watermark removal v2 — LAB colorspace + adaptive Otsu method.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/remove_watermark_v2.py samples/watermarked output_v2/
  python src/remove_watermark_v2.py samples/watermarked output_v2/ --verbose
  python src/remove_watermark_v2.py samples/clean output_v2_clean/
        """,
    )
    parser.add_argument("input_dir", help="Directory containing input images.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output_v2",
        help="Output directory for cleaned PNG images (default: output_v2).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-image classification details.",
    )

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, verbose=args.verbose)
