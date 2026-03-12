"""Side-by-side comparison of v1 vs v2 watermark removal.

Picks N random samples from samples/watermarked/ and samples/clean/, runs both
v1 and v2 on each, and saves a side-by-side strip image:

    [ Original ] | [ V1 Output ] | [ V2 Output ]

with a labelled header showing the filename and v2 type classification.

Usage:
    uv run src/compare_samples.py
    uv run src/compare_samples.py --n 10 --seed 42 --output comparison/
    uv run src/compare_samples.py --n 5 --watermarked-only
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# -- import v1 and v2 from sibling src/ files --------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from remove_watermark import remove_watermark as v1_remove  # noqa: E402
from remove_watermark_v2 import remove_watermark_v2, analyze_image  # noqa: E402


# ── Comparison strip builder ─────────────────────────────────────────────────

_STRIP_HEIGHT = 600  # each panel is resized to this height
_LABEL_BAR = 36  # pixels reserved for the text label below each panel
_HEADER_H = 48  # pixels for the top filename/type banner
_PANEL_GAP = 6  # white gap between panels
_BG_COLOR = (240, 240, 240)  # light grey background


def _resize_to_height(img: np.ndarray, h: int) -> np.ndarray:
    """Resize image to exact height, preserving aspect ratio."""
    src_h, src_w = img.shape[:2]
    scale = h / src_h
    new_w = max(1, int(src_w * scale))
    return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)


def _add_label(panel: np.ndarray, text: str, color=(50, 50, 50)) -> np.ndarray:
    """Append a text label bar below a panel image."""
    h, w = panel.shape[:2]
    bar = np.full(((_LABEL_BAR), w, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar,
        text,
        (8, _LABEL_BAR - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        1,
        cv2.LINE_AA,
    )
    return np.vstack([panel, bar])


def _bytes_to_bgr(data: bytes | None, fallback_path: str) -> np.ndarray:
    """Decode PNG bytes to BGR array; falls back to reading original on failure."""
    if data is not None:
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    return cv2.imread(fallback_path)


def build_comparison_strip(img_path: str, wm_type: str) -> np.ndarray:
    """
    Build a three-panel comparison strip for one image:
        [ Original (color) ] | [ V1 output (grey) ] | [ V2 output (grey) ]

    The V1 and V2 grey outputs are rendered as single-channel greyscale promoted
    to 3-channel BGR so all three panels can be stacked horizontally.
    """
    original_bgr = cv2.imread(img_path)
    if original_bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")

    v1_bytes = v1_remove(img_path)
    v2_bytes = remove_watermark_v2(img_path, verbose=False)

    v1_bgr = _bytes_to_bgr(v1_bytes, img_path)
    v2_bgr = _bytes_to_bgr(v2_bytes, img_path)

    # Ensure all panels are 3-channel BGR (grey outputs come back as 1-channel
    # or 3-channel depending on imencode path)
    for arr in (v1_bgr, v2_bgr):
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    if v1_bgr.ndim == 2:
        v1_bgr = cv2.cvtColor(v1_bgr, cv2.COLOR_GRAY2BGR)
    if v2_bgr.ndim == 2:
        v2_bgr = cv2.cvtColor(v2_bgr, cv2.COLOR_GRAY2BGR)

    # Resize all panels to the same height
    orig_panel = _resize_to_height(original_bgr, _STRIP_HEIGHT)
    v1_panel = _resize_to_height(v1_bgr, _STRIP_HEIGHT)
    v2_panel = _resize_to_height(v2_bgr, _STRIP_HEIGHT)

    # Add label bars
    orig_panel = _add_label(orig_panel, "Original", color=(30, 30, 30))
    v1_panel = _add_label(v1_panel, "V1 (LUT)", color=(140, 60, 0))
    v2_panel = _add_label(v2_panel, f"V2 (LAB/Otsu) [{wm_type}]", color=(0, 110, 50))

    # Uniform width for clean horizontal stack (pad narrower panels on right)
    target_w = max(orig_panel.shape[1], v1_panel.shape[1], v2_panel.shape[1])

    def _pad_to_width(p, w):
        if p.shape[1] == w:
            return p
        pad = np.full((p.shape[0], w - p.shape[1], 3), 255, dtype=np.uint8)
        return np.hstack([p, pad])

    orig_panel = _pad_to_width(orig_panel, target_w)
    v1_panel = _pad_to_width(v1_panel, target_w)
    v2_panel = _pad_to_width(v2_panel, target_w)

    # Horizontal gap separator
    gap = np.full((orig_panel.shape[0], _PANEL_GAP, 3), _BG_COLOR, dtype=np.uint8)

    strip = np.hstack([orig_panel, gap, v1_panel, gap, v2_panel])

    # Header banner: filename + wm_type on grey background
    strip_w = strip.shape[1]
    header = np.full((_HEADER_H, strip_w, 3), (60, 60, 60), dtype=np.uint8)
    filename = Path(img_path).name
    label_text = f"{filename}  |  v2 type: {wm_type}"
    cv2.putText(
        header,
        label_text,
        (12, _HEADER_H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    return np.vstack([header, strip])


# ── Main ─────────────────────────────────────────────────────────────────────


def run_comparison(
    watermarked_dir: str,
    clean_dir: str,
    output_dir: str,
    n: int,
    seed: int,
    watermarked_only: bool,
    clean_only: bool,
) -> None:
    rng = random.Random(seed)
    out = Path(output_dir)

    wm_out = out / "watermarked"
    clean_out = out / "clean"

    exts = {".jpg", ".jpeg", ".png"}

    def _sample(directory: str, count: int) -> list[Path]:
        d = Path(directory)
        files = [f for f in sorted(d.iterdir()) if f.suffix.lower() in exts]
        if not files:
            print(f"No images found in {directory}")
            return []
        k = min(count, len(files))
        return rng.sample(files, k)

    tasks: list[tuple[Path, Path, str]] = []  # (img_path, output_dir, category)

    if not clean_only:
        wm_out.mkdir(parents=True, exist_ok=True)
        for p in _sample(watermarked_dir, n):
            tasks.append((p, wm_out, "watermarked"))

    if not watermarked_only:
        clean_out.mkdir(parents=True, exist_ok=True)
        for p in _sample(clean_dir, n):
            tasks.append((p, clean_out, "clean"))

    if not tasks:
        print("Nothing to process.")
        return

    print(f"Generating {len(tasks)} comparison strips -> {out}/")
    print(f"  Each strip: [ Original ] | [ V1 (LUT) ] | [ V2 (LAB/Otsu) ]")
    print()

    for i, (img_path, dest_dir, category) in enumerate(tasks, 1):
        # Pre-classify for display (same logic as v2 internals)
        img_bgr = cv2.imread(str(img_path))
        analysis = analyze_image(img_bgr) if img_bgr is not None else None
        wm_type = analysis.wm_type if analysis else "?"

        print(f"  [{i:>2}/{len(tasks)}] {img_path.name:<30} type={wm_type}", end="  ")

        try:
            strip = build_comparison_strip(str(img_path), wm_type)
            out_name = img_path.stem + "_comparison.png"
            out_file = dest_dir / out_name
            cv2.imwrite(str(out_file), strip)
            print(f"-> {out_file.relative_to(out.parent)}")
        except Exception as exc:
            print(f"FAILED: {exc}")

    print()
    print(f"Done. Open the '{out.name}/' folder to view the comparisons.")
    print(f"  watermarked/ : {len([t for t in tasks if t[2]=='watermarked'])} images")
    if not watermarked_only:
        print(f"  clean/       : {len([t for t in tasks if t[2]=='clean'])} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare v1 vs v2 watermark removal side-by-side.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run src/compare_samples.py
  uv run src/compare_samples.py --n 5 --seed 99
  uv run src/compare_samples.py --n 15 --output my_comparison/
  uv run src/compare_samples.py --watermarked-only --n 20
        """,
    )
    parser.add_argument(
        "--watermarked-dir",
        default="samples/watermarked",
        help="Directory with watermarked images (default: samples/watermarked)",
    )
    parser.add_argument(
        "--clean-dir",
        default="samples/clean",
        help="Directory with clean images (default: samples/clean)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="comparison",
        help="Output directory (default: comparison/)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of random samples per category (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--watermarked-only",
        action="store_true",
        help="Only process watermarked samples",
    )
    parser.add_argument(
        "--clean-only", action="store_true", help="Only process clean samples"
    )

    args = parser.parse_args()
    run_comparison(
        watermarked_dir=args.watermarked_dir,
        clean_dir=args.clean_dir,
        output_dir=args.output,
        n=args.n,
        seed=args.seed,
        watermarked_only=args.watermarked_only,
        clean_only=args.clean_only,
    )
