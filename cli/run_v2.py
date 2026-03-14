"""CLI wrapper for remove_watermark_v2.py supporting single or batch input.

Examples:
  uv run cli/run_v2.py samples/watermarked output_v2
  uv run cli/run_v2.py samples/watermarked/wm_001.jpg single_out
  uv run cli/run_v2.py samples/watermarked/wm_001.jpg single_out --verbose
"""

from pathlib import Path
import argparse
import shutil
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remove_watermark_v2 import process_directory, remove_watermark_v2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_PRE_LUMA_LIFT = 235


def process_single_image(
    input_image: Path,
    output_dir: Path,
    verbose: bool = False,
    enable_multiscale_blue: bool = False,
    enable_red_inpaint: bool = False,
    pre_luma_lift: int = 0,
) -> int:
    """Process one image and write PNG output to output_dir.

    Returns shell-style status code (0=success, 1=failure).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_bytes = remove_watermark_v2(
        str(input_image),
        verbose=verbose,
        enable_multiscale_blue=enable_multiscale_blue,
        enable_red_inpaint=enable_red_inpaint,
        pre_luma_lift=pre_luma_lift,
    )

    out_file = output_dir / f"{input_image.stem}.png"
    if cleaned_bytes is not None:
        out_file.write_bytes(cleaned_bytes)
        print(f"Processed single image: {input_image.name} -> {out_file}")
        return 0

    fallback_file = output_dir / input_image.name
    shutil.copy2(input_image, fallback_file)
    print(f"FAILED: {input_image.name} (copied original to {fallback_file})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run watermark removal v2 on a single image file or directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input path can be either:
  - image file (.jpg/.jpeg/.png): runs single-image mode
  - directory: runs batch mode for all .jpg/.jpeg/.png files

Examples:
  uv run cli/run_v2.py samples/watermarked output_v2
  uv run cli/run_v2.py samples/watermarked/wm_001.jpg single_out
  uv run cli/run_v2.py samples/watermarked/wm_060.jpg final_output --enable-multiscale-blue --enable-red-inpaint --pre-luma-lift
  uv run cli/run_v2.py samples/watermarked/wm_060.jpg final_output --enable-multiscale-blue --enable-red-inpaint --pre-luma-lift 238
        """,
    )
    parser.add_argument("input_path", help="Input image path or directory path.")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="output_v2",
        help="Output directory (default: output_v2).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-image classification details.",
    )
    parser.add_argument(
        "--enable-multiscale-blue",
        action="store_true",
        help="Enable multi-scale LAB confidence fusion for faint large blue watermarks.",
    )
    parser.add_argument(
        "--enable-red-inpaint",
        action="store_true",
        help="Use Telea inpainting for red watermark regions instead of white fill.",
    )
    parser.add_argument(
        "--pre-luma-lift",
        nargs="?",
        type=int,
        const=DEFAULT_PRE_LUMA_LIFT,
        default=0,
        help=(
            "Near-white luminance lift threshold before CLAHE "
            "(220-250, 0=off). If used without a value, defaults to "
            f"{DEFAULT_PRE_LUMA_LIFT}."
        ),
    )

    args = parser.parse_args()

    if args.pre_luma_lift != 0 and not (220 <= args.pre_luma_lift <= 250):
        parser.error("--pre-luma-lift must be 0 or in the range 220-250")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            parser.error(
                f"Unsupported file type '{input_path.suffix}'. "
                f"Supported: {sorted(IMAGE_EXTENSIONS)}"
            )
        return process_single_image(
            input_image=input_path,
            output_dir=output_dir,
            verbose=args.verbose,
            enable_multiscale_blue=args.enable_multiscale_blue,
            enable_red_inpaint=args.enable_red_inpaint,
            pre_luma_lift=args.pre_luma_lift,
        )

    if input_path.is_dir():
        process_directory(
            str(input_path),
            str(output_dir),
            verbose=args.verbose,
            enable_multiscale_blue=args.enable_multiscale_blue,
            enable_red_inpaint=args.enable_red_inpaint,
            pre_luma_lift=args.pre_luma_lift,
        )
        return 0

    parser.error(f"Input path is neither file nor directory: {input_path}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
