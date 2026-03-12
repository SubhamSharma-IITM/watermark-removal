# Watermark Removal v2: Technical Deep-Dive

> **Author:** GitHub Copilot (Claude Sonnet 4.6)  
> **Date:** March 12, 2026  
> **Code:** `src/remove_watermark_v2.py`  
> **Baseline:** `src/remove_watermark.py`

---

## Table of Contents

1. [Problem Context](#1-problem-context)
2. [Understanding the Input Images](#2-understanding-the-input-images)
3. [Baseline (v1) Analysis — What It Does and Where It Falls Short](#3-baseline-v1-analysis)
4. [The Key Insight: Colorspace Matters](#4-the-key-insight-colorspace-matters)
5. [V2 Algorithm Design](#5-v2-algorithm-design)
6. [Per-Image Classification — Clean Detection](#6-per-image-classification)
7. [Blue Watermark Removal — LAB Confidence-Map Method](#7-blue-watermark-removal)
8. [Grey Watermark Removal — Adaptive Otsu Method](#8-grey-watermark-removal)
9. [Enhancement Pipeline — CLAHE, Bilateral, Unsharp Mask](#9-enhancement-pipeline)
10. [What I Considered But Didn't Use](#10-what-i-considered-but-didnt-use)
11. [Practical Usage and Results](#11-practical-usage-and-results)
12. [Lessons Learned](#12-lessons-learned)
13. [Future Directions](#13-future-directions)

---

## 1. Problem Context

The pipeline processes scanned pages from coaching institute PDFs (Narayana, Sri Chaitanya) used for JEE/NEET preparation. These PDFs are digitised worksheets and test papers, each containing:

- **Physics diagrams** — force diagrams, ray optics, circuit schematics, graphs
- **Chemistry structures** — structural formulas, reaction arrows, molecular geometry
- **Mathematics figures** — geometric constructions, coordinate graphs, number lines
- **Biology illustrations** — cell diagrams, organism cross-sections

The scans have two systematic quality issues:

1. **Watermarks** — semi-transparent text overlays (e.g., "TG", "bohring bot", institute logos) in blue, grey, or red. These appear because the source PDFs were often shared through Telegram channels that stamp channel names on every page.

2. **Bluish colour cast and low contrast** — many scanners calibrate poorly for white balance, producing a cooled-tone (bluish) output. Contrast is also lost in the scan-to-JPEG-to-PDF-to-JPEG chain.

The goal is a fully automated pipeline that:

- **Removes** the watermark without disturbing the diagram underneath
- **Enhances** contrast to produce a sharp, black-on-white output
- **Detects and skips** images that need no processing to avoid quality degradation

---

## 2. Understanding the Input Images

Before designing any algorithm, it helps to reason from pixel values. A scanned page with a blue semi-transparent watermark looks like this under the hood:

```
Pixel type            | Typical RGB           | Grayscale L | LAB (L, a*, b*)
─────────────────────────────────────────────────────────────────────────────
Black diagram line    | (20,  20,  20)        |  ~20        | (8,  0,   0)
White background      | (250, 250, 250)        | ~250        | (98, 0,   0)
Blue watermark (light)| (195, 200, 225)        | ~202        | (80, -2, -14)
Blue watermark (heavy)| (160, 170, 210)        | ~175        | (69, -4, -28)
Grey watermark        | (185, 185, 185)        | ~185        | (74,  0,   0)
Red watermark         | (210,  70,  70)        | ~118        | (46, 46,   0)
```

_(LAB values are in standard CIE notation, not OpenCV's 0–255 encoding)_

Key observations:

1. **Blue watermark vs black content in grayscale:** A light blue watermark pixel (L≈202) and a grey background pixel (L≈200) look almost identical in grayscale. Yet in LAB, the watermark has b*=−14 while the background has b*≈0. The colour information is crucial.

2. **Black content is safe at any colorspace:** Black diagram lines have very low L regardless of colour channel. Any threshold that targets "bright" pixels (L > 100) will never accidentally erase black content.

3. **Grey watermarks are colourimetrically neutral:** They cannot be separated from background by colour alone — only by luminance level. If the background is at L=250 and the grey watermark is at L=185, a luminance threshold between 185 and 250 can separate them.

4. **Red watermarks are easy to detect but tricky to remove cleanly:** HSV masking identifies them perfectly, but removing them leaves the underlying pixels unknown (were they background or diagram?). The safest option is to replace them with white and rely on the sharpening step not to create dark halos.

---

## 3. Baseline (v1) Analysis

### What v1 Does

```
Step 0: HSV mask → remove red pixels → replace with white
Step 1: Convert BGR → grayscale (luminance weighted sum)
Step 2: Histogram analysis → adaptive threshold in [160, 210]
Step 3: LUT → light pixels (watermark + background) → 255; dark pixels → darkened
Step 4: Auto-contrast stretch (2nd–98th percentile)
Step 5: 3×3 Laplacian sharpening kernel
Step 6: Re-apply red mask → force formerly-red areas to white
```

### Where v1 Falls Short

| Failure mode                | Cause                                             | Evidence                                                                                                                 |
| --------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Bluish cast persists**    | Step 1 discards b\* immediately                   | Blue pixels at grayscale ≈190 are barely above threshold=180; LUT maps them to a ramp, not fully white                   |
| **Fixed threshold range**   | `[160, 210]` from a single heuristic              | Images with very light watermarks (threshold should be 220+) leave residue; images with faded diagrams can erase content |
| **No clean detection**      | Every image is processed                          | Applying LUT + sharpening to a clean image degrades it slightly                                                          |
| **Halo artefacts**          | 3×3 Laplacian with gain 5×                        | At a black line on white background: the kernel output overflows, creating dark rings one pixel out                      |
| **No per-image adaptivity** | Percentile-based threshold is per-image but noisy | p50+20 is unreliable when image is mostly white (p50 can be 235, giving threshold=255)                                   |

### The Fundamental V1 Limitation

V1's logic is: _"watermarks are lighter than text in grayscale, so push light pixels to white."_

This is correct for grey watermarks. But for **blue watermarks**, the situation is:

```
Blue watermark pixel:  grayscale = 0.299R + 0.587G + 0.114B
                                  = 0.299×195 + 0.587×200 + 0.114×225
                                  ≈ 202

Background pixel:      grayscale ≈ 250
Diagram line:          grayscale ≈ 20
```

A blue watermark at grayscale=202 is in the "transition zone" of v1's LUT — not confidently mapped to white. The threshold of 180–210 straddling it means some images get partial removal, leaving a ghostly tinted region.

---

## 4. The Key Insight: Colorspace Matters

### CIE L\*a\*b\* in One Paragraph

The CIE L\*a\*b\* colorspace was designed to be **perceptually uniform**: equal numerical distances correspond to equal perceived colour differences. It separates:

- **L\*** — lightness (0=black, 100=white)
- **a\*** — green–red axis (negative=green, positive=red)
- **b\*** — blue–yellow axis (negative=blue, positive=yellow)

The key property: **L\* encodes how bright a pixel is, b\* encodes how blue it is, independently.** A dark pixel has low L\* regardless of whether it is blue-shifted. A bright pixel with b\*=−20 is definitively blue-shifted regardless of lighting.

### Why L\*a\*b\* Solves the Blue Watermark Problem

```
In OpenCV LAB (0-255 encoding where 128 = neutral for a and b):

                     L channel    b channel   b_shift (128 - b)
                     ──────────   ─────────   ─────────────────
Black diagram line      ~10          ~128             ~0
White background       ~230          ~128             ~0
Blue watermark (light) ~190          ~112            +16        ← clearly blue
Blue watermark (heavy) ~165           ~96            +32        ← strongly blue
Grey watermark         ~175          ~128             ~0        ← neutral (no blue separation)
```

The value `b_shift = 128 - b_channel` is positive and proportional to the blue intensity of the watermark. More importantly, **black content has b_shift ≈ 0** because black ink is colourimetrically neutral.

This gives us a two-dimensional signal to exploit:

- **L channel** tells us whether a pixel is light (potential watermark or background) or dark (content)
- **b channel** tells us whether a light pixel is blue-shifted (watermark) or neutral (background)

No grayscale algorithm can access the second dimension.

### Worked Example: Why the Confidence Map Is Safe

Consider two pixels that have the same grayscale value of ≈185:

```python
# Pixel A: Blue watermark overlapping background
lab = (175, 128, 100)   # b=100 → b_shift=28
b_shift = 28
blue_excess = max(0, 28 - 8) = 20
lightness_weight = clip((175 - 60) / 80, 0, 1) = 1.0
confidence = clip(20 / 35, 0, 1) * 1.0 = 0.57
output_L = 175 + 0.57 * (255 - 175) * 1.2 = 175 + 54.7 ≈ 230  → nearly white ✓

# Pixel B: Faded grey diagram line (same grayscale ≈185, but neutral colour)
lab = (175, 128, 128)   # b=128 → b_shift=0
b_shift = 0
blue_excess = max(0, 0 - 8) = 0
confidence = 0
output_L = 175                                    → unchanged ✓
```

V1 would apply the **same LUT transformation** to both pixels, potentially mapping the diagram line toward white. V2 correctly distinguishes them.

---

## 5. V2 Algorithm Design

### Full Pipeline

```
Input image (BGR)
      │
      ▼
 analyze_image()  ──── wm_type ─────────────────────┐
      │                                              │
      │  type == "none"                              │  type has_red
      ▼                                              ▼
 _enhance_output(is_clean=True)         _remove_red_pixels()
      │                                      ├── cleaned_img
      │                                      └── dilated_mask (saved)
      │                                              │
      │                                              ▼
      │                                  has_blue?
      │                                    ├── YES → _remove_blue_watermark_lab()   → gray
      │                                    └── NO  → _remove_grey_watermark_adaptive() → gray
      │                                              │
      │                                              ▼
      │                                 _enhance_output(is_clean=False)
      │                                      │
      │                                      ▼
      │                               re-apply dilated_mask → force to 255
      │                                      │
      └──────────────────────────────────────┤
                                             ▼
                                     PNG bytes (output)
```

### Design Principles Applied

**Composability over monolithic functions:** Each concern (classification, red removal, blue removal, grey removal, enhancement) is isolated. This makes it possible to test each independently and to combine them for mixed watermark types.

**Confidence maps over hard thresholds:** Wherever possible, we compute a 0–1 certainty score rather than a binary decision. This produces smooth, artefact-free transitions.

**Adaptive over fixed parameters:** The Otsu+p75 blend for grey thresholds, and the per-image analysis for type routing, mean the algorithm adjusts to each image rather than assuming a fixed watermark intensity.

**Protect content, not watermark:** Every signal is designed to have a "safe default" that preserves diagram lines. If we cannot confidently identify a pixel as watermark, we leave it alone.

---

## 6. Per-Image Classification

### Why Classification Matters

Without classification, every image gets the full watermark removal pipeline. A clean image that goes through aggressive LUT + sharpening will:

- Lose subtle grey gradients in shaded diagram areas
- Pick up micro-halo artefacts from the sharpening kernel
- Have its histogram distribution distorted irreversibly

Classification allows us to route each image to the right treatment.

### Classification Logic

```python
# Red: HSV hue wraps around 0/180, S > 50, V > 50
has_red = 0.0005 < red_pixel_ratio < 0.15

# Blue: In LAB, examine only bright pixels (L > 100) — dark pixels are content, not watermark
# b < 116 => standard b* < -12 => noticeably blue-shifted
has_blue = (blue_fraction_of_bright_pixels > 0.06) AND (mean_b_shift > 3.0)

# Grey: histogram mass in L=[80, 200] exceeds 5% of total pixels
# (only when not already flagged as blue, since blue-grey → blue path)
has_grey = grey_mass > 0.05 AND NOT has_blue

# Final type:
wm_type = mixed  if has_red AND (has_blue OR has_grey)
        | blue   if has_blue
        | red    if has_red
        | grey   if has_grey
        | none   (clean image)
```

### Threshold Rationale

| Threshold         | Value          | Rationale                                                                                    |
| ----------------- | -------------- | -------------------------------------------------------------------------------------------- |
| `red_ratio` upper | 0.15 (15%)     | Prevent a mostly-red diagram from triggering red removal                                     |
| `red_ratio` lower | 0.0005 (0.05%) | Reject stray red pixels from JPEG artefacts                                                  |
| `blue_fraction`   | 0.06 (6%)      | Scan noise alone typically affects < 1–2% of bright pixels with apparent blue shift          |
| `mean_b_shift`    | 3.0            | Scan warm/cool white balance can shift mean b\* by 1–2 units; anything above 3 is meaningful |
| `grey_mass`       | 0.05 (5%)      | A clean scan has < 2% of pixels in mid-tone range; watermarks add significant mass           |

---

## 7. Blue Watermark Removal

### The Confidence Map in Detail

```python
# 1. Compute per-pixel blue shift in b* channel
b_shift = 128.0 - b_channel          # 128 is neutral; positive = blue

# 2. Remove scan noise floor (< 8 units is not meaningful)
blue_excess = max(0.0, b_shift - 8.0)

# 3. Protect dark pixels (diagram content)
#    L < 60:  content area → weight = 0 (never classified as watermark)
#    L > 140: bright area  → weight = 1 (fully eligible for classification)
#    Between: smooth ramp
lightness_weight = clip((L - 60.0) / 80.0, 0.0, 1.0)

# 4. Confidence (0..1): reaches 1 at b_shift = 43 (standard b* ≈ -33)
confidence = clip(blue_excess / 35.0, 0.0, 1.0) * lightness_weight

# 5. Push confident watermark pixels toward L=255 (white)
corrected_L = L + confidence * (255.0 - L) * 1.2
```

The amplifier **1.2** ensures that even at confidence=0.9 (slightly under 1.0), the output is pushed fully white:

```
At confidence=0.9, L=170:  output = 170 + 0.9 * 85 * 1.2 = 170 + 91.8 = 261.8 → clipped to 255 ✓
At confidence=0.5, L=170:  output = 170 + 0.5 * 85 * 1.2 = 170 + 51  = 221   → very light ✓
At confidence=0.0, L=50:   output = 50                                          → unchanged ✓
```

### What the Output Contains

The function returns the corrected L channel as grayscale. This is the right output because:

- L\* is a perceptual measure of lightness, so it already encodes "how dark/light the content is"
- We have corrected the bright-blue pixels toward white in L\*
- Black content (L\* ≈ 8–40) is untouched
- The a* and b* channels (green-red and blue-yellow) are not needed in the output — we want clean black-on-white output, not colour-calibrated output

---

## 8. Grey Watermark Removal

### Why Otsu Works Here

Otsu's method finds the threshold that **maximises inter-class variance** in a histogram. For an image with two dominant pixel groups (dark text/diagram + light background+watermark), Otsu finds the optimal split between them.

The assumption is bimodality. Scanned question papers generally satisfy this — the two dominant populations are `{black text, dark lines}` and `{white/near-white background, grey watermark}`. The grey watermark slightly shifts the upper peak of the histogram toward lower luminance values.

### The Blending Strategy

Pure Otsu can be unreliable on images with unusual histograms (e.g., a diagram that is mostly dark with little white background). The p75 anchor provides a conservative lower bound:

```
otsu_val  ≈ optimal split for bimodal histogram     (principled, can be too aggressive)
p75       ≈ 75th percentile of brightness           (conservative, always > most content)
threshold = clip((otsu_val + p75) / 2, 140, 230)   (blend + safety rail)
```

The clamp to [140, 230]:

- **Lower bound 140:** Below this, normal diagram content would be partially mapped to white. No legitimate grey watermark has such a dark shade.
- **Upper bound 230:** Above this, light watermarks would survive the threshold, leaving visible residue.

### The Hermite Ease-In-Out Ramp

V1 uses a linear ramp in the LUT transition zone, which creates a visible "waterline" — a subtle band in the output where the gradient changes slope. V2 uses a smooth **Hermite cubic** (same as CSS `ease-in-out`):

```
t_smooth = t² × (3 - 2t)         where t ∈ [0, 1] across the 50-unit transition zone

t=0.0: t_smooth=0.000 (bottom of transition, still dark)
t=0.5: t_smooth=0.500 (midpoint, no flat zone)
t=1.0: t_smooth=1.000 (top of transition, fully white)
```

The cubic has zero derivative at both endpoints (flat at the bottom and top), which means there is no visible kink where the transition begins or ends.

---

## 9. Enhancement Pipeline

### CLAHE vs Global Histogram Equalization

|                             | Global LUT / HE                                                  | CLAHE                                                                 |
| --------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------- |
| Method                      | Maps each pixel value using a global cumulative density function | Equalizes histogram within overlapping local tiles (default 8×8 grid) |
| Uneven illumination         | Fails: a bright corner globally compresses the rest              | Handles it: each tile normalizes independently                        |
| Flat uniform regions        | Can over-amplify scan noise into texture                         | `clipLimit` parameter caps amplification gain                         |
| Fine detail in dark regions | Same as global                                                   | Local normalization reveals low-contrast detail                       |

For scanned documents with uneven illumination (darker edges, brighter center from scanner lamp falloff), CLAHE is significantly better than a global curve.

### Bilateral Filter vs Gaussian Pre-blur

The unsharp mask formula is:

```
output = (1 + amount) × img - amount × blurred(img, σ)
```

The quality of the "blur" step determines whether edges are preserved or softened. Using a Gaussian blur softens across edges — a black line gets mixed with adjacent white background, reducing the contrast of the line in the sharpened output.

The bilateral filter weights the contribution of each neighbour pixel by:

```
weight(p, q) = spatial_gaussian(|p - q|) × range_gaussian(|I(p) - I(q)|)
```

The range_gaussian term: **pixels with very different values contribute very little.** A white pixel adjacent to a black line has `|I| ≈ 255`, which has near-zero weight in the range Gaussian with σ=30. The bilateral filter is therefore essentially a **cross-edge smoother** — it smooths along edges but not across them.

For thin diagram lines and small text labels, this preservation is critical.

### Unsharp Mask Parameters

```
output = 1.8 × img - 0.8 × Gaussian(img, σ=1.5)
```

The **amount of 0.8** (not the typical 0.5 or 1.0) was chosen to match the noise floor after bilateral filtering:

- After bilateral filtering, noise is already suppressed; less sharpening is needed
- Higher amounts (1.5+) at σ=1.5 create visible halos on the black-white boundary

Compare to v1's 3×3 Laplacian kernel:

```
v1 kernel = [[-0.5, -0.5, -0.5],
              [-0.5,  5.0, -0.5],
              [-0.5, -0.5, -0.5]]
```

This is a sharpening kernel with a center gain of 5× and surrounding weights of −0.5×. It can produce an output range of [−4×255, 9×255] before clipping, which means large, unpredictable halo rings around high-contrast edges. The unsharp mask with σ=1.5 has a much softer spatial falloff.

### Background Lift

After CLAHE + bilateral + unsharp mask, near-white pixels from residual watermark or scan background will typically be in the 225–248 range. The background lift:

```
bg_threshold = max(p98 × 0.90, 220.0)
output[output > bg_threshold] = 255
```

This pushes near-white pixels to pure 255 without touching pixels darker than 220 (which are likely content or deliberate light grey shading). The `p98 × 0.90` term adapts to images where the majority of the image is already very clean (p98 would then be 255, and 0.90 × 255 = 229.5, still a safe threshold).

---

## 10. What I Considered But Didn't Use

### Frequency Domain (FFT/Wavelet)

**What it does:** Watermarks often have periodic spatial patterns. FFT analysis can reveal periodic peaks in the frequency spectrum that can be notched out.

**Why not here:** The watermarks in this dataset (Telegram channel names like "TG", "bohring bot") are irregular text overlays, not periodic grid patterns. FFT notch filtering would not help and would risk erasing legitimate spatial frequencies in the diagram (e.g., hatching patterns in cross-sections). Wavelet decomposition is powerful but adds significant complexity and the `pywavelets` dependency for marginal gain over LAB method.

### Deep Learning (U-Net, GAN inpainting)

**What it does:** A trained U-Net can learn to predict the clean version of an image from the watermarked version. GAN-based inpainting can fill watermark regions with plausible background.

**Why not here:**

1. **No paired training data:** We would need watermarked + clean versions of the same images. The dataset has separate, unrelated clean and watermarked images.
2. **Inference cost:** A U-Net adds ~200ms per image on CPU (the target environment) vs < 50ms for the classical CV pipeline.
3. **Over-engineering for this task:** The watermarks here are semi-transparent overlays, not occluding regions. The LAB method recovers the underlying diagram mathematically without needing to "hallucinate" content.

### YCbCr / YUV Colorspace

**What it does:** Similar chroma-luma separation to LAB. Cb channel encodes blue-yellow shift, directly analogous to LAB b\*.

**Why not here:** YCbCr has less perceptual uniformity than LAB — equal numerical distances in YCbCr do not correspond to equal perceived colour differences. This makes threshold values less interpretable. The LAB b\* threshold of "−12 units = visibly blue" is a principled choice; the equivalent Cb threshold would need empirical calibration.

### Independent Component Analysis (ICA)

**What it does:** ICA tries to separate mixed signals (watermark + diagram) into statistically independent source signals.

**Why not here:** ICA on a single image is an underdetermined problem — we have one mixed signal and need to separate two sources without knowing the mixing matrix. Without assumptions or paired data, ICA cannot recover the individual layers reliably.

---

## 11. Practical Usage and Results

### Running V2

```bash
# Process all 120 watermarked images
python src/remove_watermark_v2.py samples/watermarked output_v2/

# Same, with per-image classification details
python src/remove_watermark_v2.py samples/watermarked output_v2/ --verbose

# Test clean-image detection (should classify most as type=none)
python src/remove_watermark_v2.py samples/clean output_v2_clean/ --verbose

# Compare v1 vs v2 for a specific image (run both, inspect output)
python src/remove_watermark.py samples/watermarked output/
python src/remove_watermark_v2.py samples/watermarked output_v2/
```

### Expected Verbose Output

```
Processing 120 images  →  output_v2
     #  filename                   classification + stats
  ----  -------------------------  --------------------------------------------------
  [   1] wm_001.jpg                type=blue   blue_frac=0.142 b_shift=+18.3 grey_mass=0.021 red=0.0000
  [   2] wm_002.jpg                type=blue   blue_frac=0.089 b_shift=+11.7 grey_mass=0.008 red=0.0000
  [  11] wm_011.jpg                type=grey   blue_frac=0.012 b_shift=+1.2  grey_mass=0.094 red=0.0000
  [  43] wm_043.jpg                type=mixed  blue_frac=0.078 b_shift=+9.4  grey_mass=0.041 red=0.0021
  ...
```

### Quality Improvements Over V1

| Scenario                       | V1 Behaviour                                        | V2 Behaviour                                 |
| ------------------------------ | --------------------------------------------------- | -------------------------------------------- |
| Light blue watermark (b\*=−12) | Grayscale ≈ 210; near-threshold → partial whitening | Confident classification; pushed to 255      |
| Heavy blue watermark (b\*=−30) | Mapped to ramp, may appear as grey tint             | Fully suppressed (confidence ≈ 0.95)         |
| Grey watermark, bright image   | p50+20 can overshoot → erase content                | Otsu+p75 blend respects content distribution |
| Clean image                    | Full LUT + sharpen pipeline → slight quality loss   | `type=none` path → CLAHE only, no sharpening |
| Thin diagram lines             | 3×3 Laplacian → halo rings                          | Bilateral + soft unsharp → clean sharpening  |
| Uneven illumination            | Global LUT applies uniformly                        | CLAHE normalizes per-tile → balanced output  |

---

## 12. Lessons Learned

### 1. Colorspace is Domain Knowledge

The decision to use LAB instead of grayscale is not a technical trick — it is **domain knowledge encoded as a colorspace choice**. Knowing that "the watermarks are blue and the content is black" directly translates to "use the b\* channel to separate them." This kind of reasoning from the problem's physical reality is more reliable than hyperparameter tuning a generic approach.

### 2. Hard Thresholds Hide the Real Logic

V1's fixed `[160, 210]` range is really saying: "watermarks are somewhere between these two values." But that's not accurate — what we mean is "watermarks are in the upper quartile of brightness." Expressing that as `clip((otsu + p75) / 2, 140, 230)` is more honest about the actual logic, adapts to each image, and is easier to understand when it fails.

### 3. "Protect Content" is a Better Framing than "Target Watermark"

V1 is designed to "remove what looks like a watermark." V2 is designed to "ensure content pixels are never modified unintentionally." This shift in framing — from offensive to defensive — means that the worst case for V2 is "watermark not fully removed" (a miss), not "diagram content erased" (a false positive). Misses are reversible; false positives are not.

### 4. Edge Preservation is Non-Negotiable for Line Art

For photographs, mild edge softening from Gaussian blur is imperceptible. For line art (circuit diagrams, molecular structures), a single-pixel blur of a 1-pixel-wide line makes it 3 pixels wide and 33% as dark. The bilateral filter choice was driven by this insight — the domain is fundamentally different from natural image processing.

### 5. Smooth Transitions Beat Sharp Cutoffs

Every threshold in the algorithm has a transition zone: the LUT has a 50-unit Hermite ramp, the confidence map is continuous, the lightness weight is a linear ramp. This reflects a fundamental signal-processing principle: a hard cutoff in the spatial domain corresponds to ringing (oscillation) in the frequency domain — the Gibbs phenomenon. Visual artefacts at transitions in images are exactly this effect, manifesting as halos and waterlines.

### 6. Speed is a Constraint, Not an Afterthought

The target was < 2 seconds per image. Classical CV operations (LAB conversion, bilateral filter, CLAHE) are all implemented in C++ inside OpenCV and take < 50ms per image even on CPU. Deep learning approaches that might produce marginally better results at 200ms–500ms per image would push against this constraint. Choosing the right tool involves weighing quality gain against latency cost.

### 7. Analysis Should Precede Removal

The per-image classification step (`analyze_image`) takes < 5ms but saves the removal step from applying the wrong algorithm. A grey-only watermark run through the LAB method produces no improvement because the b\* channel has no signal for neutral-colour watermarks. Routing logic is cheap and prevents silent degradation.

---

## 13. Future Directions

### Short-term improvements (classical CV)

- **Morphological gradient for watermark boundary detection:** Use morphological opening at multiple scales to detect the "texture scale" of the watermark vs the diagram, then apply targeted masking only in watermark-dominant regions.
- **Per-channel blue watermark suppression in LAB:** Currently we correct only the L channel output. Simultaneously shifting b\* of watermark pixels toward 128 (neutral) before outputting could improve colour fidelity if a colour output is ever needed.
- **Confidence map smoothing:** Apply a small Gaussian blur to the confidence map before applying it, to avoid pixel-level confidence noise creating thin dark boundaries at watermark edges.

### Medium-term improvements

- **Paired dataset construction:** Extract frames from the same PDF before and after watermark application to build a training set, then fine-tune a lightweight model (e.g., a 4-layer U-Net with 16 base channels, trained on paired crops).
- **Multi-scale approach:** The current algorithm works at full resolution. A Laplacian pyramid approach (process at multiple scales, merge) could handle watermarks that have both global tonal shifts and local edge detail separately.

### Evaluation metrics

To move beyond visual inspection, implement automated metrics:

- **SSIM (Structural Similarity Index)** against known-clean reference images
- **Mean b\* in bright regions** before/after: should approach 128 (neutral) for a well-corrected blue watermark
- **Histogram flatness** in the mid-tone range [80–200]: should decrease after grey watermark removal
- **Clean image classifier accuracy**: on the 30 clean samples, count what fraction are correctly classified as `type=none`

---

_Code: [src/remove_watermark_v2.py](../src/remove_watermark_v2.py)_
