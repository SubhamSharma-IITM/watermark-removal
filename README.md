# Watermark Removal Solution

##  Author

Aman Jaiswal

---

##  Problem Statement

The task was to remove watermarks from scanned question paper images while preserving important content such as diagrams, text, and fine details.

Challenges included:

* Semi-transparent watermarks (blue/red)
* Overlapping diagrams and text
* Low contrast between watermark and background
* Avoiding damage to original content

---

##  Approach

### 🔍 Initial Approach (What I Tried)

Initially, I explored:

* Histogram-based thresholding
* LAB color space transformations
* Background normalization (Gaussian blur)
* CLAHE and adaptive contrast enhancement

###  Problem with Initial Methods

These approaches modified the entire image, which led to:

* Loss of diagram details
* Over-enhancement
* Inconsistent results across different image types

---

##  Final Approach (Region-Based Removal)

I shifted to a **region-based approach**, which focuses only on watermark pixels instead of modifying the full image.

### Step 1: Image Classification

Images are classified into:

* White background (question papers)
* Colorful images (posters)
* Dark background (skipped)

---

### Step 2: Watermark Detection (Mask Creation)

* **Red watermark detection** using HSV color space

* **Blue watermark detection** using LAB color space:

  * Only light-blue pixels are selected
  * Dark-blue diagram elements are preserved

* Morphological operations are applied to refine the mask

---

### Step 3: Watermark Removal (Core Step)

* OpenCV **inpainting (Telea algorithm)** is used
* Removes watermark region and reconstructs background naturally

---

### Step 4: Post-processing

* Convert to grayscale
* Apply CLAHE for contrast improvement (only when needed)
* Light sharpening to enhance readability

---

##  Key Advantages

* ✅ Preserves diagrams and text
* ✅ Removes watermark completely (not just fading)
* ✅ Avoids global image distortion
* ✅ Works across different image types

---

##  Results

* Successfully processed all 120 images
* High-quality output with minimal artifacts
* No significant loss of important content

---

## ▶ How to Run

```bash
python src/remove_watermark.py samples/watermarked output/
```

---

##  Dependencies

```bash
pip install -r requirements.txt
```

---

##  Observations

* Region-based inpainting is significantly more robust than global enhancement
* Proper mask design is the most critical part of watermark removal
* Simpler pipelines are often more stable and effective

---

##  Future Improvements

* Deep learning-based watermark segmentation
* Adaptive mask refinement using edge detection
* GPU optimization for faster processing
