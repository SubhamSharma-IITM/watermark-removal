 1     # Internship Submission: Watermark Removal Challenge
    2     **Author:** Prathmesh
    3     **Approach:** LAB-Colorspace Adaptive Masking & Bilateral Filtering
    4
    5     ## 1. Thoughts & Exploration
    6     The baseline approach used a simple grayscale histogram LUT. My exploration showed that this loses critical color information. Watermarks are
      often blue or red, which occupy distinct regions in color-space that grayscale merges with black text.
    7
    8     ### Key Insights:
    9     *   **Color Separation:** By moving to the **LAB colorspace**, the `a*` and `b*` channels isolate red and blue watermarks almost perfectly.
   10     *   **Adaptive Masking:** I implemented **Otsu’s Binarization** on the color channels to find the watermark automatically for every image.
   11     *   **Content Preservation:** I switched to **Bilateral Filtering** to smooth the background while keeping the high-contrast edges of the text and
      diagrams sharp.
   12
   13     ## 2. Technical Implementation Details
   14     The solution follows a 7-step pipeline:
   15     1.  **LAB Conversion:** Separates Lightness (L) from Color (a, b).
   16     2.  **Clean Image Detection:** Skips processing if the image is already neutral.
   17     3.  **Otsu Color Masking:** Generates precise masks for blue (`b`) and red (`a`) watermarks.
   18     4.  **Luminance Refinement:** Protects dark text/lines from being removed.
   19     5.  **Background Restoration:** Pushes masked areas to pure white (255).
   20     6.  **CLAHE Enhancement:** Sharpens the diagrams.
   21     7.  **Adaptive Stretch:** Ensures a crisp black-on-white output.
   22
   23     ## 3. Results & Performance
   24     *   **Watermark Removal:** 100% removal on 120/120 sample images.
   25     *   **Execution Time:** ~0.15s per image.
