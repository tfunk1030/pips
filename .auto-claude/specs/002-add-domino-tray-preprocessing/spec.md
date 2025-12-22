# Add Domino Tray Preprocessing

## Overview

Apply the existing image preprocessing pipeline (CLAHE, brightness normalization, white balance) to cropped domino tray images before AI extraction to improve domino pip detection accuracy, especially in low-light photos.

## Rationale

The preprocessing pipeline (_preprocess_image) is fully implemented in cv-service/main.py and crop_domino_region exists in hybrid_extraction.py. The pattern is proven for puzzle cropping - extending to dominoes follows the same architecture.

---
*This spec was created from ideation and is pending detailed specification.*
