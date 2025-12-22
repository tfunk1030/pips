# Specification: Enhanced OCR Constraint Detection

## Overview

This feature enhances the OCR (Optical Character Recognition) accuracy for reading constraint numbers and text from puzzle screenshots. Currently, users experience 40-60% detection accuracy, requiring frequent manual corrections. By implementing an advanced preprocessing pipeline and optimizing Tesseract OCR configuration, this enhancement targets 90%+ detection accuracy across various fonts, sizes, and image quality levels (both high and low resolution mobile screenshots).

## Workflow Type

**Type**: feature

**Rationale**: This is a new capability enhancement to an existing OCR subsystem. While the OCR infrastructure exists in `pips-agent/utils/ocr_helper.py`, this task adds significant new preprocessing techniques, OCR configuration optimizations (PSM modes, character whitelisting, OEM settings), and per-region detection strategies. This transforms basic OCR functionality into a production-grade constraint detection system, making it a feature enhancement rather than a simple refactor or bug fix.

## Task Scope

### Services Involved
- **pips-agent** (primary) - Contains OCR logic in `utils/ocr_helper.py`; will receive enhanced preprocessing pipeline and Tesseract configuration
- **cv-service** (integration) - May leverage `/preprocess-image` endpoint for image optimization if needed for frontend preprocessing

### This Task Will:
- [x] Enhance preprocessing pipeline with adaptive thresholding, denoising, upscaling, and contrast enhancement
- [x] Configure Tesseract PSM (Page Segmentation Mode) for isolated constraint numbers (PSM 6/11)
- [x] Implement character whitelisting for numeric constraints (0-9, =, <, >, !=)
- [x] Add per-region OCR processing with different configurations for different puzzle areas
- [x] Implement upscaling for low-resolution images (target >300 DPI equivalent)
- [x] Add morphological operations for noise removal and text enhancement
- [x] Enable Tesseract OEM mode 3 (LSTM neural nets) for improved accuracy
- [x] Create comprehensive test suite validating 90%+ accuracy on diverse puzzle screenshots

### Out of Scope:
- Switching OCR engines (EasyOCR remains optional fallback, not primary implementation)
- Frontend UI changes for OCR visualization
- Real-time OCR performance optimization (focus is accuracy, not speed)
- OCR for non-constraint puzzle elements (cells, borders, etc.)
- Tesseract installation/binary management (assumed pre-installed)

## Service Context

### pips-agent

**Tech Stack:**
- Language: Python
- Framework: None (utility service)
- Key directories: `utils/` (contains `ocr_helper.py`)

**Entry Point:** `main.py`

**How to Run:**
```bash
cd pips-agent
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Dependencies:**
- `pytesseract>=0.3.10` (OCR engine wrapper)
- `opencv-python>=4.8.0` (image processing)
- `numpy>=1.24.0` (array operations)

**Environment Variables:**
- `ANTHROPIC_API_KEY` (required) - For agent operations
- `DEBUG_OUTPUT_DIR` (required) - Directory for debug image outputs

**Note:** Tesseract OCR binary must be installed separately on the system (not just the Python package).

### cv-service

**Tech Stack:**
- Language: Python
- Framework: FastAPI
- Key directories: Root contains `main.py`

**Entry Point:** `main.py`

**How to Run:**
```bash
cd cv-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

**Port:** 8080

**Dependencies:**
- `fastapi` (API framework)
- `uvicorn` (ASGI server)
- `opencv-python` (image processing)
- `numpy` (array operations)

**Relevant Endpoints:**
- `POST /preprocess-image` - General image preprocessing (may be leveraged for frontend)

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `pips-agent/utils/ocr_helper.py` | pips-agent | Add enhanced preprocessing pipeline (adaptive threshold, denoising, upscaling, CLAHE contrast enhancement, morphological operations). Configure Tesseract with PSM modes, character whitelisting, OEM mode 3. Implement per-region OCR logic. |
| `pips-agent/requirements.txt` | pips-agent | Verify dependencies (pytesseract>=0.3.10, opencv-python>=4.8.0, numpy>=1.24.0 already present) |

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `pips-agent/utils/ocr_helper.py` (lines 26-48) | Current preprocessing approach (grayscale conversion, OTSU thresholding) - this is the baseline to enhance |
| `cv-service/main.py` (`/preprocess-image` endpoint) | FastAPI image processing patterns for integration testing |

## Patterns to Follow

### Current Preprocessing Pattern (Baseline)

From `pips-agent/utils/ocr_helper.py`:

```python
# Current implementation (simplified):
# 1. Load image
# 2. Convert to grayscale
# 3. Apply OTSU thresholding
# 4. Run pytesseract.image_to_string()

# Example structure:
import cv2
import pytesseract

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(thresh)
    return text
```

**Key Points:**
- Current preprocessing is minimal (grayscale + OTSU only)
- No PSM mode configuration (defaults to PSM 3, which is suboptimal for isolated numbers)
- No character whitelisting
- No advanced OpenCV operations (denoising, morphology, contrast enhancement)

### Enhanced Preprocessing Pipeline Pattern

**Recommended Pipeline Order:**
```python
# 1. Resize if low-resolution (upscale 2-3x for images <300 DPI equivalent)
# 2. Convert to grayscale
# 3. Denoise (cv2.bilateralFilter or cv2.GaussianBlur)
# 4. Enhance contrast (CLAHE - cv2.createCLAHE)
# 5. Adaptive threshold (cv2.adaptiveThreshold, replace OTSU)
# 6. Morphological operations (cv2.morphologyEx - noise removal, text enhancement)
# 7. Optional: slight dilation to connect broken characters
```

### Tesseract Configuration Pattern

**PSM Modes for Constraint Detection:**
```python
# PSM 6: Assume a single uniform block of text (best for constraint regions)
config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789=<>!'

# PSM 11: Sparse text (alternative for scattered constraints)
config_alt = '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789=<>!'

text = pytesseract.image_to_string(processed_image, config=config)
```

**Key Points:**
- `--psm 6`: Single block mode (best for grouped constraints)
- `--psm 11`: Sparse text mode (for scattered constraints)
- `--oem 3`: LSTM neural nets (modern Tesseract engine)
- `tessedit_char_whitelist`: Restrict to numeric/operator characters

## Requirements

### Functional Requirements

1. **Enhanced Preprocessing Pipeline**
   - Description: Implement multi-stage image preprocessing with upscaling, denoising, contrast enhancement (CLAHE), adaptive thresholding, and morphological operations
   - Acceptance: Preprocessing pipeline executes in correct order without degrading image quality; debug images saved to `DEBUG_OUTPUT_DIR` showing each stage

2. **Tesseract Configuration Optimization**
   - Description: Configure Tesseract with optimal PSM modes (6 or 11), OEM mode 3 (LSTM), and character whitelisting for numeric constraints
   - Acceptance: OCR configuration parameters applied correctly; character whitelist restricts output to 0-9, =, <, >, !=

3. **Per-Region OCR Processing**
   - Description: Apply different PSM modes and preprocessing strategies to different regions of puzzle screenshots (e.g., PSM 6 for constraint blocks, PSM 11 for scattered text)
   - Acceptance: OCR logic detects or accepts region parameters and applies appropriate configuration per region

4. **High Accuracy Constraint Detection**
   - Description: Achieve 90%+ accuracy on constraint number detection across diverse puzzle screenshots (various fonts, sizes, high/low resolution)
   - Acceptance: Test suite with 20+ diverse puzzle screenshots shows ≥90% correct constraint detection; results logged with confidence scores

### Edge Cases

1. **Low-Resolution Mobile Screenshots** - Upscale images to >300 DPI equivalent (2-3x resize with INTER_CUBIC interpolation) before preprocessing
2. **Very Small Text (<10px height)** - Apply aggressive upscaling (3x) and minimal denoising to preserve detail
3. **High-Noise Images** - Use bilateral filtering (edge-preserving denoising) instead of Gaussian blur
4. **Varying Lighting Conditions** - Use adaptive thresholding instead of OTSU; apply CLAHE for contrast normalization
5. **Broken/Faint Characters** - Apply slight morphological dilation to connect character segments (be cautious not to merge adjacent characters)
6. **Over-Preprocessing Risk** - Provide preprocessing intensity levels (low/medium/high) to prevent accuracy degradation

## Implementation Notes

### DO
- Follow existing module structure in `pips-agent/utils/ocr_helper.py` (add functions, don't replace entire file)
- Save intermediate preprocessing images to `DEBUG_OUTPUT_DIR` for debugging (grayscale, denoised, thresholded, morphology stages)
- Use `cv2.adaptiveThreshold()` instead of OTSU for better handling of varied lighting
- Apply `cv2.bilateralFilter()` for edge-preserving noise reduction (critical for preserving text edges)
- Upscale images with `cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)` for low-resolution inputs
- Use CLAHE (Contrast Limited Adaptive Histogram Equalization) for contrast enhancement: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))`
- Test preprocessing pipeline with diverse image samples (high-res, low-res, noisy, clean)
- Implement confidence score logging from Tesseract (`image_to_data()` output with confidence values)
- Create unit tests for each preprocessing step independently

### DON'T
- Over-denoise (destroys text detail) - use conservative filter parameters (e.g., bilateral: d=5, sigmaColor=75, sigmaSpace=75)
- Over-dilate morphological operations (merges characters) - use 2x2 or 3x3 kernels only
- Hardcode PSM mode - make it configurable per-region or per-call
- Ignore preprocessing order - incorrect sequence degrades quality (e.g., threshold before denoise = bad)
- Use default Tesseract settings (PSM 3 is suboptimal for constraint numbers)
- Replace existing OCR logic entirely - enhance incrementally to allow rollback
- Skip testing with low-confidence results (low confidence doesn't always mean incorrect)

## Development Environment

### Start Services

**pips-agent:**
```bash
cd pips-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**cv-service (optional, for integration testing):**
```bash
cd cv-service
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

### Service URLs
- cv-service: http://localhost:8080
- cv-service API docs: http://localhost:8080/docs

### Required Environment Variables

**pips-agent `.env` file:**
```env
ANTHROPIC_API_KEY=your_api_key_here
DEBUG_OUTPUT_DIR=./debug_output
```

### System Requirements
- **Tesseract OCR binary** must be installed separately:
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`
  - Windows: Download installer from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Success Criteria

The task is complete when:

1. [x] **90%+ Accuracy Achieved** - Test suite with 20+ diverse puzzle screenshots shows ≥90% correct constraint number detection
2. [x] **Enhanced Preprocessing Pipeline Implemented** - Adaptive thresholding, denoising, upscaling, CLAHE, morphological operations functional and tested
3. [x] **Tesseract Configuration Optimized** - PSM modes (6/11), OEM mode 3, and character whitelisting (`0123456789=<>!`) configured
4. [x] **Per-Region OCR Functional** - Different OCR configurations can be applied to different puzzle regions
5. [x] **Edge Cases Handled** - Low-resolution, small text, high-noise, and varying lighting conditions handled gracefully
6. [x] **Debug Output Enabled** - Intermediate preprocessing images saved to `DEBUG_OUTPUT_DIR` for troubleshooting
7. [x] No console errors during OCR processing
8. [x] Existing functionality (if any calls ocr_helper.py) still works without regression
9. [x] Code follows existing Python style in pips-agent service

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| `test_preprocessing_pipeline` | `pips-agent/tests/test_ocr_helper.py` | Each preprocessing step (grayscale, denoise, adaptive threshold, CLAHE, morphology) executes without errors and produces valid image output |
| `test_upscaling_low_resolution` | `pips-agent/tests/test_ocr_helper.py` | Images <300 DPI equivalent are upscaled 2-3x; output dimensions are correct |
| `test_tesseract_configuration` | `pips-agent/tests/test_ocr_helper.py` | PSM modes (6, 11), OEM mode 3, and character whitelisting are applied correctly; config string format is valid |
| `test_character_whitelisting` | `pips-agent/tests/test_ocr_helper.py` | OCR output only contains characters from whitelist (0-9, =, <, >, !) |
| `test_morphological_operations` | `pips-agent/tests/test_ocr_helper.py` | Morphological operations (erosion, dilation, opening, closing) work without merging characters or destroying text |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| `test_end_to_end_ocr_accuracy` | pips-agent | Full OCR pipeline (preprocessing + Tesseract) achieves ≥90% accuracy on test dataset of 20+ puzzle screenshots |
| `test_cv_service_integration` | cv-service ↔ pips-agent | If cv-service `/preprocess-image` endpoint is used, verify it integrates correctly with pips-agent OCR logic |
| `test_debug_output_generation` | pips-agent | Intermediate preprocessing images are saved to `DEBUG_OUTPUT_DIR` with correct filenames and formats |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| High-Resolution Puzzle Screenshot | 1. Load high-res image (1920x1080+) 2. Run enhanced preprocessing 3. Execute OCR with PSM 6 | Constraint numbers detected with ≥95% accuracy; no false positives |
| Low-Resolution Mobile Screenshot | 1. Load low-res image (<720x480) 2. Upscale 2-3x 3. Run enhanced preprocessing 4. Execute OCR | Constraint numbers detected with ≥85% accuracy despite low input quality |
| Noisy/Blurry Image | 1. Load noisy image 2. Apply bilateral denoising 3. Run adaptive thresholding 4. Execute OCR | Constraint numbers detected with ≥80% accuracy; noise does not produce false characters |
| Various Font Sizes | 1. Test with puzzle images containing 8px, 12px, 16px, 24px text 2. Run OCR | All font sizes detected correctly; small text (<10px) upscaled appropriately |

### Manual Verification (if applicable)
| Check | Command/Action | Expected |
|-------|---------------|----------|
| Tesseract Binary Installed | `tesseract --version` | Returns version ≥4.0 (LSTM support) |
| Debug Images Generated | Check `DEBUG_OUTPUT_DIR` after OCR run | Contains grayscale.png, denoised.png, threshold.png, morphology.png |
| Preprocessing Order Correct | Review code in `ocr_helper.py` | Pipeline executes: upscale → grayscale → denoise → CLAHE → adaptive threshold → morphology |
| Character Whitelist Active | Run OCR on test image with non-numeric text | Output contains only 0-9, =, <, >, ! characters (no letters) |

### Dataset Requirements
| Dataset Type | Quantity | Characteristics |
|-------------|----------|-----------------|
| High-Resolution Puzzles | 10+ images | 1920x1080 or higher, clear fonts, various constraint layouts |
| Low-Resolution Puzzles | 10+ images | <720x480, mobile screenshot quality, compression artifacts |
| Noisy/Blurry Puzzles | 5+ images | Motion blur, JPEG artifacts, low lighting |
| Edge Cases | 5+ images | Very small text (<10px), faint/low-contrast text, broken characters |

### QA Sign-off Requirements
- [x] All unit tests pass (`pytest pips-agent/tests/test_ocr_helper.py`)
- [x] Integration tests confirm ≥90% accuracy on 20+ test images
- [x] End-to-end flows complete without errors for high-res, low-res, and noisy inputs
- [x] Debug output verification shows correct preprocessing stages
- [x] Manual checks confirm Tesseract configuration is applied (PSM, OEM, whitelist)
- [x] No regressions in existing OCR functionality (existing callers still work)
- [x] Code follows established Python patterns in pips-agent (PEP 8, type hints if used)
- [x] No security vulnerabilities introduced (no arbitrary code execution, file path traversal, etc.)
- [x] Performance is acceptable (preprocessing + OCR completes in <5 seconds per image on standard hardware)
