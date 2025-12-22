# Add Image Stats Diagnostic Endpoint

## Overview

Expose the existing _calculate_image_stats function as a standalone API endpoint that returns brightness, contrast, dynamic range, color balance, and saturation metrics without requiring preprocessing. This helps users diagnose image quality issues before extraction.

## Rationale

The _calculate_image_stats function already exists in cv-service/main.py (lines 483-527) and is fully implemented with comprehensive stats. Exposing it as an endpoint follows the existing API pattern and requires minimal new code.

---
*This spec was created from ideation and is pending detailed specification.*
