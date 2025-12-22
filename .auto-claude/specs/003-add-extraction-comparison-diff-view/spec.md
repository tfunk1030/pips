# Add Extraction Comparison/Diff View

## Overview

Allow users to compare extraction results from different AI models or extraction runs side-by-side. The pipeline already collects per-model responses in debug mode - this surfaces them in the UI for manual verification.

## Rationale

The extraction pipeline stores rawResponses from each model (Gemini, GPT, Claude) in result.debug.rawResponses when saveDebugResponses is enabled. The compareCellDetections function (gridValidator.ts lines 602-644) already computes cell-by-cell disagreement analysis. This data exists but isn't exposed to users.

---
*This spec was created from ideation and is pending detailed specification.*
