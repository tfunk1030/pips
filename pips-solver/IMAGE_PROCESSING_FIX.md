# Image Processing Fix - December 2025

## Problem

Users were encountering "Could not process image" errors (HTTP 400) when attempting AI extraction. The error occurred because:

1. **Image Manipulation Failure**: `ImageManipulator.manipulateAsync` was failing with "Image context has been lost" error, particularly on iOS
2. **HEIC Format Issue**: When manipulation failed, the app fell back to using the original base64 from the image picker, which could be in HEIC format (common on iOS)
3. **Unsupported Format**: The Claude Vision API does not support HEIC format, only JPEG, PNG, WebP, and GIF
4. **Poor Error Messages**: The generic "Could not process image" error didn't guide users on how to fix the issue

## Root Cause

The image processing pipeline had a critical flaw in its fallback logic:

```typescript
// OLD BEHAVIOR (problematic):
try {
  manipulated = await tryManipulate(1600);
} catch {
  try {
    manipulated = await tryManipulate(1024);
  } catch {
    // Falls back to original base64 (could be HEIC!)
    manipulated = null;
  }
}

const finalBase64 = manipulated?.base64 || base64FromFile; // HEIC base64 → API rejects
```

## Solution

### 1. Early Detection of HEIC Format

Added HEIC detection based on file URI:

```typescript
const isLikelyHEIC =
  asset.uri.toLowerCase().includes('.heic') || asset.uri.toLowerCase().includes('.heif');
```

### 2. Fail Fast on HEIC When Manipulation Fails

When image manipulation fails AND the image is HEIC, we now show a clear error and prevent the API call:

```typescript
if (!base64FromFile || isLikelyHEIC) {
  const reason = isLikelyHEIC
    ? 'This image is in HEIC format which must be converted to JPEG for AI processing, but the conversion failed.'
    : 'Failed to process the selected image.';

  Alert.alert('Image Processing Error', reason + helpfulSuggestions);
  return; // Don't attempt API call
}
```

### 3. Enhanced Error Messages

Added context-specific error messages for different failure scenarios:

- **HEIC detected**: Explains that HEIC must be converted and suggests taking a screenshot
- **Image processing failed**: Provides actionable steps (screenshot, convert to JPEG, use different image)
- **API 400 error**: Recognizes "Could not process image" and provides detailed troubleshooting steps

### 4. Validation Before API Call

Added validation to ensure base64 data exists and has minimum viable length:

```typescript
if (!finalBase64 || finalBase64.length < 100) {
  Alert.alert('Image Data Error', 'The selected image could not be processed...');
  return;
}
```

### 5. Enhanced Debugging

Added diagnostic logging in the extraction service:

```typescript
console.log(`[DEBUG] extractBoard - Image media type detected: ${mediaType}`);
console.log(`[DEBUG] extractBoard - Base64 length: ${normalizedBase64.length} chars`);
console.log(
  `[DEBUG] extractBoard - Base64 first 50 chars: ${normalizedBase64.substring(0, 50)}...`
);
```

## User-Facing Changes

### Before

- Generic error: "Extraction Failed: API error: 400"
- No guidance on what went wrong or how to fix it
- App would attempt API call with unsupported image format

### After

- Specific error messages based on failure type
- Clear instructions: "Take a screenshot" or "Convert to JPEG first"
- Prevention of API calls with invalid image data (saves user's API quota)
- Better logging for troubleshooting

## Testing Recommendations

1. **Test with HEIC image** (iOS Photos Library) - should show clear error before API call
2. **Test with screenshot** (PNG format) - should work successfully
3. **Test with large JPEG** - should successfully downscale or show helpful error
4. **Test on iOS vs Android** - both platforms should handle image formats appropriately

## Technical Notes

### Why HEIC Fails

- HEIC (High Efficiency Image Format) is Apple's default photo format since iOS 11
- Claude Vision API specification (as of 2023-06-01) supports: JPEG, PNG, WebP, GIF only
- HEIC base64 data is valid base64, so it passes validation but fails at API level with 400 error

### Why Image Manipulation Fails

The "Image context has been lost" error typically occurs when:

- Device is memory-constrained
- Image is very large (> 4000px)
- GPU context is lost during async operation
- Image format is not supported by the manipulation library

### Fallback Strategy

The proper fallback sequence is now:

1. Try manipulation at 1600px width (optimal for AI)
2. Try manipulation at 1024px width (more likely to succeed on low memory)
3. If both fail AND image is not HEIC, allow original base64 (with warning)
4. If both fail AND image IS HEIC, reject and guide user

## Related Files

- `src/app/screens/OverlayBuilderScreen.tsx` - Image picker and processing logic
- `src/services/aiExtraction.ts` - AI extraction service with base64 validation
- `src/model/overlayTypes.ts` - Type definitions (fixed merge conflicts)

## Future Improvements

1. **Client-side HEIC → JPEG conversion**: Explore libraries like `react-native-heic-converter`
2. **Progressive image resizing**: Try multiple sizes automatically (1600 → 1024 → 800 → 600)
3. **Format detection from base64 header**: More reliable than URI-based detection
4. **Compression optimization**: Better balance between quality and size
5. **User preference**: Allow users to choose quality/size tradeoffs
