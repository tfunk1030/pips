# Testing Guide: AI Extraction Improvements

## Quick Test Checklist

### Test 1: High-Quality Screenshot

**Goal:** Verify high confidence scores and accurate extraction

1. Use a clear, well-lit Pips puzzle screenshot
2. Run AI extraction
3. **Expected Results:**
   - ✅ Grid confidence ≥ 90%
   - ✅ Regions confidence ≥ 90%
   - ✅ Constraints confidence ≥ 85%
   - ✅ All green confidence indicators
   - ✅ No verification pass triggered

### Test 2: Poor Quality Screenshot

**Goal:** Verify low confidence detection and helpful warnings

1. Use a blurry, cropped, or low-resolution screenshot
2. Run AI extraction
3. **Expected Results:**
   - ⚠️ One or more confidence scores < 80%
   - ⚠️ Orange or red confidence indicators
   - ⚠️ Alert message listing low-confidence areas
   - ℹ️ Verification pass may trigger

### Test 3: Verification Pass

**Goal:** Test self-correction feature

1. Use a moderately ambiguous screenshot (partial crop, slight blur)
2. Run AI extraction with verification enabled
3. **Expected Results:**
   - 📊 Initial extraction with medium confidence (80-89%)
   - 🔄 Verification pass triggered
   - ✅ Corrections applied if AI finds errors
   - ℹ️ Total extraction time ~2x normal (acceptable trade-off)

### Test 4: Constraint Edge Cases

**Goal:** Verify improved constraint detection

Test with puzzles containing:

- [ ] Operator symbols: `<`, `>`, `≠`, `=`
- [ ] Large sum values: `> 15`, `< 30`
- [ ] "All equal" constraints
- [ ] Multiple constraints per region

**Expected Results:**

- Constraints correctly parsed
- Constraint confidence reflects text clarity

### Test 5: UI Integration

**Goal:** Verify confidence indicators display correctly

1. Complete AI extraction (any screenshot)
2. Navigate through all 4 steps
3. **Check:**
   - [ ] Step 1 shows grid confidence indicator
   - [ ] Step 2 shows region confidence indicator
   - [ ] Indicators use correct colors (green/orange/red)
   - [ ] Compact format displays cleanly

### Test 6: Error Handling

**Goal:** Verify improved error messages

Test scenarios:

- [ ] Invalid API key → "Please check your API key in Settings"
- [ ] Model unavailable → "Issue with AI model, try again later"
- [ ] Malformed JSON → "AI response malformed, try again or extract manually"

### Test 7: Partial Success

**Goal:** Verify graceful degradation

1. Use screenshot with dominoes cropped out
2. Run AI extraction
3. **Expected Results:**
   - ✅ Board extracted successfully
   - ⚠️ Domino extraction fails gracefully
   - ℹ️ Alert: "Board extracted, add dominoes manually in Step 4"
   - ✅ User can proceed to Step 4 and add dominoes manually

## Performance Tests

### Extraction Speed

- **Baseline:** ~3-5 seconds for board + dominoes
- **With Verification (low confidence):** ~6-10 seconds
- **Target:** <10 seconds even with verification

### API Cost

- **Baseline:** ~$0.01-0.03 per extraction
- **With Verification:** ~$0.015-0.04 per extraction
- **Acceptable Range:** <$0.05 per extraction

## Regression Tests

Verify existing functionality still works:

- [ ] Manual grid alignment (drag edges, adjust rows/cols)
- [ ] Manual region painting (tap/drag)
- [ ] Manual constraint entry
- [ ] Manual domino entry
- [ ] Draft auto-save
- [ ] Draft recovery on app restart
- [ ] Puzzle creation and saving

## Known Limitations

### Expected Failures (Acceptable)

1. **Extremely Low Resolution:** <400px width
2. **Severe Cropping:** More than 50% of grid cut off
3. **Rotated Images:** Puzzle not upright
4. **Multiple Puzzles:** Multiple puzzles in one screenshot

### Workarounds

- Re-take screenshot with better framing
- Use manual extraction workflow
- Rotate image before importing

## Bug Report Template

If AI extraction produces unexpected results:

```
**Screenshot Quality:** [Clear/Blurry/Cropped/Other]
**Confidence Scores:**
  - Grid: [0.XX]
  - Regions: [0.XX]
  - Constraints: [0.XX]
  - Dominoes: [0.XX]

**What Went Wrong:**
[Description of extraction error]

**Expected:**
[What should have been extracted]

**Actual:**
[What was actually extracted]

**Verification Pass Triggered:** [Yes/No]
```

## Success Criteria

✅ **Passing Grade:** 9 out of 10 high-quality screenshots extract correctly with confidence ≥ 90%

⚠️ **Needs Improvement:** <7 out of 10 high-quality screenshots extract correctly

❌ **Critical Issue:** Extraction fails on clear, well-framed screenshots

## Monitoring in Production

### Key Metrics to Track

1. **Extraction Success Rate** (target: >90%)
2. **Average Confidence Scores** (target: >0.85)
3. **Verification Pass Frequency** (target: <20% of extractions)
4. **User Manual Corrections** (target: <30% of extractions need edits)
5. **API Cost per Extraction** (target: <$0.04)

### User Feedback Questions

- "How often does AI extraction work on the first try?"
- "Which step requires the most manual correction?"
- "Are confidence indicators helpful for identifying errors?"

---

**Last Updated:** December 19, 2025
**Test Coverage:** Phases 1 & 2 (Enhanced Prompts + Verification)
