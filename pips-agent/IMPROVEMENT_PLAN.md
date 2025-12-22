# CV Extraction Improvement Plan

## Goal
Achieve 90%+ accuracy for extracting puzzle structure from screenshots, including complex irregular grids.

---

## Phase 1: Better Grid Detection (High Priority)

### Problem
Current grid detection only works for simple rectangular grids with uniform spacing.

### Solutions

#### 1.1 Multi-Strategy Detection
Instead of relying solely on edge detection + projection analysis:

```python
def extract_puzzle_structure_v2(image_path):
    """Try multiple strategies and pick best result"""
    strategies = [
        detect_by_gridlines(),      # Current method
        detect_by_region_contours(), # NEW
        detect_by_template_matching(), # NEW
        detect_by_cell_corners(),    # NEW
    ]

    results = [strategy(image_path) for strategy in strategies]
    return pick_best_result(results)
```

#### 1.2 Region Contour Detection
Use colored region boundaries to infer cell locations:

```python
def detect_by_region_contours(image):
    """
    1. Segment image by color (find all regions)
    2. Find contours of each colored region
    3. Infer cell grid from region boundaries
    4. Works better for irregular layouts
    """
    # Find all unique colors in image
    regions = segment_by_color(image)

    # Get contours for each region
    contours = [find_contours(region) for region in regions]

    # Find intersection points (likely cell corners)
    corners = find_intersection_points(contours)

    # Reconstruct grid from corners
    cells = reconstruct_grid_from_corners(corners)

    return cells
```

**Advantages:**
- Works with irregular grids
- Handles varied cell sizes
- Less sensitive to gridline artifacts

#### 1.3 Template Matching for Constraint Labels
Detect diamond-shaped constraint labels to infer grid structure:

```python
def detect_by_constraint_labels(image):
    """
    1. Find all diamond shapes (constraint labels)
    2. Use their positions to infer cell boundaries
    3. Diamonds are always between cells/regions
    """
    diamonds = detect_diamonds(image)

    # Diamonds mark region boundaries
    # Use them to infer cell layout
    grid = infer_grid_from_markers(diamonds)

    return grid
```

#### 1.4 Machine Learning Approach (Advanced)
Train a model on puzzle screenshots:

```python
def detect_by_ml(image):
    """
    Use trained model to:
    - Segment cells
    - Identify regions
    - Detect constraints
    - Read dominoes
    """
    model = load_trained_model('pips_detector.h5')
    predictions = model.predict(image)

    cells = predictions['cells']
    regions = predictions['regions']
    constraints = predictions['constraints']

    return {
        'cells': cells,
        'regions': regions,
        'constraints': constraints
    }
```

**Training Data Needed:**
- 100+ annotated puzzle screenshots
- Cell boundaries marked
- Region labels identified
- Constraints labeled

---

## Phase 2: Improved Region Detection (High Priority)

### Problem
K-means color clustering doesn't match logical puzzle regions.

### Solutions

#### 2.1 Constraint-Guided Region Detection
Use detected constraint labels to guide region grouping:

```python
def detect_regions_v2(image, cells, constraint_labels):
    """
    1. Detect constraint diamond labels via OCR/template matching
    2. Use labels to identify which regions exist (A, B, C, etc.)
    3. Group cells by region based on:
       - Color similarity
       - Proximity to constraint labels
       - Connectivity
    """
    # Detect all constraint labels (A, B, C, etc.)
    labels = detect_constraint_labels(image)

    # For each label, find nearby cells with similar colors
    regions = {}
    for label in labels:
        region_cells = find_cells_near_label(label, cells, image)
        regions[label.text] = region_cells

    return regions
```

#### 2.2 Connected Component Analysis
Group cells that touch and have similar colors:

```python
def detect_regions_by_connectivity(cells, image):
    """
    1. Sample color from each cell
    2. Build adjacency graph (which cells touch)
    3. Merge adjacent cells with similar colors
    4. Result: connected regions with consistent colors
    """
    adjacency = build_cell_adjacency(cells)
    colors = sample_cell_colors(cells, image)

    regions = []
    visited = set()

    for cell in cells:
        if cell in visited:
            continue

        # Find all connected cells with similar color
        region = flood_fill_similar_color(
            cell, adjacency, colors,
            color_threshold=30
        )

        regions.append(region)
        visited.update(region)

    return regions
```

#### 2.3 User-Assisted Correction
When detection is uncertain, ask user to clarify:

```python
def detect_with_user_confirmation(image, cells):
    """
    1. Attempt automatic detection
    2. Show result to user with confidence scores
    3. Ask user to correct any mistakes
    4. Learn from corrections over time
    """
    auto_regions = detect_regions_auto(image, cells)

    # Visual confirmation
    annotated_image = draw_detected_regions(image, auto_regions)

    # Ask user: "Does this look correct? [Y/n]"
    confirmed = ask_user_confirmation(annotated_image)

    if not confirmed:
        # Interactive correction
        corrected = user_interactive_correction(auto_regions)
        return corrected

    return auto_regions
```

---

## Phase 3: Domino Detection (Medium Priority)

### Problem
No capability to detect domino tiles from screenshot tray.

### Solutions

#### 3.1 Pip Counting via Template Matching
Detect pip dots on dominoes:

```python
def detect_dominoes(image):
    """
    1. Locate domino tray region (bottom of screenshot)
    2. Segment individual dominoes
    3. Count pips on each half
    4. Return list of domino tuples
    """
    tray_region = locate_domino_tray(image)
    dominoes_images = segment_dominoes(tray_region)

    domino_list = []
    for domino_img in dominoes_images:
        left_half, right_half = split_domino(domino_img)
        left_pips = count_pips(left_half)
        right_pips = count_pips(right_half)
        domino_list.append([left_pips, right_pips])

    return domino_list

def count_pips(half_image):
    """Count circular pip dots in domino half"""
    # Use Hough Circle detection
    circles = cv2.HoughCircles(
        half_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=15
    )

    return len(circles) if circles is not None else 0
```

#### 3.2 OCR for Domino Values (Alternative)
Some puzzle apps display pip counts as numbers:

```python
def detect_dominoes_by_ocr(image):
    """
    If dominoes show numbers instead of pips:
    1. Locate domino tray
    2. OCR each domino
    3. Parse "6-1", "3-3" format
    """
    tray = locate_domino_tray(image)
    domino_regions = segment_dominoes(tray)

    dominoes = []
    for region in domino_regions:
        text = ocr_image(region)  # e.g., "6-1"
        left, right = parse_domino_text(text)
        dominoes.append([left, right])

    return dominoes
```

---

## Phase 4: Enhanced OCR for Constraints (Medium Priority)

### Problem
Current OCR implementation is untested with real puzzles.

### Improvements Needed

#### 4.1 Better Text Localization
Find constraint text more accurately:

```python
def detect_constraints_v2(image, regions):
    """
    1. Detect diamond-shaped labels
    2. Extract text from inside diamonds
    3. Parse constraint format: "=8", ">4", "< 2"
    4. Map each constraint to nearest region
    """
    # Find all diamond shapes
    diamonds = detect_diamond_shapes(image)

    constraints = {}
    for diamond in diamonds:
        # Extract text from diamond interior
        text = ocr_diamond_text(diamond)

        # Parse constraint: "=8", ">4", etc.
        constraint = parse_constraint_text(text)

        # Find closest region
        nearest_region = find_nearest_region(diamond.center, regions)

        constraints[nearest_region] = constraint

    return constraints
```

#### 4.2 Constraint Format Parsing
Handle various text formats:

```python
def parse_constraint_text(text):
    """
    Parse formats:
    - "= 8", "=8"
    - "> 4", ">4"
    - "< 2", "<2"
    - "all equal"
    - "sum = 12"
    """
    text = text.strip().lower()

    # All equal
    if 'all' in text and 'equal' in text:
        return {'type': 'all_equal'}

    # Numeric constraints
    match = re.match(r'([<>=]+)\s*(\d+)', text)
    if match:
        op, value = match.groups()
        return {
            'type': 'sum',
            'operator': op,
            'value': int(value)
        }

    # Sum format
    match = re.match(r'sum\s*([<>=]+)\s*(\d+)', text)
    if match:
        op, value = match.groups()
        return {
            'type': 'sum',
            'operator': op,
            'value': int(value)
        }

    return None
```

#### 4.3 Confidence-Based Validation
Ask user to confirm low-confidence detections:

```python
def ocr_with_validation(image, regions):
    """
    1. Attempt OCR
    2. Calculate confidence score
    3. If confidence < threshold, ask user
    """
    detected = detect_constraints_v2(image, regions)

    verified_constraints = {}
    uncertain_constraints = {}

    for region, (constraint, confidence) in detected.items():
        if confidence >= 0.8:
            verified_constraints[region] = constraint
        else:
            uncertain_constraints[region] = (constraint, confidence)

    # Ask user to verify uncertain ones
    if uncertain_constraints:
        print("Please verify these constraints:")
        for region, (constraint, conf) in uncertain_constraints.items():
            confirmed = ask_user(
                f"Region {region}: {constraint} (confidence: {conf:.0%})"
            )
            if confirmed:
                verified_constraints[region] = constraint

    return verified_constraints
```

---

## Phase 5: Validation and Feedback Loop (Low Priority)

### Solution: Learning from Corrections

```python
class AdaptiveDetector:
    """
    Learn from user corrections over time to improve detection
    """

    def __init__(self):
        self.correction_history = []

    def detect_with_learning(self, image):
        # Attempt detection
        result = self.detect(image)

        # Get user feedback
        corrected = self.get_user_corrections(result)

        # Store correction for learning
        if corrected != result:
            self.correction_history.append({
                'image': image,
                'detected': result,
                'correct': corrected
            })

        return corrected

    def improve_from_history(self):
        """
        Analyze correction history to:
        - Tune detection parameters
        - Identify common failure patterns
        - Improve heuristics
        """
        for correction in self.correction_history:
            self.analyze_failure(correction)
            self.tune_parameters(correction)
```

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ Document limitations
2. ✅ Create improvement plan
3. ⚠️ Implement region contour detection
4. ⚠️ Add constraint label detection (diamonds)
5. ⚠️ Improve OCR constraint parsing

### Phase 2: Core Improvements (2-4 weeks)
1. Multi-strategy grid detection
2. Constraint-guided region detection
3. User-assisted correction UI
4. Comprehensive testing with real puzzles

### Phase 3: Advanced Features (4-8 weeks)
1. Domino pip counting
2. Machine learning detector (if training data available)
3. Adaptive learning from corrections

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Simple grid detection accuracy | 80% | 95% |
| Complex grid detection accuracy | 10% | 80% |
| Region detection accuracy | 30% | 85% |
| Constraint OCR accuracy | ? | 90% |
| Domino detection accuracy | 0% | 85% |
| End-to-end success rate | 15% | 75% |

---

## Required Resources

### Tools
- OpenCV advanced features
- Tesseract OCR with fine-tuning
- Possibly: TensorFlow/PyTorch for ML approach

### Data
- 50+ puzzle screenshots for testing
- 100+ annotated screenshots for ML training (optional)
- User feedback on detection failures

### Time Estimate
- Phase 1: 1-2 weeks (1 developer)
- Phase 2: 2-4 weeks (1 developer)
- Phase 3: 4-8 weeks (1 developer) - optional

---

## Next Steps

1. ✅ Document current limitations
2. ✅ Create this improvement plan
3. Implement region contour detection (highest impact)
4. Test with real user puzzles
5. Iterate based on feedback

---

**Last Updated:** 2024-12-18
**Version:** 1.0
**Status:** Planning Complete, Ready for Implementation
