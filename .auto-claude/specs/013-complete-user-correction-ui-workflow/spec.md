# Specification: Complete User Correction UI Workflow

## Overview

This task completes the user correction UI workflow in the pips-solver React Native application, enabling users to fix detection errors from the cv-service before solving puzzles. The workflow includes editing grid dimensions, correcting region assignments, fixing constraint values through tap-to-edit interactions, adjusting domino pip values visually, and providing real-time preview of all changes. This feature addresses the fundamental limitation that even improved detection algorithms will occasionally produce errors, and provides users with a smooth path to convert partial detection failures into successful puzzle solves.

## Workflow Type

**Type**: feature

**Rationale**: This is new feature development that adds correction capabilities to an incomplete workflow. While detection infrastructure exists in cv-service, the user-facing correction interface is currently missing or incomplete. This requires building new UI components, state management for edited values, and integration with existing detection outputs.

## Task Scope

### Services Involved
- **pips-solver** (primary) - React Native frontend where correction UI will be implemented
- **cv-service** (integration) - Provides detection output that users will correct

### This Task Will:
- [ ] Implement UI for editing detected grid dimensions (rows/columns)
- [ ] Create region assignment correction interface for individual cells
- [ ] Build tap-to-edit interface for constraint value corrections
- [ ] Add visual domino pip value adjustment controls
- [ ] Implement real-time preview system that reflects changes immediately
- [ ] Connect correction UI to existing detection output from cv-service
- [ ] Manage state for user edits separate from original detection results
- [ ] Validate corrected data before allowing user to proceed to solving

### Out of Scope:
- Improving detection algorithms in cv-service (detection quality improvements)
- Implementing the puzzle solver logic itself
- Adding new detection endpoints or changing detection output format
- Backend persistence of correction history
- Undo/redo functionality (future enhancement)

## Service Context

### pips-solver

**Tech Stack:**
- Language: TypeScript
- Framework: React Native (Expo)
- Key directories:
  - `src/` - Source code
- UI Components: React components
- State Management: React hooks/context (to be determined during implementation)

**Entry Point:** `index.ts`

**How to Run:**
```bash
cd pips-solver
npm run start
# Or with yarn
yarn start
```

**Port:** 3000

**Dependencies:**
- React Navigation for screen flow
- React Native Gesture Handler for interactions
- React Native SVG for visual elements
- Expo Image Picker/Manipulator for image handling

### cv-service

**Tech Stack:**
- Language: Python
- Framework: FastAPI
- Key endpoints:
  - `/extract-geometry` - Grid detection
  - `/crop-puzzle` - Puzzle extraction
  - `/crop-dominoes` - Domino detection

**Entry Point:** `main.py`

**How to Run:**
```bash
cd cv-service
python main.py
# Or
uvicorn main:app --reload --port 8080
```

**Port:** 8080

**Role in This Task:** Provides initial detection results that users will correct

## Files to Modify

**Note:** Specific files not identified during context gathering. Implementation will begin with discovery phase to locate:

| Expected File Type | Service | What to Change |
|-------------------|---------|----------------|
| Correction screen component | pips-solver | Add/complete correction UI implementation |
| Detection result state management | pips-solver | Store and manage edited values |
| Grid dimension editor component | pips-solver | Create dimension adjustment controls |
| Region assignment editor component | pips-solver | Create cell-to-region assignment UI |
| Constraint editor component | pips-solver | Implement tap-to-edit for constraint values |
| Domino pip editor component | pips-solver | Create visual pip adjustment interface |
| Preview renderer component | pips-solver | Update to reflect edited values in real-time |

## Files to Reference

**Note:** Specific pattern files not identified during context gathering. Implementation should discover and follow:

| Expected Pattern | What to Learn |
|-----------------|---------------|
| Existing screen components | React Native screen structure and navigation patterns |
| Current detection result usage | Data structure of cv-service output |
| Existing grid/puzzle rendering | Visual rendering patterns for puzzle elements |
| Form input patterns | How user input is currently handled in the app |
| State management patterns | Redux/Context/hooks usage in existing code |

## Patterns to Follow

### React Native Interactive UI Pattern

Expected pattern based on technology stack:

```typescript
// Tap-to-edit pattern for constraint values
const ConstraintEditor: React.FC<ConstraintEditorProps> = ({
  constraint,
  onValueChange
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(constraint.value);

  const handleSave = () => {
    onValueChange(constraint.id, value);
    setIsEditing(false);
  };

  return (
    <TouchableOpacity onPress={() => setIsEditing(true)}>
      {isEditing ? (
        <TextInput
          value={value}
          onChangeText={setValue}
          onBlur={handleSave}
          autoFocus
          keyboardType="numeric"
        />
      ) : (
        <Text>{value}</Text>
      )}
    </TouchableOpacity>
  );
};
```

**Key Points:**
- Use TouchableOpacity for tap-to-edit interactions
- Maintain local state during editing
- Call parent callback on save to update global state
- Auto-focus input when entering edit mode
- Numeric keyboard for constraint values

### State Management Pattern

```typescript
// Correction state management
interface CorrectionState {
  originalDetection: DetectionResult;
  editedValues: {
    gridDimensions?: { rows: number; cols: number };
    regionAssignments?: Map<CellId, RegionId>;
    constraintValues?: Map<ConstraintId, number>;
    dominoPips?: Map<DominoId, PipValue>;
  };
}

const useCorrectionState = () => {
  const [corrections, setCorrections] = useState<CorrectionState>({
    originalDetection: initialDetection,
    editedValues: {}
  });

  const updateGridDimensions = (rows: number, cols: number) => {
    setCorrections(prev => ({
      ...prev,
      editedValues: {
        ...prev.editedValues,
        gridDimensions: { rows, cols }
      }
    }));
  };

  // Similar methods for other correction types...

  return { corrections, updateGridDimensions, ... };
};
```

**Key Points:**
- Keep original detection separate from edits
- Store only changed values in editedValues
- Provide specific update methods for each correction type
- Use TypeScript interfaces for type safety

### Real-Time Preview Pattern

```typescript
// Preview that merges original and edited values
const PuzzlePreview: React.FC<PreviewProps> = ({
  original,
  edits
}) => {
  const effectiveValues = useMemo(() => ({
    dimensions: edits.gridDimensions || original.dimensions,
    regions: mergeRegionAssignments(original.regions, edits.regionAssignments),
    constraints: mergeConstraints(original.constraints, edits.constraintValues),
    dominoes: mergeDominoes(original.dominoes, edits.dominoPips)
  }), [original, edits]);

  return (
    <View>
      <Grid dimensions={effectiveValues.dimensions} />
      <Regions assignments={effectiveValues.regions} />
      <Constraints values={effectiveValues.constraints} />
      <Dominoes pips={effectiveValues.dominoes} />
    </View>
  );
};
```

**Key Points:**
- Use useMemo to compute effective values from original + edits
- Merge functions prioritize edited values over original
- Preview updates automatically when edits change
- Keep preview logic separate from editor logic

## Requirements

### Functional Requirements

1. **Grid Dimension Editor**
   - Description: Allow users to adjust the detected grid size (rows and columns). Provide numeric inputs or stepper controls to increase/decrease dimensions. Show grid overlay updating in real-time as dimensions change.
   - Acceptance: Users can edit detected grid dimensions, and the preview updates immediately to reflect the new size.

2. **Region Assignment Corrector**
   - Description: Enable users to change which region each cell belongs to. Provide a selection mode where tapping a cell allows choosing from available regions (likely color-coded or numbered). Visual feedback shows region boundaries as they're modified.
   - Acceptance: Users can correct region assignments for any cell, with changes visible immediately in the puzzle preview.

3. **Constraint Value Tap-to-Edit**
   - Description: Implement tap-to-edit interface for constraint numbers. When user taps a constraint, show numeric keyboard to enter correct value. Support different constraint types (sums, values, etc.) as applicable to puzzle types.
   - Acceptance: Users can fix constraint values by tapping and editing, changes reflect immediately without form submission.

4. **Domino Pip Value Adjuster**
   - Description: Create visual controls for adjusting domino pip values (dot patterns). Allow incrementing/decrementing pip counts, or direct selection from valid pip values (0-6 for standard dominoes). Show domino visual updating in real-time.
   - Acceptance: Users can adjust domino pip values visually, with immediate preview feedback.

5. **Real-Time Preview System**
   - Description: Implement preview rendering that merges original detection with user edits. Preview updates immediately when any correction is made. Show both edited and unedited elements, with visual distinction for edited items (e.g., highlight color).
   - Acceptance: All changes (dimensions, regions, constraints, dominoes) are reflected in real-time preview as users make corrections.

### Edge Cases

1. **Invalid Grid Dimensions** - Validate that rows/columns are positive integers within reasonable bounds (e.g., 3-20). Show error message if user enters invalid values. Prevent proceeding to solve with invalid dimensions.

2. **Orphaned Cells After Dimension Change** - If user reduces grid dimensions, handle cells that fall outside new bounds. Either remove them automatically with confirmation, or prevent dimension reduction if it would orphan cells.

3. **Region Assignment to Non-Existent Region** - Validate that users can only assign cells to regions that exist in the puzzle. If regions are dynamically created, provide UI to add/remove regions.

4. **Constraint Value Out of Range** - Validate constraint values against puzzle rules (e.g., sum constraints must be positive, values must be within possible range). Show validation errors inline.

5. **Domino Pip Values Exceeding Valid Range** - Restrict domino pips to valid values (0-6 for standard dominoes). Disable increment button at maximum, decrement button at zero.

6. **Multiple Edits to Same Element** - Handle users changing their mind and editing the same element multiple times. Always use most recent edit, allow reverting to original detection value.

7. **Empty or Missing Detection Results** - Handle case where cv-service returns incomplete or empty detection results. Show appropriate message, allow manual entry of all values from scratch if needed.

## Implementation Notes

### DO
- **Discover existing code first**: Before implementing, locate existing detection result handling, puzzle rendering components, and navigation flow
- **Follow React Native patterns**: Use standard React Native components (TouchableOpacity, TextInput, View) and hooks (useState, useEffect, useMemo)
- **Separate state concerns**: Keep original detection immutable, store only edits in correction state
- **Use TypeScript types**: Define interfaces for detection results, edited values, and correction state
- **Implement incremental preview**: Update preview on each change, don't wait for batch save
- **Provide visual feedback**: Highlight edited elements differently from original detection (e.g., border color, background tint)
- **Validate user input**: Check bounds on dimensions, validate region/constraint/pip values before applying
- **Test with real detection output**: Use actual cv-service responses to ensure integration works correctly
- **Handle loading states**: Show loading indicators while detection runs, disable editing during processing

### DON'T
- **Don't modify cv-service detection logic**: This task is UI-only, detection improvements are out of scope
- **Don't create new detection endpoints**: Use existing cv-service API as-is
- **Don't implement backend persistence**: Corrections are session-only unless explicitly required
- **Don't add complex undo/redo**: Keep initial implementation simple, track current edits only
- **Don't batch corrections**: Apply and preview each change immediately, not on form submit
- **Don't duplicate rendering logic**: Reuse existing puzzle rendering components where possible
- **Don't hardcode puzzle types**: Make correction UI flexible enough to handle different puzzle types
- **Don't skip validation**: Always validate edited values before applying to state

## Development Environment

### Start Services

```bash
# Terminal 1: Start cv-service backend
cd cv-service
python main.py

# Terminal 2: Start pips-solver frontend
cd pips-solver
npm run start
# Follow Expo prompts to run on simulator/device
```

### Service URLs
- **cv-service**: http://localhost:8080
- **pips-solver**: Metro bundler on 3000, Expo client on device/simulator

### Required Environment Variables

From pips-agent (if needed for integration):
- `ANTHROPIC_API_KEY`: Claude API key (in .env)
- `DEBUG_OUTPUT_DIR`: Debug output directory (in .env)

**Note:** Check pips-solver for any required environment variables during discovery phase.

## Success Criteria

The task is complete when:

1. [ ] Users can edit detected grid dimensions via UI controls (spinners, inputs, or sliders)
2. [ ] Users can correct region assignments by tapping cells and selecting new regions
3. [ ] Users can tap constraint values to edit them with numeric keyboard
4. [ ] Users can visually adjust domino pip values (increment/decrement or picker)
5. [ ] All changes reflect immediately in puzzle preview without page reload or submit
6. [ ] No console errors or React Native warnings during correction workflow
7. [ ] Existing tests still pass (if test suite exists)
8. [ ] Correction UI integrates smoothly with navigation flow (can return to solve puzzle after correcting)
9. [ ] Validation prevents invalid corrections (negative dimensions, out-of-range values)
10. [ ] Edited values are clearly distinguishable from original detection in UI

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests

| Test | File | What to Verify |
|------|------|----------------|
| Correction state management | `**/correction.test.ts` or similar | Test that correction state correctly merges original detection with edits |
| Grid dimension validation | `**/GridEditor.test.tsx` or similar | Test dimension bounds validation (min/max rows/cols) |
| Region assignment updates | `**/RegionEditor.test.tsx` or similar | Test that region assignments update correctly for tapped cells |
| Constraint value validation | `**/ConstraintEditor.test.tsx` or similar | Test numeric validation for constraint values |
| Domino pip range validation | `**/DominoEditor.test.tsx` or similar | Test pip values constrained to 0-6 range |
| Preview merge logic | `**/Preview.test.tsx` or similar | Test that preview correctly combines original + edited values |

### Integration Tests

| Test | Services | What to Verify |
|------|----------|----------------|
| Detection result loading | pips-solver ↔ cv-service | Verify correction UI receives and displays detection results from cv-service |
| Correction persistence in session | pips-solver | Verify edited values persist through navigation within session |
| Proceed to solve with corrections | pips-solver | Verify corrected values are passed to solver when user proceeds |

### End-to-End Tests

| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| Complete correction workflow | 1. Upload puzzle image 2. Review detection 3. Tap dimension editor 4. Change rows/cols 5. See preview update 6. Proceed to solve | Corrected dimensions used in puzzle |
| Region correction flow | 1. Load detected puzzle 2. Enter region edit mode 3. Tap cell with wrong region 4. Select correct region 5. See cell change color | Region assignment corrected visually |
| Constraint correction flow | 1. View detected constraints 2. Tap incorrect constraint value 3. Enter correct value 4. See value update 5. Continue to solve | Constraint value updated in puzzle data |
| Domino correction flow | 1. View detected dominoes 2. Tap domino pip adjuster 3. Increment/decrement pips 4. See domino visual update | Domino shows correct pip count |
| Multi-edit workflow | 1. Correct dimension 2. Fix region 3. Edit constraint 4. Adjust domino 5. Verify all show in preview | All edits coexist correctly |

### Browser Verification (if applicable)

**Note:** React Native app runs on mobile devices/simulators, not browsers. Use Expo Go or simulator.

| Screen/Component | Device/Simulator | Checks |
|------------------|------------------|--------|
| Correction screen | iOS Simulator | Grid dimension controls visible and functional |
| Correction screen | Android Emulator | Region assignment tap-to-edit works |
| Preview area | iOS Simulator | Real-time updates render without lag |
| Constraint editor | Android Emulator | Numeric keyboard appears on tap |
| Domino adjuster | iOS Simulator | Visual pip indicators update correctly |

### Manual Testing Checklist

- [ ] **Grid dimension editor**: Increase/decrease rows and cols, verify preview updates
- [ ] **Region editor**: Tap cells in different regions, change assignments, verify colors update
- [ ] **Constraint editor**: Tap constraint numbers, edit values, verify immediate preview change
- [ ] **Domino adjuster**: Modify pip values, verify domino visuals update
- [ ] **Preview synchronization**: Make multiple edits, verify all changes show simultaneously
- [ ] **Validation**: Try invalid inputs (negative numbers, out of range), verify error messages
- [ ] **Navigation**: Correct values, navigate away, return, verify edits persist in session
- [ ] **Proceed to solve**: After corrections, continue to solve screen, verify corrected values used
- [ ] **Original detection preserved**: Verify ability to revert to original detection if desired

### Device/Platform Testing

| Platform | Version | Must Test |
|----------|---------|-----------|
| iOS Simulator | Latest | All correction interactions |
| Android Emulator | API 30+ | All correction interactions |
| Physical iOS Device | iOS 14+ | Touch interactions, keyboard behavior |
| Physical Android Device | Android 10+ | Touch interactions, keyboard behavior |

### Database Verification (if applicable)

**Note:** Current scope is UI-only with session state. If backend persistence added:

| Check | Query/Command | Expected |
|-------|---------------|----------|
| No database changes | N/A | This feature should not persist corrections to backend |

### Performance Verification

| Check | Metric | Expected |
|-------|--------|----------|
| Preview update latency | Time from edit to visual update | < 100ms for responsive feel |
| Memory usage | Memory footprint during corrections | No memory leaks during repeated edits |
| Render performance | Frame rate during preview updates | Maintain 60fps on device |

### QA Sign-off Requirements

- [ ] All unit tests pass for new correction components
- [ ] Integration tests verify cv-service detection results load correctly
- [ ] All E2E test flows complete successfully
- [ ] Manual testing checklist completed on iOS simulator
- [ ] Manual testing checklist completed on Android emulator
- [ ] Device testing on at least one physical device (iOS or Android)
- [ ] No regressions in existing detection flow (without corrections)
- [ ] No regressions in puzzle solving flow
- [ ] Code follows existing pips-solver React Native patterns
- [ ] No security vulnerabilities introduced (input validation present)
- [ ] No console errors or React Native warnings during correction workflow
- [ ] Performance metrics met (preview updates < 100ms)
- [ ] Validation prevents all identified edge cases
- [ ] Edited elements visually distinguishable from original detection
- [ ] Navigation to/from correction screen works smoothly
- [ ] State management doesn't conflict with existing app state

---

## Implementation Plan Structure

Given that specific files were not identified during context gathering, implementation should proceed in phases:

### Phase 1: Discovery (Estimated: 2-4 hours)
- Locate existing detection result handling code
- Find puzzle rendering components (grid, regions, constraints, dominoes)
- Identify navigation flow and screen structure
- Determine state management approach in use
- Document findings for implementation team

### Phase 2: State Management (Estimated: 4-6 hours)
- Design correction state structure
- Implement state hooks/context for managing edits
- Create merge logic for original + edited values
- Add validation functions for each edit type

### Phase 3: UI Components (Estimated: 8-12 hours)
- Build grid dimension editor component
- Build region assignment editor component
- Build constraint tap-to-edit component
- Build domino pip adjuster component
- Ensure consistent styling and UX patterns

### Phase 4: Preview Integration (Estimated: 4-6 hours)
- Update preview renderer to use merged values
- Implement real-time update mechanism
- Add visual distinction for edited elements
- Test preview performance

### Phase 5: Navigation & Flow (Estimated: 3-4 hours)
- Integrate correction screen into navigation
- Connect detection results to correction UI
- Pass corrected values to solver
- Handle session persistence of edits

### Phase 6: Testing & QA (Estimated: 6-8 hours)
- Write unit tests for components and state
- Implement integration tests
- Execute E2E test scenarios
- Perform device testing
- Address QA feedback

**Total Estimated Time:** 27-40 hours

---

## Risk Assessment

### High Risk
- **Complexity of region assignment UI**: Selecting regions visually may require complex gesture handling
  - Mitigation: Start with simple tap-select approach, iterate if needed

- **Real-time preview performance**: Frequent re-renders could cause lag
  - Mitigation: Use React.memo, useMemo, and optimize render logic

### Medium Risk
- **Detection output format unknown**: May need to adapt to unexpected data structures
  - Mitigation: Discovery phase will reveal format, build adapters if needed

- **State management conflicts**: Corrections state may conflict with existing app state
  - Mitigation: Use isolated state context, clear separation of concerns

### Low Risk
- **Validation complexity**: Different puzzle types may have different validation rules
  - Mitigation: Start with basic validation, extend as puzzle types discovered

- **Platform differences**: iOS/Android may behave differently for inputs
  - Mitigation: Test on both platforms, use platform-specific code if needed

---

## Additional Notes

- **Discovery-First Approach**: Since context gathering didn't identify specific files, implementation must begin with thorough codebase discovery
- **Incremental Development**: Build and test each editor (dimensions, regions, constraints, dominoes) independently before integrating
- **User Experience Priority**: Real-time preview is critical for good UX, prioritize smooth interactions over feature completeness
- **Future Enhancements**: Consider undo/redo, correction templates, and backend persistence as follow-up features after core workflow complete
