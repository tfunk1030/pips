# Specification: Graduated Hint System

## Overview

Implement a multi-level hint system that progressively reveals puzzle-solving insights through four escalating disclosure levels. This feature addresses the user pain point of wanting learning assistance without complete spoilers, providing an educational alternative to binary "show solution" or "online lookup" approaches. The system will analyze the current puzzle state and generate contextually relevant hints at each level, allowing users to control their own pace of discovery.

## Workflow Type

**Type**: feature

**Rationale**: This is net-new functionality that adds a graduated hint revelation system to the puzzle application. It's not refactoring existing code, fixing bugs, or migrating infrastructure—it's building a new user-facing capability that requires both frontend UI components and backend logic for hint generation.

## Task Scope

### Services Involved
- **pips-solver** (primary) - React Native frontend that will display the hint UI and manage user progression through hint levels
- **pips-agent** (integration) - Python backend that will generate contextually relevant hints based on current puzzle state analysis

### This Task Will:
- [ ] Create Level 1 hint generation: General strategy hints without revealing specific cells or regions
- [ ] Create Level 2 hint generation: Identify specific region or constraint to examine
- [ ] Create Level 3 hint generation: Reveal a single specific cell placement as a concrete nudge
- [ ] Create Level 4 hint generation: Show partial solution subset for truly stuck situations
- [ ] Build frontend UI component for displaying hints with user-controlled level progression
- [ ] Implement state management to track hint level and prevent repetition
- [ ] Create API endpoint for hint generation requests with puzzle state payload
- [ ] Add puzzle state analysis logic to generate contextually appropriate hints

### Out of Scope:
- Full solution reveal functionality (separate feature)
- Pre-generated static hints (all hints must be dynamic based on puzzle state)
- Automatic hint escalation (user must explicitly request each level)
- Hint history/replay functionality across puzzle sessions
- Hint generation for puzzle types beyond the current puzzle system

## Service Context

### pips-solver

**Tech Stack:**
- Language: TypeScript
- Framework: React Native (Expo)
- Key directories: `src/`

**Entry Point:** `index.ts`

**How to Run:**
```bash
cd pips-solver
npm run start
```

**Port:** 3000

**Dependencies:**
- React Navigation for screen management
- AsyncStorage for hint state persistence
- React Native Gesture Handler for hint UI interactions

### pips-agent

**Tech Stack:**
- Language: Python
- Framework: CLI/Agent (Claude SDK)
- Key directories: `utils/`

**Entry Point:** `main.py`

**How to Run:**
```bash
cd pips-agent
python main.py
```

**Dependencies:**
- Claude Agent SDK for AI-powered hint generation
- PyYAML for configuration
- NumPy/scikit-learn for puzzle state analysis

**Environment Variables:**
- `ANTHROPIC_API_KEY`: Required for Claude API access (hint generation)
- `DEBUG_OUTPUT_DIR`: Optional for debugging hint generation

## Files to Modify

| File | Service | What to Change |
|------|---------|---------------|
| `pips-agent/main.py` | pips-agent | Add hint generation endpoint/handler with 4-level logic |
| `pips-solver/src/[HintComponent].tsx` | pips-solver | Create new hint UI component with progressive disclosure controls |
| `pips-solver/src/[PuzzleScreen].tsx` | pips-solver | Integrate hint button and state management into main puzzle interface |
| `pips-agent/utils/[hint_generator].py` | pips-agent | Create hint generation utility with puzzle state analysis |

**Note**: Exact file paths require discovery phase to identify current puzzle state management and UI component structure.

## Files to Reference

These files show patterns to follow:

| File | Pattern to Copy |
|------|----------------|
| `cv-service/main.py` | FastAPI endpoint pattern for structured API responses |
| `pips-solver/src/[existing UI components]` | React Native component styling and navigation patterns |
| `pips-agent/utils/[existing utilities]` | Python utility structure and error handling patterns |

**Note**: Context phase did not identify specific reference files. Discovery of existing patterns is required before implementation.

## Patterns to Follow

### API Response Structure Pattern

From typical FastAPI services (reference `cv-service/main.py`):

```python
@app.post("/generate-hint")
async def generate_hint(request: HintRequest):
    try:
        # Validate puzzle state
        # Analyze current state
        # Generate hint for requested level
        return {
            "success": True,
            "hint": {
                "level": 1,
                "content": "Strategy hint text...",
                "type": "strategy"
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

**Key Points:**
- Use Pydantic models for request/response validation
- Return structured JSON with success/error fields
- Include hint level metadata in response

### React Native Component Pattern

From typical React Native components:

```typescript
interface HintModalProps {
  visible: boolean;
  onClose: () => void;
  currentLevel: number;
  onRequestHint: (level: number) => void;
}

export const HintModal: React.FC<HintModalProps> = ({
  visible,
  onClose,
  currentLevel,
  onRequestHint
}) => {
  // Component implementation
};
```

**Key Points:**
- Use TypeScript interfaces for props
- Handle modal visibility state
- Separate concerns: UI vs. logic

## Requirements

### Functional Requirements

1. **Level 1: Strategic Guidance**
   - Description: Generate general strategy hints that guide thinking without revealing specific cells, regions, or values. Examples: "Look for naked pairs in rows", "Consider X-wing patterns", "Focus on constraint propagation".
   - Acceptance: User can request Level 1 hint and receive strategic guidance text that doesn't reveal puzzle specifics.

2. **Level 2: Focused Direction**
   - Description: Analyze puzzle state and identify a specific region (row/column/box) or constraint where progress can be made. Example: "Examine row 3 for hidden singles", "Check the top-left 3x3 box for constraint violations".
   - Acceptance: User can request Level 2 hint and receive direction to a specific puzzle region without cell values revealed.

3. **Level 3: Specific Cell Placement**
   - Description: Analyze puzzle state, identify a solvable cell, and reveal its correct value as a concrete nudge. Example: "Cell R3C5 should be 7" or "Place a 4 in the middle cell of row 2".
   - Acceptance: User can request Level 3 hint and receive specific cell coordinates with correct value to place.

4. **Level 4: Partial Solution**
   - Description: Generate and reveal a small subset of the complete solution (e.g., 3-5 cells) to provide substantial progress for stuck users without full spoiler.
   - Acceptance: User can request Level 4 hint and receive multiple cell placements that constitute partial solution progress.

5. **User-Controlled Progression**
   - Description: Users must explicitly request each hint level via UI controls. No automatic escalation. Users can skip levels or request same level multiple times.
   - Acceptance: Hint UI provides buttons/controls for each level, hints only appear on user action, same level can be requested again if needed.

6. **Context-Aware Hint Generation**
   - Description: All hints must analyze the current puzzle state to ensure relevance. Hints adapt to user progress and don't repeat already-revealed information.
   - Acceptance: Requesting hints for same puzzle state with different progress yields different contextually appropriate hints.

### Edge Cases

1. **Puzzle Already Solved** - If puzzle is complete, hint system should detect this and display "Puzzle already solved" message instead of generating hints.

2. **Invalid Puzzle State** - If current puzzle state contains errors (constraint violations), Level 1-2 hints should guide user to identify errors rather than proceed with solving. Level 3-4 should fail gracefully with error message.

3. **Multiple Hint Requests at Same Level** - System should track previously revealed hints and avoid repetition. Generate alternative hints at the same strategic level if available.

4. **No Available Hints** - If puzzle state is valid but solver cannot determine next logical step (extremely rare), system should escalate to "Try a different approach" message or suggest backtracking.

5. **Network Failure** - Frontend should handle API timeout/failure gracefully with retry mechanism and offline indicator.

## Implementation Notes

### DO
- Follow React Native component patterns in `pips-solver/src/` for consistent UI styling
- Reuse existing puzzle state management structures for hint request payloads
- Leverage Claude Agent SDK in `pips-agent` for natural language hint generation (Level 1-2)
- Use algorithmic analysis for Level 3-4 (may require puzzle solver logic)
- Persist hint level state in AsyncStorage to remember user's last hint level across app restarts
- Implement request debouncing to prevent duplicate hint API calls
- Add loading states and skeleton screens while hint generation is in progress

### DON'T
- Create new puzzle state serialization formats—use existing puzzle representation
- Pre-generate static hints—all hints must be dynamic based on current state
- Auto-escalate hint levels—user must control progression
- Mutate puzzle state when generating hints—hint generation should be read-only operation
- Expose full solver logic in hint responses—only reveal what's appropriate for each level
- Store hint content in frontend—fetch fresh hints each time to reflect latest puzzle state

## Development Environment

### Start Services

```bash
# Start frontend (pips-solver)
cd pips-solver
npm run start

# Start agent backend (pips-agent) - if required for hint generation
cd pips-agent
python main.py
```

### Service URLs
- pips-solver: http://localhost:3000 (Expo dev server)
- pips-agent: CLI-based (no HTTP server by default)

### Required Environment Variables

For pips-agent (if hint generation uses Claude SDK):
- `ANTHROPIC_API_KEY`: Required for AI-powered hint generation
- `DEBUG_OUTPUT_DIR`: Optional for debugging hint analysis

For pips-solver:
- Standard Expo/React Native environment (no custom variables identified)

### Discovery Tasks Required

Before implementation can begin, the following must be discovered:

1. **Puzzle State Structure**: How is the current puzzle state represented? (grid array, object, serialization format)
2. **Puzzle Type**: What puzzle type is this? (Sudoku, dominoes-based, custom logic puzzle)
3. **Existing Solver Logic**: Does the codebase have puzzle-solving algorithms that can be leveraged for hint generation?
4. **UI Component Locations**: Where are existing puzzle UI components located in `pips-solver/src/`?
5. **State Management Pattern**: Redux, Context API, or custom state management for puzzle state?
6. **API Integration Pattern**: How does pips-solver currently communicate with backend services (cv-service integration)?

## Success Criteria

The task is complete when:

1. [ ] User can tap "Hint" button in puzzle interface to open hint modal
2. [ ] User can request Level 1 hint and receive general strategy guidance text
3. [ ] User can request Level 2 hint and receive specific region/constraint to focus on
4. [ ] User can request Level 3 hint and receive specific cell coordinate with correct value
5. [ ] User can request Level 4 hint and receive partial solution (multiple cells)
6. [ ] Users can progress through hint levels at their own pace (no forced escalation)
7. [ ] Hint generation is contextually aware—hints change based on current puzzle state
8. [ ] Requesting same hint level multiple times yields different hints (no repetition)
9. [ ] No console errors during hint request/display flow
10. [ ] Existing puzzle functionality still works (no regressions)
11. [ ] Hint UI follows existing pips-solver design language and UX patterns
12. [ ] Edge cases handled gracefully (solved puzzle, invalid state, network errors)

## QA Acceptance Criteria

**CRITICAL**: These criteria must be verified by the QA Agent before sign-off.

### Unit Tests
| Test | File | What to Verify |
|------|------|----------------|
| Hint Level 1 Generation | `pips-agent/tests/test_hint_generator.py` | Given puzzle state, Level 1 returns strategic guidance without specific details |
| Hint Level 2 Generation | `pips-agent/tests/test_hint_generator.py` | Given puzzle state, Level 2 identifies specific region/constraint |
| Hint Level 3 Generation | `pips-agent/tests/test_hint_generator.py` | Given puzzle state, Level 3 returns valid cell coordinate with correct value |
| Hint Level 4 Generation | `pips-agent/tests/test_hint_generator.py` | Given puzzle state, Level 4 returns multiple valid cells (partial solution) |
| Puzzle State Validation | `pips-agent/tests/test_hint_generator.py` | Invalid puzzle states are detected and handled appropriately |
| Hint Component Rendering | `pips-solver/src/__tests__/HintComponent.test.tsx` | Hint modal renders correctly with all 4 level buttons |
| Hint State Management | `pips-solver/src/__tests__/HintComponent.test.tsx` | Hint level progression tracked correctly, user can control pace |

### Integration Tests
| Test | Services | What to Verify |
|------|----------|----------------|
| Hint API Request/Response | pips-solver ↔ pips-agent | Frontend sends puzzle state, backend returns hint at requested level |
| Hint API Error Handling | pips-solver ↔ pips-agent | Network failures handled gracefully with retry/error messages |
| Context-Aware Hints | pips-solver ↔ pips-agent | Same puzzle with different progress yields different contextual hints |

### End-to-End Tests
| Flow | Steps | Expected Outcome |
|------|-------|------------------|
| First-Time Hint Request | 1. Open puzzle 2. Tap "Hint" button 3. Request Level 1 | Hint modal opens, Level 1 strategy hint displayed |
| Progressive Hint Usage | 1. Request Level 1 2. Request Level 2 3. Request Level 3 4. Request Level 4 | Each level reveals progressively more specific information |
| Repeat Hint Request | 1. Request Level 2 hint 2. Close modal 3. Reopen and request Level 2 again | Second Level 2 hint differs from first (no repetition) |
| Edge Case: Solved Puzzle | 1. Complete puzzle 2. Request hint | "Puzzle already solved" message shown |
| Edge Case: Invalid State | 1. Create constraint violation 2. Request Level 3 hint | Error message guiding user to fix errors, or Level 1-2 helps identify issue |

### Browser Verification (if frontend)
| Page/Component | URL | Checks |
|----------------|-----|--------|
| Puzzle Interface with Hint Button | `http://localhost:3000/puzzle` | "Hint" button visible and accessible in puzzle UI |
| Hint Modal - Level 1 | `http://localhost:3000/puzzle` | Clicking Level 1 displays strategic guidance text |
| Hint Modal - Level 2 | `http://localhost:3000/puzzle` | Clicking Level 2 displays region/constraint direction |
| Hint Modal - Level 3 | `http://localhost:3000/puzzle` | Clicking Level 3 displays cell coordinate with value |
| Hint Modal - Level 4 | `http://localhost:3000/puzzle` | Clicking Level 4 displays partial solution (multiple cells) |
| Hint Loading State | `http://localhost:3000/puzzle` | Loading indicator appears while hint generates |
| Hint Error State | `http://localhost:3000/puzzle` (simulate network error) | Error message displays with retry option |

### Database Verification (if applicable)
| Check | Query/Command | Expected |
|-------|---------------|----------|
| N/A | N/A | Hint system does not require persistent database storage (state tracked in-memory/AsyncStorage) |

### QA Sign-off Requirements
- [ ] All unit tests pass for hint generation logic (all 4 levels)
- [ ] All unit tests pass for hint UI component
- [ ] All integration tests pass (frontend-backend hint API)
- [ ] All E2E tests pass (user flows for all 4 hint levels)
- [ ] Browser verification complete: all hint levels functional in UI
- [ ] Edge cases verified: solved puzzle, invalid state, network errors handled
- [ ] No regressions in existing puzzle functionality (solving, resetting, navigation)
- [ ] Code follows established patterns in pips-solver and pips-agent
- [ ] No security vulnerabilities introduced (hint generation doesn't expose sensitive data)
- [ ] Performance acceptable: hint generation completes within 3 seconds
- [ ] Hint content quality verified: Level 1 is strategic, Level 2 is directional, Level 3 is specific, Level 4 is helpful
- [ ] User control verified: no automatic hint escalation, user controls pace
