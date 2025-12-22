/**
 * Haptics utility module
 * Provides app-specific haptic feedback patterns for tactile interactions
 * Wraps expo-haptics with graceful fallback for unsupported platforms
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * HAPTIC PATTERN REFERENCE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * This module uses three categories of haptic feedback from expo-haptics:
 *
 * 1. IMPACT FEEDBACK (impactAsync) - Single physical tap sensations
 *    - Light:  Subtle tap, ~10ms vibration  → Button presses, drag start
 *    - Medium: Moderate tap, ~20ms vibration → Cell taps, toggle actions
 *    - Heavy:  Strong tap, ~30ms vibration  → Important state changes
 *
 * 2. SELECTION FEEDBACK (selectionAsync) - UI selection confirmation
 *    - Single crisp tick sensation → Picker changes, list selections
 *    - Lighter than Light impact, but more "clicky"
 *
 * 3. NOTIFICATION FEEDBACK (notificationAsync) - Multi-pulse patterns
 *    - Success: Two quick pulses (positive)   → Task completion
 *    - Warning: Three short pulses (caution)  → Attention needed
 *    - Error:   Three strong pulses (negative) → Action failed
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * INTERACTION → HAPTIC MAPPING
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Grid Alignment (Step 1):
 *   • Edge drag start    → Light Impact    (triggerEdgeDragStart)
 *   • Edge drag end      → Selection       (triggerEdgeDragEnd)
 *   • Cell tap (hole)    → Medium Impact   (triggerCellTap)
 *   • Row/col +/- button → Light Impact    (triggerControlButton)
 *
 * Region Painting (Step 2):
 *   • Paint start        → Light Impact    (triggerImpactLight)
 *   • Enter new cell     → Selection       (triggerPaintCell)
 *   • Palette selection  → Selection       (triggerPaletteSelect)
 *
 * General Feedback:
 *   • Success action     → Notification Success (triggerSuccess)
 *   • Warning condition  → Notification Warning (triggerWarning)
 *   • Error condition    → Notification Error   (triggerError)
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import * as Haptics from 'expo-haptics';
import { Platform } from 'react-native';

/**
 * Check if haptics are available on the current platform
 * Returns false for web platform where haptics may not be supported
 */
const isHapticsAvailable = (): boolean => {
  // Haptics are fully supported on iOS and Android
  // Web support is limited and may not work on all browsers/devices
  return Platform.OS === 'ios' || Platform.OS === 'android';
};

/**
 * Safely execute a haptic function with error handling
 * Silently fails if haptics are unavailable or error occurs
 */
const safeHaptic = async (hapticFn: () => Promise<void>): Promise<void> => {
  if (!isHapticsAvailable()) {
    return;
  }

  try {
    await hapticFn();
  } catch {
    // Silently ignore haptic errors (e.g., Low Power Mode on iOS)
  }
};

/**
 * Trigger selection haptic feedback
 * Use for: UI selections, picker changes, toggle switches
 * Feel: Light, crisp "tick" sensation - lighter than Light impact but more tactile
 *
 * @example
 * // When user selects an item from a list
 * onItemSelect={() => triggerSelection()}
 */
export const triggerSelection = (): void => {
  // Uses Haptics.selectionAsync() - designed for picker/selection UI feedback
  safeHaptic(() => Haptics.selectionAsync());
};

/**
 * Trigger light impact haptic feedback
 * Use for: Button presses, drag start, light touches
 * Feel: Subtle, gentle tap (~10ms vibration) - barely perceptible but present
 *
 * @example
 * // When user starts dragging an element
 * onDragStart={() => triggerImpactLight()}
 */
export const triggerImpactLight = (): void => {
  // Uses ImpactFeedbackStyle.Light - lowest intensity physical tap
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light));
};

/**
 * Trigger medium impact haptic feedback
 * Use for: Cell taps, toggle actions, confirmations
 * Feel: Moderate, noticeable tap (~20ms vibration) - clearly felt without being jarring
 *
 * @example
 * // When user taps a cell to toggle its state
 * onCellTap={() => triggerImpactMedium()}
 */
export const triggerImpactMedium = (): void => {
  // Uses ImpactFeedbackStyle.Medium - balanced intensity for primary interactions
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium));
};

/**
 * Trigger heavy impact haptic feedback
 * Use for: Important actions, significant state changes
 * Feel: Strong, pronounced tap (~30ms vibration) - demands attention
 *
 * @example
 * // When user confirms a destructive action
 * onConfirmDelete={() => triggerImpactHeavy()}
 */
export const triggerImpactHeavy = (): void => {
  // Uses ImpactFeedbackStyle.Heavy - highest intensity, use sparingly
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy));
};

/**
 * Trigger cell tap haptic feedback
 * Use for: Tapping grid cells to toggle hole state in Step1GridAlignment
 * Feel: Medium impact for clear tactile confirmation of state change
 *
 * Interaction: User taps a cell → cell toggles between normal/hole state
 * Pattern: Medium Impact (ImpactFeedbackStyle.Medium)
 * Rationale: Noticeable feedback confirms the toggle action registered
 */
export const triggerCellTap = (): void => {
  // Delegates to medium impact - provides clear "something changed" feedback
  triggerImpactMedium();
};

/**
 * Trigger edge drag start haptic feedback
 * Use for: When user begins dragging a grid edge in Step1GridAlignment
 * Feel: Light impact to indicate drag gesture has been recognized
 *
 * Interaction: User touches and begins dragging left/right/top/bottom edge
 * Pattern: Light Impact (ImpactFeedbackStyle.Light)
 * Rationale: Subtle feedback confirms drag started without being distracting during gesture
 */
export const triggerEdgeDragStart = (): void => {
  // Delegates to light impact - subtle "acknowledged" feedback
  triggerImpactLight();
};

/**
 * Trigger edge drag end haptic feedback
 * Use for: When user releases a grid edge after dragging in Step1GridAlignment
 * Feel: Selection haptic to confirm position is set
 *
 * Interaction: User releases finger after dragging an edge to new position
 * Pattern: Selection (selectionAsync)
 * Rationale: Crisp "click" confirms the edge position has been set
 */
export const triggerEdgeDragEnd = (): void => {
  // Delegates to selection - crisp "position set" feedback
  triggerSelection();
};

/**
 * Trigger edge drag haptic feedback (combined start action)
 * Alias for triggerEdgeDragStart for backward compatibility
 * @deprecated Use triggerEdgeDragStart and triggerEdgeDragEnd separately
 */
export const triggerEdgeDrag = (): void => {
  // Legacy alias - delegates to start haptic
  triggerEdgeDragStart();
};

/**
 * Trigger error haptic feedback
 * Use for: Invalid actions, validation failures, errors
 * Feel: Distinctive triple-pulse error pattern - unmistakably "wrong"
 *
 * Pattern: NotificationFeedbackType.Error
 * Physical: Three strong, quick vibration pulses
 * @example
 * // When form validation fails
 * if (!isValid) triggerError();
 */
export const triggerError = (): void => {
  // Uses notification error pattern - distinctive multi-pulse for negative feedback
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error)
  );
};

/**
 * Trigger success haptic feedback
 * Use for: Successful completions, confirmations
 * Feel: Positive double-pulse pattern - satisfying "done" feeling
 *
 * Pattern: NotificationFeedbackType.Success
 * Physical: Two quick, lighter vibration pulses
 * @example
 * // When puzzle is solved successfully
 * onSolved={() => triggerSuccess()}
 */
export const triggerSuccess = (): void => {
  // Uses notification success pattern - positive double-pulse for completion
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
  );
};

/**
 * Trigger warning haptic feedback
 * Use for: Caution indicators, potential issues
 * Feel: Attention-getting pattern - noticeable but not alarming
 *
 * Pattern: NotificationFeedbackType.Warning
 * Physical: Three short pulses (between success and error intensity)
 * @example
 * // When user is about to overwrite data
 * onOverwriteWarning={() => triggerWarning()}
 */
export const triggerWarning = (): void => {
  // Uses notification warning pattern - attention-getting pulse sequence
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning)
  );
};

/**
 * Trigger paint cell haptic feedback
 * Use for: When entering a new cell during region painting in Step2RegionPainting
 * Feel: Selection haptic for smooth, rapid painting feedback
 *
 * Interaction: User drags finger across cells, each new cell triggers this
 * Pattern: Selection (selectionAsync)
 * Rationale: Light "tick" for each cell painted allows rapid feedback without fatigue
 *
 * Note: This is called frequently during painting - selection haptic is
 * chosen for its lightweight feel that won't become overwhelming
 */
export const triggerPaintCell = (): void => {
  // Delegates to selection - light tick for each painted cell
  triggerSelection();
};

/**
 * Trigger palette selection haptic feedback
 * Use for: When selecting a color from the palette in Step2RegionPainting
 * Feel: Selection haptic to confirm color choice
 *
 * Interaction: User taps a color swatch in the region color palette
 * Pattern: Selection (selectionAsync)
 * Rationale: Crisp feedback confirms the color selection registered
 */
export const triggerPaletteSelect = (): void => {
  // Delegates to selection - crisp "selected" feedback
  triggerSelection();
};

/**
 * Trigger row/column control haptic feedback
 * Use for: +/- buttons for changing grid dimensions in Step1GridAlignment
 * Feel: Light impact for control activation
 *
 * Interaction: User taps +/- buttons to adjust row or column count
 * Pattern: Light Impact (ImpactFeedbackStyle.Light)
 * Rationale: Subtle feedback for repeated button presses (users may tap multiple times)
 */
export const triggerControlButton = (): void => {
  // Delegates to light impact - subtle feedback for repeated control presses
  triggerImpactLight();
};
