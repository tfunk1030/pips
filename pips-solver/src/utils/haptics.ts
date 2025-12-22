/**
 * Haptics utility module
 * Provides app-specific haptic feedback patterns for tactile interactions
 * Wraps expo-haptics with graceful fallback for unsupported platforms
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
 * Feel: Light, crisp tap
 */
export const triggerSelection = (): void => {
  safeHaptic(() => Haptics.selectionAsync());
};

/**
 * Trigger light impact haptic feedback
 * Use for: Button presses, drag start, light touches
 * Feel: Subtle, gentle tap
 */
export const triggerImpactLight = (): void => {
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light));
};

/**
 * Trigger medium impact haptic feedback
 * Use for: Cell taps, toggle actions, confirmations
 * Feel: Moderate, noticeable tap
 */
export const triggerImpactMedium = (): void => {
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium));
};

/**
 * Trigger heavy impact haptic feedback
 * Use for: Important actions, significant state changes
 * Feel: Strong, pronounced tap
 */
export const triggerImpactHeavy = (): void => {
  safeHaptic(() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy));
};

/**
 * Trigger cell tap haptic feedback
 * Use for: Tapping grid cells to toggle hole state
 * Feel: Medium impact for clear tactile confirmation
 */
export const triggerCellTap = (): void => {
  triggerImpactMedium();
};

/**
 * Trigger edge drag start haptic feedback
 * Use for: When user begins dragging a grid edge
 * Feel: Light impact to indicate drag has started
 */
export const triggerEdgeDragStart = (): void => {
  triggerImpactLight();
};

/**
 * Trigger edge drag end haptic feedback
 * Use for: When user releases a grid edge after dragging
 * Feel: Selection haptic to confirm position is set
 */
export const triggerEdgeDragEnd = (): void => {
  triggerSelection();
};

/**
 * Trigger edge drag haptic feedback (combined start action)
 * Alias for triggerEdgeDragStart for backward compatibility
 */
export const triggerEdgeDrag = (): void => {
  triggerEdgeDragStart();
};

/**
 * Trigger error haptic feedback
 * Use for: Invalid actions, validation failures, errors
 * Feel: Distinctive error pattern
 */
export const triggerError = (): void => {
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error)
  );
};

/**
 * Trigger success haptic feedback
 * Use for: Successful completions, confirmations
 * Feel: Positive, affirming pattern
 */
export const triggerSuccess = (): void => {
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success)
  );
};

/**
 * Trigger warning haptic feedback
 * Use for: Caution indicators, potential issues
 * Feel: Attention-getting pattern
 */
export const triggerWarning = (): void => {
  safeHaptic(() =>
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning)
  );
};

/**
 * Trigger paint cell haptic feedback
 * Use for: When entering a new cell during region painting
 * Feel: Selection haptic for smooth painting feedback
 */
export const triggerPaintCell = (): void => {
  triggerSelection();
};

/**
 * Trigger palette selection haptic feedback
 * Use for: When selecting a color from the palette
 * Feel: Selection haptic to confirm color choice
 */
export const triggerPaletteSelect = (): void => {
  triggerSelection();
};

/**
 * Trigger row/column control haptic feedback
 * Use for: +/- buttons for changing grid dimensions
 * Feel: Light impact for control activation
 */
export const triggerControlButton = (): void => {
  triggerImpactLight();
};
