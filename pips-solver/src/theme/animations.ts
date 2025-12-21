/**
 * Animation Utilities
 *
 * Reusable animation configurations using react-native-reanimated
 * For the "Tactile Game Table" aesthetic with smooth, playful motion
 */

import {
  Easing,
  SharedValue,
  WithSpringConfig,
  WithTimingConfig,
  withDelay,
  withSequence,
  withSpring,
  withTiming,
  interpolate,
  Extrapolate,
} from 'react-native-reanimated';
import { animation } from './tokens';

// ════════════════════════════════════════════════════════════════════════════
// Spring Configurations
// ════════════════════════════════════════════════════════════════════════════

export const springConfigs = {
  // Gentle spring for UI elements
  gentle: {
    damping: 20,
    stiffness: 200,
    mass: 1,
  } as WithSpringConfig,

  // Snappy spring for buttons and quick interactions
  snappy: {
    damping: 15,
    stiffness: 400,
    mass: 0.8,
  } as WithSpringConfig,

  // Bouncy spring for playful elements (solve celebration)
  bouncy: {
    damping: 8,
    stiffness: 300,
    mass: 1,
  } as WithSpringConfig,

  // Stiff spring for precise movements
  stiff: {
    damping: 25,
    stiffness: 500,
    mass: 1,
  } as WithSpringConfig,
};

// ════════════════════════════════════════════════════════════════════════════
// Timing Configurations
// ════════════════════════════════════════════════════════════════════════════

export const timingConfigs = {
  fast: {
    duration: animation.duration.fast,
    easing: Easing.out(Easing.cubic),
  } as WithTimingConfig,

  normal: {
    duration: animation.duration.normal,
    easing: Easing.out(Easing.cubic),
  } as WithTimingConfig,

  slow: {
    duration: animation.duration.slow,
    easing: Easing.out(Easing.cubic),
  } as WithTimingConfig,

  // Ease in-out for smooth transitions
  smooth: {
    duration: animation.duration.normal,
    easing: Easing.inOut(Easing.cubic),
  } as WithTimingConfig,

  // Linear for progress indicators
  linear: {
    duration: animation.duration.normal,
    easing: Easing.linear,
  } as WithTimingConfig,
};

// ════════════════════════════════════════════════════════════════════════════
// Staggered Animation Helpers
// ════════════════════════════════════════════════════════════════════════════

/**
 * Calculate stagger delay for an item in a list
 * @param index - Index of the item
 * @param staggerType - 'fast' | 'normal' | 'slow'
 */
export function getStaggerDelay(
  index: number,
  staggerType: keyof typeof animation.stagger = 'normal'
): number {
  return index * animation.stagger[staggerType];
}

/**
 * Create a staggered fade-in animation value
 */
export function staggeredFadeIn(
  index: number,
  staggerType: keyof typeof animation.stagger = 'normal'
) {
  return withDelay(
    getStaggerDelay(index, staggerType),
    withTiming(1, timingConfigs.normal)
  );
}

/**
 * Create a staggered slide-up animation value
 */
export function staggeredSlideUp(
  index: number,
  staggerType: keyof typeof animation.stagger = 'normal'
) {
  return withDelay(
    getStaggerDelay(index, staggerType),
    withSpring(0, springConfigs.gentle)
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Preset Animations
// ════════════════════════════════════════════════════════════════════════════

/**
 * Fade in animation
 */
export function fadeIn(opacity: SharedValue<number>) {
  opacity.value = withTiming(1, timingConfigs.normal);
}

/**
 * Fade out animation
 */
export function fadeOut(opacity: SharedValue<number>) {
  opacity.value = withTiming(0, timingConfigs.normal);
}

/**
 * Scale press animation (button press effect)
 */
export function scalePress(scale: SharedValue<number>) {
  scale.value = withTiming(0.96, timingConfigs.fast);
}

/**
 * Scale release animation (button release effect)
 */
export function scaleRelease(scale: SharedValue<number>) {
  scale.value = withSpring(1, springConfigs.snappy);
}

/**
 * Pulse animation (for highlights, success states)
 */
export function pulse(scale: SharedValue<number>) {
  scale.value = withSequence(
    withSpring(1.05, springConfigs.bouncy),
    withSpring(1, springConfigs.gentle)
  );
}

/**
 * Shake animation (for errors, invalid input)
 */
export function shake(translateX: SharedValue<number>) {
  translateX.value = withSequence(
    withTiming(-10, { duration: 50 }),
    withTiming(10, { duration: 50 }),
    withTiming(-8, { duration: 50 }),
    withTiming(8, { duration: 50 }),
    withTiming(-5, { duration: 50 }),
    withSpring(0, springConfigs.stiff)
  );
}

/**
 * Celebration animation (for solve success)
 * Returns an animation that bounces up and settles
 */
export function celebrate(scale: SharedValue<number>) {
  scale.value = withSequence(
    withSpring(1.15, springConfigs.bouncy),
    withSpring(0.95, springConfigs.bouncy),
    withSpring(1.05, springConfigs.bouncy),
    withSpring(1, springConfigs.gentle)
  );
}

/**
 * Domino deal animation
 * Simulates a domino being "dealt" onto the board
 */
export function dominoDeal(
  opacity: SharedValue<number>,
  translateY: SharedValue<number>,
  scale: SharedValue<number>,
  index: number
) {
  const delay = index * animation.stagger.normal;

  opacity.value = withDelay(delay, withTiming(1, timingConfigs.normal));
  translateY.value = withDelay(delay, withSpring(0, springConfigs.gentle));
  scale.value = withDelay(
    delay,
    withSequence(
      withSpring(1.05, springConfigs.bouncy),
      withSpring(1, springConfigs.gentle)
    )
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Interpolation Helpers
// ════════════════════════════════════════════════════════════════════════════

/**
 * Interpolate value for opacity fade
 */
export function interpolateOpacity(
  value: number,
  inputRange: [number, number] = [0, 1]
): number {
  return interpolate(value, inputRange, [0, 1], Extrapolate.CLAMP);
}

/**
 * Interpolate value for slide animation
 */
export function interpolateSlide(
  value: number,
  slideDistance: number = 20
): number {
  return interpolate(value, [0, 1], [slideDistance, 0], Extrapolate.CLAMP);
}

/**
 * Interpolate value for scale animation
 */
export function interpolateScale(
  value: number,
  fromScale: number = 0.9,
  toScale: number = 1
): number {
  return interpolate(value, [0, 1], [fromScale, toScale], Extrapolate.CLAMP);
}

// ════════════════════════════════════════════════════════════════════════════
// Animation Hooks
// ════════════════════════════════════════════════════════════════════════════

import { useCallback } from 'react';
import { useSharedValue } from 'react-native-reanimated';

/**
 * Hook for press animation state
 */
export function usePressAnimation() {
  const scale = useSharedValue(1);

  const onPressIn = useCallback(() => {
    scalePress(scale);
  }, [scale]);

  const onPressOut = useCallback(() => {
    scaleRelease(scale);
  }, [scale]);

  return { scale, onPressIn, onPressOut };
}

/**
 * Hook for entrance animation
 */
export function useEntranceAnimation(delay: number = 0) {
  const opacity = useSharedValue(0);
  const translateY = useSharedValue(20);

  const animate = useCallback(() => {
    opacity.value = withDelay(delay, withTiming(1, timingConfigs.normal));
    translateY.value = withDelay(delay, withSpring(0, springConfigs.gentle));
  }, [delay, opacity, translateY]);

  return { opacity, translateY, animate };
}
