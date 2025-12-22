/**
 * Theme System - Main Export
 */

// Core tokens
export {
  theme,
  colors,
  typography,
  spacing,
  radii,
  shadows,
  animation,
  grid,
  breakpoints,
  zIndex,
  confidence,
} from './tokens';

export type { Theme, Colors, Typography } from './tokens';

// Context and hooks
export {
  ThemeProvider,
  useTheme,
  useConfidenceColor,
  useRegionColor,
} from './ThemeContext';

// Styled utilities
export { createStyles, styled, commonStyles } from './styled';

// Typography
export {
  useAppFonts,
  fontFamilies,
  textStyles,
} from './fonts';

export type { TextStyleName } from './fonts';
