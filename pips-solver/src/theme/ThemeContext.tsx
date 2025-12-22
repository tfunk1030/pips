/**
 * Theme Context Provider
 * Provides design tokens throughout the app via React Context
 */

import React, { createContext, useContext, ReactNode } from 'react';
import { theme, Theme } from './tokens';

// ════════════════════════════════════════════════════════════════════════════
// Context
// ════════════════════════════════════════════════════════════════════════════

const ThemeContext = createContext<Theme>(theme);

// ════════════════════════════════════════════════════════════════════════════
// Provider
// ════════════════════════════════════════════════════════════════════════════

interface ThemeProviderProps {
  children: ReactNode;
}

export function ThemeProvider({ children }: ThemeProviderProps) {
  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Hook
// ════════════════════════════════════════════════════════════════════════════

export function useTheme(): Theme {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}

// ════════════════════════════════════════════════════════════════════════════
// Utility Hooks
// ════════════════════════════════════════════════════════════════════════════

/**
 * Get color for confidence level
 */
export function useConfidenceColor(confidence: number): string {
  const { colors, confidence: thresholds } = useTheme();

  if (confidence >= thresholds.high) {
    return colors.semantic.jade;
  } else if (confidence >= thresholds.medium) {
    return colors.semantic.amber;
  }
  return colors.semantic.coral;
}

/**
 * Get region color by index (cycles through palette)
 */
export function useRegionColor(index: number): string {
  const { colors } = useTheme();
  return colors.regions[index % colors.regions.length];
}

// Re-export theme for direct access
export { theme } from './tokens';
