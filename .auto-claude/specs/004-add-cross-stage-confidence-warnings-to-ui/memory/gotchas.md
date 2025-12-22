# Gotchas & Pitfalls

Things to watch out for in this codebase.

## [2025-12-22 08:33]
React Native does not support 'region' as an accessibilityRole. Use the 'accessible' prop instead of accessibilityRole="region" for container elements.

_Context: Found in ConfidenceSummary.tsx when trying to use accessibilityRole="region" - TypeScript will error as it's not a valid accessibility role in React Native._
