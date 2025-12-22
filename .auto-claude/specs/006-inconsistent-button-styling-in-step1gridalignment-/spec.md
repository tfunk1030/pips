# Inconsistent button styling in Step1GridAlignment uses hardcoded colors

## Overview

The Step1GridAlignment screen uses hardcoded color values (#007AFF, #9C27B0, #333) instead of the established design system tokens from theme/tokens.ts. This creates visual inconsistency with the 'Tactile Game Table' brass accent theme used throughout the rest of the app.

## Rationale

Consistent use of design tokens ensures visual cohesion, easier theming, and reduces maintenance burden. The current implementation mixes iOS system blue (#007AFF) with a purple AI button, which clashes with the warm brass/copper palette defined in the design system.

---
*This spec was created from ideation and is pending detailed specification.*
