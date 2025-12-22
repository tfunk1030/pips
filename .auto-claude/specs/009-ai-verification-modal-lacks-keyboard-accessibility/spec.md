# AI Verification Modal lacks keyboard accessibility for constraint editing

## Overview

The AIVerificationModal constraint editing panel uses TextInput for numeric values but the keyboard type is 'number-pad' which lacks a 'done' button on iOS. Additionally, the edit panels don't support keyboard dismissal on background tap.

## Rationale

Users editing constraint values on iOS see a number pad without a dismiss button, requiring awkward interaction. Good form UX should allow easy keyboard dismissal and navigation between inputs.

---
*This spec was created from ideation and is pending detailed specification.*
