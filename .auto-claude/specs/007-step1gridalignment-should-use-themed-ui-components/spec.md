# Step1GridAlignment should use themed UI components

## Overview

The Step1GridAlignment screen manually creates buttons using TouchableOpacity with inline StyleSheet styles instead of leveraging the established Button, Card, and Text components from the UI library. This duplicates styling code and creates inconsistency.

## Rationale

The app has a well-designed component library (Button, Card, Text, Label, Heading, etc.) with built-in animations, consistent styling, and accessibility features. Using raw primitives bypasses these benefits.

---
*This spec was created from ideation and is pending detailed specification.*
