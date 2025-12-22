import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ViewStyle, LayoutAnimation, Platform, UIManager } from 'react-native';
import Svg, { Path } from 'react-native-svg';
import ConfidenceIndicator from './ConfidenceIndicator';
import ConfidenceHintsList from './ConfidenceHintsList';
import { ConfidenceHint } from '../../../extraction/types';

// Enable LayoutAnimation for Android
if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

interface Props {
  /** Overall confidence value from 0 to 1 */
  confidence: number;
  /** Array of confidence hints to display */
  hints: ConfidenceHint[];
  /** Whether the hints section is initially expanded */
  initiallyExpanded?: boolean;
  /** Additional container styles */
  style?: ViewStyle;
}

/**
 * Composite component combining overall confidence indicator with hints list.
 *
 * Features:
 * - Card-style container with shadow and rounded corners
 * - Shows 'No warnings' state when hints array is empty
 * - Expandable/collapsible hints section for mobile ergonomics
 * - Animated transitions for expand/collapse
 */
export default function ConfidenceSummary({
  confidence,
  hints,
  initiallyExpanded = false,
  style,
}: Props) {
  const [expanded, setExpanded] = useState(initiallyExpanded);
  const hasWarnings = hints.length > 0;

  const toggleExpanded = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpanded(!expanded);
  };

  const warningCount = hints.length;
  const warningText = warningCount === 1 ? '1 warning' : `${warningCount} warnings`;

  return (
    <View
      style={[styles.card, style]}
      accessible
      accessibilityLabel={`Extraction confidence summary. ${Math.round(confidence * 100)} percent confidence with ${warningCount} warnings.`}
    >
      {/* Confidence Indicator */}
      <ConfidenceIndicator confidence={confidence} />

      {/* Divider */}
      <View style={styles.divider} />

      {/* Warnings Section Header */}
      <TouchableOpacity
        style={styles.warningsHeader}
        onPress={toggleExpanded}
        disabled={!hasWarnings}
        accessibilityRole="button"
        accessibilityLabel={
          hasWarnings
            ? `${warningText}. ${expanded ? 'Collapse' : 'Expand'} to ${expanded ? 'hide' : 'show'} details.`
            : 'No warnings'
        }
        accessibilityState={{ expanded: hasWarnings ? expanded : undefined }}
        accessibilityHint={hasWarnings ? 'Double tap to toggle warning details' : undefined}
      >
        <View style={styles.warningsHeaderContent}>
          {hasWarnings ? (
            <WarningBadge count={warningCount} />
          ) : (
            <CheckIcon />
          )}
          <Text style={[styles.warningsHeaderText, !hasWarnings && styles.noWarningsText]}>
            {hasWarnings ? warningText : 'No warnings'}
          </Text>
        </View>
        {hasWarnings && (
          <View style={styles.expandIconContainer}>
            <ExpandIcon expanded={expanded} />
          </View>
        )}
      </TouchableOpacity>

      {/* Expandable Hints List */}
      {hasWarnings && expanded && (
        <View style={styles.hintsContainer}>
          <ConfidenceHintsList hints={hints} />
        </View>
      )}
    </View>
  );
}

/**
 * Warning count badge component.
 */
function WarningBadge({ count }: { count: number }) {
  return (
    <View style={styles.warningBadge}>
      <Text style={styles.warningBadgeText}>{count}</Text>
    </View>
  );
}

/**
 * Checkmark icon for no warnings state.
 */
function CheckIcon() {
  return (
    <View style={styles.checkIconContainer}>
      <Svg width={16} height={16} viewBox="0 0 24 24" fill="none">
        <Path
          d="M20 6L9 17l-5-5"
          stroke="#22c55e"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </Svg>
    </View>
  );
}

/**
 * Expand/collapse chevron icon.
 */
function ExpandIcon({ expanded }: { expanded: boolean }) {
  return (
    <Svg width={20} height={20} viewBox="0 0 24 24" fill="none">
      <Path
        d={expanded ? 'M18 15l-6-6-6 6' : 'M6 9l6 6 6-6'}
        stroke="#9aa5ce"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </Svg>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: '#1a1b26',
    borderRadius: 12,
    padding: 16,
    // Shadow for iOS
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    // Elevation for Android
    elevation: 8,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginVertical: 16,
  },
  warningsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    minHeight: 32,
  },
  warningsHeaderContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  warningsHeaderText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#e6e6e6',
  },
  noWarningsText: {
    color: '#22c55e',
  },
  warningBadge: {
    backgroundColor: '#dc2626',
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    paddingHorizontal: 6,
    justifyContent: 'center',
    alignItems: 'center',
  },
  warningBadgeText: {
    color: '#ffffff',
    fontSize: 12,
    fontWeight: 'bold',
  },
  checkIconContainer: {
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  expandIconContainer: {
    padding: 4,
  },
  hintsContainer: {
    marginTop: 16,
  },
});
