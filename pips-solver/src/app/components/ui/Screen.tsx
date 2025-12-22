/**
 * Screen Component
 *
 * Base layout wrapper for all screens with consistent theming
 * Handles safe area, background, and scroll behavior
 */

import React from 'react';
import {
  View,
  ScrollView,
  StyleSheet,
  ViewStyle,
  StatusBar,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { colors, spacing } from '../../../theme';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface ScreenProps {
  children: React.ReactNode;
  scrollable?: boolean;
  padded?: boolean;
  edges?: ('top' | 'bottom' | 'left' | 'right')[];
  style?: ViewStyle;
  contentStyle?: ViewStyle;
}

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

export function Screen({
  children,
  scrollable = false,
  padded = true,
  edges = ['top', 'bottom'],
  style,
  contentStyle,
}: ScreenProps) {
  const paddingStyle = padded ? styles.padded : undefined;

  const content = scrollable ? (
    <ScrollView
      style={styles.scrollView}
      contentContainerStyle={[styles.scrollContent, paddingStyle, contentStyle]}
      showsVerticalScrollIndicator={false}
      keyboardShouldPersistTaps="handled"
    >
      {children}
    </ScrollView>
  ) : (
    <View style={[styles.content, paddingStyle, contentStyle]}>{children}</View>
  );

  return (
    <SafeAreaView style={[styles.container, style]} edges={edges}>
      <StatusBar barStyle="light-content" backgroundColor={colors.surface.obsidian} />
      {content}
    </SafeAreaView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.surface.obsidian,
  },
  content: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  padded: {
    paddingHorizontal: spacing[4],
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Header Component
// ════════════════════════════════════════════════════════════════════════════

interface ScreenHeaderProps {
  title?: string;
  subtitle?: string;
  leftAction?: React.ReactNode;
  rightAction?: React.ReactNode;
  style?: ViewStyle;
}

export function ScreenHeader({
  title,
  subtitle,
  leftAction,
  rightAction,
  style,
}: ScreenHeaderProps) {
  return (
    <View style={[headerStyles.container, style]}>
      <View style={headerStyles.leftAction}>{leftAction}</View>
      <View style={headerStyles.titleContainer}>
        {title && (
          <View style={headerStyles.titleWrapper}>
            {/* Title handled by parent */}
          </View>
        )}
      </View>
      <View style={headerStyles.rightAction}>{rightAction}</View>
    </View>
  );
}

const headerStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: spacing[3],
    paddingHorizontal: spacing[4],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  leftAction: {
    minWidth: 44,
    alignItems: 'flex-start',
  },
  titleContainer: {
    flex: 1,
    alignItems: 'center',
  },
  titleWrapper: {
    alignItems: 'center',
  },
  rightAction: {
    minWidth: 44,
    alignItems: 'flex-end',
  },
});

export default Screen;
