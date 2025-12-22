/**
 * Extraction Comparison Modal
 * Shows side-by-side comparison of extraction results from different AI models.
 * Enables users to review per-model results and see disagreements highlighted.
 */

import React, { useState, useMemo, useCallback } from 'react';
import { Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import type {
  RawResponses,
  BoardModelResponse,
  DominoModelResponse,
} from '../../model/overlayTypes';
import {
  compareCellDetections,
  type ComparisonResult,
  type DisagreementSummary,
  type NormalizedModelResult,
} from '../../services/extraction/validation/gridValidator';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

interface Props {
  /** Whether the modal is visible */
  visible: boolean;
  /** Raw responses from all models (required for comparison) */
  rawResponses: RawResponses | null;
  /** Called when user closes the modal without accepting */
  onClose: () => void;
  /** Called when user accepts the current consensus result */
  onAccept: () => void;
}

/** Tab types for the comparison view */
type TabType = 'summary' | string; // 'summary' or model name

// ════════════════════════════════════════════════════════════════════════════
// Subcomponents
// ════════════════════════════════════════════════════════════════════════════

/**
 * Severity badge showing count of issues by severity level
 */
function SeverityBadge({
  severity,
  count
}: {
  severity: 'critical' | 'warning' | 'info';
  count: number;
}) {
  if (count === 0) return null;

  const colors = {
    critical: { bg: '#d32f2f', text: '#fff' },
    warning: { bg: '#f9a825', text: '#000' },
    info: { bg: '#1976d2', text: '#fff' },
  };

  return (
    <View style={[styles.badge, { backgroundColor: colors[severity].bg }]}>
      <Text style={[styles.badgeText, { color: colors[severity].text }]}>
        {count}
      </Text>
    </View>
  );
}

/**
 * Tab bar showing available models and summary tab
 */
function TabBar({
  tabs,
  selectedTab,
  onSelectTab,
  modelSummaries,
}: {
  tabs: TabType[];
  selectedTab: TabType;
  onSelectTab: (tab: TabType) => void;
  modelSummaries: Record<string, { success: boolean; hasIssues: boolean }>;
}) {
  return (
    <View style={styles.tabBar}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        {tabs.map((tab) => {
          const isSelected = tab === selectedTab;
          const isSummaryTab = tab === 'summary';
          const modelInfo = !isSummaryTab ? modelSummaries[tab] : null;

          return (
            <TouchableOpacity
              key={tab}
              style={[styles.tab, isSelected && styles.tabSelected]}
              onPress={() => onSelectTab(tab)}
            >
              <Text
                style={[styles.tabText, isSelected && styles.tabTextSelected]}
                numberOfLines={1}
              >
                {isSummaryTab ? 'Summary' : formatModelName(tab)}
              </Text>
              {modelInfo && !modelInfo.success && (
                <View style={styles.errorDot} />
              )}
            </TouchableOpacity>
          );
        })}
      </ScrollView>
    </View>
  );
}

/**
 * Summary view showing overall comparison statistics
 */
function SummaryView({
  comparison,
}: {
  comparison: ComparisonResult;
}) {
  const { summary, modelsCompared, isUnanimous } = comparison;

  return (
    <View style={styles.summaryContainer}>
      {/* Status Banner */}
      <View style={[
        styles.statusBanner,
        isUnanimous ? styles.statusBannerSuccess : styles.statusBannerWarning
      ]}>
        <Text style={styles.statusText}>
          {isUnanimous
            ? '✓ All models agree'
            : `⚠ ${summary.total} disagreement${summary.total !== 1 ? 's' : ''} found`}
        </Text>
      </View>

      {/* Models Compared */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Models Compared</Text>
        <View style={styles.modelList}>
          {modelsCompared.map((model) => (
            <View key={model} style={styles.modelChip}>
              <Text style={styles.modelChipText}>{formatModelName(model)}</Text>
            </View>
          ))}
        </View>
      </View>

      {/* Disagreement Summary */}
      {!isUnanimous && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Disagreements by Severity</Text>
          <View style={styles.severityRow}>
            <SeveritySummaryItem label="Critical" count={summary.critical} severity="critical" />
            <SeveritySummaryItem label="Warning" count={summary.warning} severity="warning" />
            <SeveritySummaryItem label="Info" count={summary.info} severity="info" />
          </View>
        </View>
      )}

      {/* Disagreement Types Summary */}
      {!isUnanimous && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>By Category</Text>
          <DisagreementCategorySummary comparison={comparison} />
        </View>
      )}
    </View>
  );
}

/**
 * Individual severity summary item
 */
function SeveritySummaryItem({
  label,
  count,
  severity,
}: {
  label: string;
  count: number;
  severity: 'critical' | 'warning' | 'info';
}) {
  const colors = {
    critical: '#d32f2f',
    warning: '#f9a825',
    info: '#1976d2',
  };

  return (
    <View style={styles.severityItem}>
      <View style={[styles.severityDot, { backgroundColor: colors[severity] }]} />
      <Text style={styles.severityLabel}>{label}</Text>
      <Text style={styles.severityCount}>{count}</Text>
    </View>
  );
}

/**
 * Summary of disagreements by category
 */
function DisagreementCategorySummary({ comparison }: { comparison: ComparisonResult }) {
  const { disagreementsByType } = comparison;

  const categories = [
    { label: 'Grid Dimensions', count: disagreementsByType.gridDimensions.length },
    { label: 'Hole Positions', count: disagreementsByType.holePositions.length },
    { label: 'Region Assignments', count: disagreementsByType.regionAssignments.length },
    { label: 'Constraints', count: disagreementsByType.constraints.length },
    { label: 'Dominoes', count: disagreementsByType.dominoes.length },
  ].filter((c) => c.count > 0);

  return (
    <View style={styles.categoryList}>
      {categories.map((cat) => (
        <View key={cat.label} style={styles.categoryItem}>
          <Text style={styles.categoryLabel}>{cat.label}</Text>
          <Text style={styles.categoryCount}>{cat.count}</Text>
        </View>
      ))}
    </View>
  );
}

/**
 * Model-specific result view (placeholder for subtask 3.2)
 */
function ModelResultView({
  modelResult,
}: {
  modelResult: NormalizedModelResult;
}) {
  if (!modelResult.success) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorTitle}>Extraction Failed</Text>
        <Text style={styles.errorText}>{modelResult.error || 'Unknown error'}</Text>
      </View>
    );
  }

  // Placeholder for detailed model result display (subtask 3.2)
  return (
    <View style={styles.modelResultContainer}>
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Grid Dimensions</Text>
        <Text style={styles.text}>
          {modelResult.dimensions?.rows || '?'} rows × {modelResult.dimensions?.cols || '?'} columns
        </Text>
      </View>

      {modelResult.dominoes && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Dominoes ({modelResult.dominoes.length})</Text>
          <View style={styles.dominoContainer}>
            {modelResult.dominoes.map((domino, i) => (
              <Text key={i} style={styles.domino}>
                [{domino[0]},{domino[1]}]
              </Text>
            ))}
          </View>
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.textSmall}>
          Full model result display coming in subtask 3.2
        </Text>
      </View>
    </View>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * Format model name for display (shorten common prefixes)
 */
function formatModelName(model: string): string {
  // Common mappings for cleaner display
  const mappings: Record<string, string> = {
    'gemini-2.0-flash-exp': 'Gemini Flash',
    'gemini-1.5-flash': 'Gemini 1.5',
    'gemini-1.5-pro': 'Gemini Pro',
    'claude-sonnet-4-20250514': 'Claude Sonnet',
    'claude-3-5-sonnet-20241022': 'Claude 3.5',
    'claude-3-opus-20240229': 'Claude Opus',
    'gpt-4o': 'GPT-4o',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4-turbo': 'GPT-4 Turbo',
  };

  return mappings[model] || model.split('-').slice(0, 2).join(' ');
}

// ════════════════════════════════════════════════════════════════════════════
// Main Component
// ════════════════════════════════════════════════════════════════════════════

export default function ExtractionComparisonModal({
  visible,
  rawResponses,
  onClose,
  onAccept,
}: Props) {
  const [selectedTab, setSelectedTab] = useState<TabType>('summary');

  // Compute comparison result from raw responses
  const comparison = useMemo<ComparisonResult | null>(() => {
    if (!rawResponses) return null;

    return compareCellDetections(
      rawResponses.board,
      rawResponses.dominoes
    );
  }, [rawResponses]);

  // Build tab list: summary + one per model
  const tabs = useMemo<TabType[]>(() => {
    if (!comparison) return ['summary'];
    return ['summary', ...comparison.modelsCompared];
  }, [comparison]);

  // Model summaries for tab badges
  const modelSummaries = useMemo(() => {
    if (!comparison) return {};

    const summaries: Record<string, { success: boolean; hasIssues: boolean }> = {};
    for (const result of comparison.modelResults) {
      summaries[result.model] = {
        success: result.success,
        hasIssues: !result.success,
      };
    }
    return summaries;
  }, [comparison]);

  // Handle tab selection with reset on modal close
  const handleTabSelect = useCallback((tab: TabType) => {
    setSelectedTab(tab);
  }, []);

  // Get current model result if viewing a model tab
  const currentModelResult = useMemo(() => {
    if (selectedTab === 'summary' || !comparison) return null;
    return comparison.modelResults.find((r) => r.model === selectedTab) || null;
  }, [selectedTab, comparison]);

  // Early return if no data
  if (!rawResponses || !comparison) {
    return (
      <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
        <View style={styles.container}>
          <View style={styles.header}>
            <Text style={styles.title}>Extraction Comparison</Text>
            <Text style={styles.subtitle}>No comparison data available</Text>
          </View>
          <View style={styles.emptyState}>
            <Text style={styles.emptyText}>
              Debug responses were not captured for this extraction.
            </Text>
            <Text style={styles.emptyTextSmall}>
              Enable "Save Debug Responses" in settings to compare model outputs.
            </Text>
          </View>
          <View style={styles.buttons}>
            <TouchableOpacity style={[styles.button, styles.closeButton]} onPress={onClose}>
              <Text style={styles.buttonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    );
  }

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerTop}>
            <Text style={styles.title}>Extraction Comparison</Text>
            <View style={styles.headerBadges}>
              <SeverityBadge severity="critical" count={comparison.summary.critical} />
              <SeverityBadge severity="warning" count={comparison.summary.warning} />
              <SeverityBadge severity="info" count={comparison.summary.info} />
            </View>
          </View>
          <Text style={styles.subtitle}>
            {comparison.modelsCompared.length} model{comparison.modelsCompared.length !== 1 ? 's' : ''} compared
          </Text>
        </View>

        {/* Tab Bar */}
        <TabBar
          tabs={tabs}
          selectedTab={selectedTab}
          onSelectTab={handleTabSelect}
          modelSummaries={modelSummaries}
        />

        {/* Scrollable Content */}
        <ScrollView
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
        >
          {selectedTab === 'summary' ? (
            <SummaryView comparison={comparison} />
          ) : currentModelResult ? (
            <ModelResultView modelResult={currentModelResult} />
          ) : (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>No data for this model</Text>
            </View>
          )}
        </ScrollView>

        {/* Footer Buttons */}
        <View style={styles.buttons}>
          <TouchableOpacity style={[styles.button, styles.closeButton]} onPress={onClose}>
            <Text style={styles.buttonText}>Close</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.acceptButton]} onPress={onAccept}>
            <Text style={styles.buttonText}>Accept Consensus</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  header: {
    padding: 20,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerBadges: {
    flexDirection: 'row',
    gap: 4,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
  },

  // Tab Bar
  tabBar: {
    borderBottomWidth: 1,
    borderBottomColor: '#333',
    backgroundColor: '#1a1a1a',
  },
  tab: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 2,
    borderBottomColor: 'transparent',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  tabSelected: {
    borderBottomColor: '#4CAF50',
  },
  tabText: {
    fontSize: 14,
    color: '#888',
  },
  tabTextSelected: {
    color: '#fff',
    fontWeight: '600',
  },
  errorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#d32f2f',
  },

  // Badges
  badge: {
    minWidth: 20,
    height: 20,
    borderRadius: 10,
    paddingHorizontal: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  badgeText: {
    fontSize: 11,
    fontWeight: 'bold',
  },

  // Content Area
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
  },

  // Summary View
  summaryContainer: {
    gap: 20,
  },
  statusBanner: {
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  statusBannerSuccess: {
    backgroundColor: 'rgba(76, 175, 80, 0.2)',
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  statusBannerWarning: {
    backgroundColor: 'rgba(249, 168, 37, 0.2)',
    borderWidth: 1,
    borderColor: '#f9a825',
  },
  statusText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },

  // Sections
  section: {
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  text: {
    fontSize: 14,
    color: '#ccc',
    marginBottom: 4,
  },
  textSmall: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
    fontStyle: 'italic',
  },

  // Model List
  modelList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  modelChip: {
    backgroundColor: '#333',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  modelChipText: {
    fontSize: 13,
    color: '#ccc',
  },

  // Severity Row
  severityRow: {
    flexDirection: 'row',
    gap: 16,
  },
  severityItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  severityDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  severityLabel: {
    fontSize: 13,
    color: '#ccc',
  },
  severityCount: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#fff',
  },

  // Category List
  categoryList: {
    gap: 8,
  },
  categoryItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
  categoryLabel: {
    fontSize: 14,
    color: '#ccc',
  },
  categoryCount: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#fff',
  },

  // Model Result View
  modelResultContainer: {
    gap: 20,
  },
  dominoContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  domino: {
    fontSize: 14,
    color: '#ccc',
    backgroundColor: '#333',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    fontFamily: 'Courier',
  },

  // Error States
  errorContainer: {
    padding: 20,
    backgroundColor: 'rgba(211, 47, 47, 0.1)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#d32f2f',
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#d32f2f',
    marginBottom: 8,
  },
  errorText: {
    fontSize: 14,
    color: '#f88',
  },

  // Empty States
  emptyState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 16,
    color: '#888',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptyTextSmall: {
    fontSize: 13,
    color: '#666',
    textAlign: 'center',
  },

  // Buttons
  buttons: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  button: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  closeButton: {
    backgroundColor: '#666',
  },
  acceptButton: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
