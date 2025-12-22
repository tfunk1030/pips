/**
 * Extraction Comparison Modal
 * Shows side-by-side comparison of extraction results from different AI models.
 * Enables users to review per-model results and see disagreements highlighted.
 */

import React, { useState, useMemo, useCallback, useRef } from 'react';
import { Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View, Pressable } from 'react-native';
import type {
  RawResponses,
  BoardModelResponse,
  DominoModelResponse,
  ConstraintDef,
} from '../../model/overlayTypes';
import {
  compareCellDetections,
  cellKey,
  type ComparisonResult,
  type DisagreementSummary,
  type NormalizedModelResult,
  type CellDisagreement,
  type DisagreementSeverity,
  type CellCoordinate,
  type Disagreement,
  type GridDimensionDisagreement,
  type ConstraintDisagreement,
  type DominoDisagreement,
  sortDisagreementsBySeverity,
} from '../../services/extraction/validation/gridValidator';

// ════════════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════════════

/** Severity colors for disagreement highlighting */
const SEVERITY_COLORS: Record<DisagreementSeverity, { border: string; bg: string; text: string }> = {
  critical: { border: '#d32f2f', bg: 'rgba(211, 47, 47, 0.3)', text: '#ff6b6b' },
  warning: { border: '#f9a825', bg: 'rgba(249, 168, 37, 0.3)', text: '#ffd54f' },
  info: { border: '#1976d2', bg: 'rgba(25, 118, 210, 0.3)', text: '#64b5f6' },
};

/** Get the highest severity from a list of cell disagreements */
function getHighestCellSeverity(disagreements: CellDisagreement[]): DisagreementSeverity | null {
  if (disagreements.length === 0) return null;
  if (disagreements.some(d => d.severity === 'critical')) return 'critical';
  if (disagreements.some(d => d.severity === 'warning')) return 'warning';
  return 'info';
}

/** Color palette for regions matching DEFAULT_PALETTE in overlayTypes */
const REGION_COLORS: Record<string, string> = {
  'A': '#FF9800', // Orange
  'B': '#009688', // Teal
  'C': '#9C27B0', // Purple
  'D': '#E91E63', // Pink
  'E': '#4CAF50', // Green
  'F': '#2196F3', // Blue
  'G': '#FF5722', // Deep Orange
  'H': '#607D8B', // Blue Gray
  'I': '#795548', // Brown
  'J': '#00BCD4', // Cyan
  '.': '#444',    // Unlabeled
};

/** Get color for a region label */
function getRegionColor(region: string | null): string {
  if (!region || region === '.') return REGION_COLORS['.'];
  return REGION_COLORS[region] || '#888';
}

/** Timing data for a model response */
interface ModelTiming {
  responseMs: number;
  parseMs: number;
}

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

/** Severity filter state */
type SeverityFilters = Record<DisagreementSeverity, boolean>;

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
 * Cell detail popup showing what each model reported for a disagreeing cell
 */
function CellDetailPopup({
  coordinate,
  disagreements,
  onClose,
}: {
  coordinate: CellCoordinate;
  disagreements: CellDisagreement[];
  onClose: () => void;
}) {
  const severity = getHighestCellSeverity(disagreements);
  const severityColor = severity ? SEVERITY_COLORS[severity] : SEVERITY_COLORS.info;

  return (
    <Modal visible={true} transparent animationType="fade">
      <Pressable style={styles.popupOverlay} onPress={onClose}>
        <Pressable style={styles.popupContainer} onPress={e => e.stopPropagation()}>
          {/* Header */}
          <View style={[styles.popupHeader, { borderBottomColor: severityColor.border }]}>
            <View style={styles.popupHeaderContent}>
              <Text style={styles.popupTitle}>Cell ({coordinate.row}, {coordinate.col})</Text>
              {severity && (
                <View style={[styles.popupSeverityBadge, { backgroundColor: severityColor.bg }]}>
                  <Text style={[styles.popupSeverityText, { color: severityColor.text }]}>
                    {severity.toUpperCase()}
                  </Text>
                </View>
              )}
            </View>
            <TouchableOpacity style={styles.popupCloseButton} onPress={onClose}>
              <Text style={styles.popupCloseText}>×</Text>
            </TouchableOpacity>
          </View>

          {/* Content: List disagreements */}
          <ScrollView style={styles.popupContent}>
            {disagreements.map((disagreement, idx) => (
              <View key={disagreement.id || idx} style={styles.popupDisagreementItem}>
                <View style={[
                  styles.popupDisagreementHeader,
                  { borderLeftColor: SEVERITY_COLORS[disagreement.severity].border }
                ]}>
                  <Text style={styles.popupDisagreementType}>
                    {formatDisagreementType(disagreement.type)}
                  </Text>
                </View>

                {/* Per-model values */}
                <View style={styles.popupModelValues}>
                  {Object.entries(disagreement.detections).map(([model, detection]) => (
                    <View key={model} style={styles.popupModelRow}>
                      <Text style={styles.popupModelName}>{formatModelName(model)}</Text>
                      <Text style={styles.popupModelValue}>
                        {disagreement.type === 'hole_position'
                          ? (detection.isHole ? '# (hole)' : '· (cell)')
                          : (detection.region || 'null')}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
            ))}
          </ScrollView>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

/**
 * Format disagreement type for display
 */
function formatDisagreementType(type: string): string {
  const typeMap: Record<string, string> = {
    hole_position: 'Hole Position',
    region_assignment: 'Region Assignment',
    grid_dimensions: 'Grid Dimensions',
    constraint_type: 'Constraint Type',
    constraint_value: 'Constraint Value',
    constraint_operator: 'Constraint Operator',
    domino_count: 'Domino Count',
    domino_value: 'Domino Value',
  };
  return typeMap[type] || type;
}

/**
 * Severity filter toggle chips
 */
function SeverityFilterBar({
  filters,
  onToggle,
  summary,
}: {
  filters: SeverityFilters;
  onToggle: (severity: DisagreementSeverity) => void;
  summary: DisagreementSummary;
}) {
  const severities: DisagreementSeverity[] = ['critical', 'warning', 'info'];

  return (
    <View style={styles.filterBar}>
      <Text style={styles.filterLabel}>Show:</Text>
      {severities.map((severity) => {
        const count = summary[severity];
        const isActive = filters[severity];
        const colors = SEVERITY_COLORS[severity];

        return (
          <Pressable
            key={severity}
            style={[
              styles.filterChip,
              isActive && { backgroundColor: colors.bg, borderColor: colors.border },
              !isActive && styles.filterChipInactive,
            ]}
            onPress={() => onToggle(severity)}
          >
            <View style={[styles.filterDot, { backgroundColor: colors.border }]} />
            <Text style={[
              styles.filterChipText,
              isActive && { color: colors.text },
              !isActive && styles.filterChipTextInactive,
            ]}>
              {severity.charAt(0).toUpperCase() + severity.slice(1)} ({count})
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

/**
 * Format disagreement values for display based on type
 */
function formatDisagreementValues(disagreement: Disagreement): Record<string, string> {
  const formatted: Record<string, string> = {};

  if ('coordinate' in disagreement) {
    // CellDisagreement - format detections
    const cellDisagreement = disagreement as CellDisagreement;
    for (const [model, detection] of Object.entries(cellDisagreement.detections)) {
      if (disagreement.type === 'hole_position') {
        formatted[model] = detection.isHole ? '# (hole)' : '· (cell)';
      } else {
        formatted[model] = detection.region || 'null';
      }
    }
  } else if ('dimension' in disagreement) {
    // GridDimensionDisagreement
    const dimDisagreement = disagreement as GridDimensionDisagreement;
    for (const [model, value] of Object.entries(dimDisagreement.values)) {
      formatted[model] = String(value);
    }
  } else if ('region' in disagreement) {
    // ConstraintDisagreement
    const constDisagreement = disagreement as ConstraintDisagreement;
    for (const [model, constraint] of Object.entries(constDisagreement.values)) {
      if (constraint === null) {
        formatted[model] = 'none';
      } else if (disagreement.type === 'constraint_type') {
        formatted[model] = constraint.type || 'none';
      } else if (disagreement.type === 'constraint_value') {
        formatted[model] = constraint.value !== undefined ? String(constraint.value) : 'undefined';
      } else if (disagreement.type === 'constraint_operator') {
        formatted[model] = constraint.op || '==';
      }
    }
  } else {
    // DominoDisagreement
    const dominoDisagreement = disagreement as DominoDisagreement;
    for (const [model, value] of Object.entries(dominoDisagreement.values)) {
      if (typeof value === 'number') {
        formatted[model] = String(value);
      } else if (typeof value === 'boolean') {
        formatted[model] = value ? '✓ present' : '✗ missing';
      } else if (Array.isArray(value)) {
        formatted[model] = `${value.length} dominoes`;
      }
    }
  }

  return formatted;
}

/**
 * Get location info for a disagreement
 */
function getDisagreementLocation(disagreement: Disagreement): string | null {
  if ('coordinate' in disagreement) {
    const coord = (disagreement as CellDisagreement).coordinate;
    return `Cell (${coord.row}, ${coord.col})`;
  } else if ('region' in disagreement) {
    return `Region ${(disagreement as ConstraintDisagreement).region}`;
  } else if ('dimension' in disagreement) {
    return (disagreement as GridDimensionDisagreement).dimension === 'rows' ? 'Rows' : 'Columns';
  }
  return null;
}

/**
 * Single disagreement list item
 */
function DisagreementListItem({
  disagreement,
  onPress,
}: {
  disagreement: Disagreement;
  onPress?: () => void;
}) {
  const severityColor = SEVERITY_COLORS[disagreement.severity];
  const location = getDisagreementLocation(disagreement);
  const values = formatDisagreementValues(disagreement);
  const isNavigable = 'coordinate' in disagreement;

  const content = (
    <View style={[styles.disagreementItem, { borderLeftColor: severityColor.border }]}>
      {/* Header row with type and location */}
      <View style={styles.disagreementItemHeader}>
        <View style={styles.disagreementItemHeaderLeft}>
          <Text style={styles.disagreementItemType}>
            {formatDisagreementType(disagreement.type)}
          </Text>
          {location && (
            <Text style={styles.disagreementItemLocation}>{location}</Text>
          )}
        </View>
        <View style={[styles.disagreementItemSeverityBadge, { backgroundColor: severityColor.bg }]}>
          <Text style={[styles.disagreementItemSeverityText, { color: severityColor.text }]}>
            {disagreement.severity.toUpperCase()}
          </Text>
        </View>
      </View>

      {/* Model values */}
      <View style={styles.disagreementItemValues}>
        {Object.entries(values).map(([model, value]) => (
          <View key={model} style={styles.disagreementItemValueRow}>
            <Text style={styles.disagreementItemModelName}>{formatModelName(model)}</Text>
            <Text style={styles.disagreementItemModelValue}>{value}</Text>
          </View>
        ))}
      </View>

      {/* Navigation hint for cell-based disagreements */}
      {isNavigable && (
        <View style={styles.disagreementItemNav}>
          <Text style={styles.disagreementItemNavText}>Tap to view in grid →</Text>
        </View>
      )}
    </View>
  );

  if (isNavigable && onPress) {
    return (
      <Pressable onPress={onPress}>
        {content}
      </Pressable>
    );
  }

  return content;
}

/**
 * Grouped disagreements section
 */
function DisagreementGroup({
  title,
  disagreements,
  onDisagreementPress,
}: {
  title: string;
  disagreements: Disagreement[];
  onDisagreementPress?: (disagreement: Disagreement) => void;
}) {
  if (disagreements.length === 0) return null;

  return (
    <View style={styles.disagreementGroup}>
      <View style={styles.disagreementGroupHeader}>
        <Text style={styles.disagreementGroupTitle}>{title}</Text>
        <View style={styles.disagreementGroupCountBadge}>
          <Text style={styles.disagreementGroupCountText}>{disagreements.length}</Text>
        </View>
      </View>
      {disagreements.map((disagreement) => (
        <DisagreementListItem
          key={disagreement.id}
          disagreement={disagreement}
          onPress={() => onDisagreementPress?.(disagreement)}
        />
      ))}
    </View>
  );
}

/**
 * Diff Summary Panel - shows all disagreements in a filterable list
 */
function DiffSummaryPanel({
  comparison,
  filters,
  onToggleFilter,
  onDisagreementPress,
}: {
  comparison: ComparisonResult;
  filters: SeverityFilters;
  onToggleFilter: (severity: DisagreementSeverity) => void;
  onDisagreementPress: (disagreement: Disagreement) => void;
}) {
  // Filter disagreements by active severity filters
  const filteredDisagreements = useMemo(() => {
    return sortDisagreementsBySeverity(
      comparison.allDisagreements.filter(d => filters[d.severity])
    );
  }, [comparison.allDisagreements, filters]);

  // Group filtered disagreements by type
  const groupedDisagreements = useMemo(() => {
    const filtered = {
      gridDimensions: comparison.disagreementsByType.gridDimensions.filter(d => filters[d.severity]),
      holePositions: comparison.disagreementsByType.holePositions.filter(d => filters[d.severity]),
      regionAssignments: comparison.disagreementsByType.regionAssignments.filter(d => filters[d.severity]),
      constraints: comparison.disagreementsByType.constraints.filter(d => filters[d.severity]),
      dominoes: comparison.disagreementsByType.dominoes.filter(d => filters[d.severity]),
    };
    return filtered;
  }, [comparison.disagreementsByType, filters]);

  const hasVisibleDisagreements = filteredDisagreements.length > 0;
  const totalFiltered = filteredDisagreements.length;
  const totalAll = comparison.allDisagreements.length;

  return (
    <View style={styles.diffSummaryPanel}>
      {/* Filter controls */}
      <SeverityFilterBar
        filters={filters}
        onToggle={onToggleFilter}
        summary={comparison.summary}
      />

      {/* Results count */}
      <Text style={styles.diffSummaryResultCount}>
        {totalFiltered === totalAll
          ? `Showing all ${totalAll} disagreement${totalAll !== 1 ? 's' : ''}`
          : `Showing ${totalFiltered} of ${totalAll} disagreement${totalAll !== 1 ? 's' : ''}`}
      </Text>

      {/* Grouped disagreement list */}
      {hasVisibleDisagreements ? (
        <View style={styles.diffSummaryList}>
          <DisagreementGroup
            title="Grid Dimensions"
            disagreements={groupedDisagreements.gridDimensions}
            onDisagreementPress={onDisagreementPress}
          />
          <DisagreementGroup
            title="Hole Positions"
            disagreements={groupedDisagreements.holePositions}
            onDisagreementPress={onDisagreementPress}
          />
          <DisagreementGroup
            title="Region Assignments"
            disagreements={groupedDisagreements.regionAssignments}
            onDisagreementPress={onDisagreementPress}
          />
          <DisagreementGroup
            title="Constraints"
            disagreements={groupedDisagreements.constraints}
            onDisagreementPress={onDisagreementPress}
          />
          <DisagreementGroup
            title="Dominoes"
            disagreements={groupedDisagreements.dominoes}
            onDisagreementPress={onDisagreementPress}
          />
        </View>
      ) : (
        <View style={styles.diffSummaryEmpty}>
          <Text style={styles.diffSummaryEmptyText}>
            No disagreements match the current filters.
          </Text>
          <Text style={styles.diffSummaryEmptyHint}>
            Try enabling more severity levels above.
          </Text>
        </View>
      )}
    </View>
  );
}

/**
 * Summary view showing overall comparison statistics
 */
function SummaryView({
  comparison,
  filters,
  onToggleFilter,
  onDisagreementPress,
}: {
  comparison: ComparisonResult;
  filters: SeverityFilters;
  onToggleFilter: (severity: DisagreementSeverity) => void;
  onDisagreementPress: (disagreement: Disagreement) => void;
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

      {/* Disagreement Summary - Quick Stats */}
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

      {/* Detailed Diff Summary Panel with Filter and Navigation */}
      {!isUnanimous && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>All Disagreements</Text>
          <DiffSummaryPanel
            comparison={comparison}
            filters={filters}
            onToggleFilter={onToggleFilter}
            onDisagreementPress={onDisagreementPress}
          />
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
 * Grid visualization component showing holes and cells with disagreement highlighting
 */
function GridVisualization({
  holes,
  rows,
  cols,
  cellDisagreementMap,
  onCellPress,
}: {
  holes: boolean[][];
  rows: number;
  cols: number;
  cellDisagreementMap?: Map<string, CellDisagreement[]>;
  onCellPress?: (coordinate: CellCoordinate, disagreements: CellDisagreement[]) => void;
}) {
  return (
    <View style={styles.gridVisualization}>
      {holes.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.gridRow}>
          {row.map((isHole, colIndex) => {
            const key = cellKey(rowIndex, colIndex);
            const disagreements = cellDisagreementMap?.get(key) || [];
            const severity = getHighestCellSeverity(disagreements);
            const hasDisagreement = disagreements.length > 0;
            const severityStyle = severity ? {
              borderColor: SEVERITY_COLORS[severity].border,
              borderWidth: 2,
              backgroundColor: SEVERITY_COLORS[severity].bg,
            } : {};

            const cellContent = (
              <View
                style={[
                  styles.gridCell,
                  isHole ? styles.gridCellHole : styles.gridCellValid,
                  hasDisagreement && severityStyle,
                ]}
              >
                <Text style={[
                  styles.gridCellText,
                  isHole && styles.gridCellTextHole,
                  hasDisagreement && { color: SEVERITY_COLORS[severity!].text },
                ]}>
                  {isHole ? '#' : '·'}
                </Text>
              </View>
            );

            if (hasDisagreement && onCellPress) {
              return (
                <Pressable
                  key={colIndex}
                  onPress={() => onCellPress({ row: rowIndex, col: colIndex }, disagreements)}
                >
                  {cellContent}
                </Pressable>
              );
            }

            return <View key={colIndex}>{cellContent}</View>;
          })}
        </View>
      ))}
    </View>
  );
}

/**
 * Region visualization component with color-coded cells and disagreement highlighting
 */
function RegionVisualization({
  regions,
  holes,
  rows,
  cols,
  cellDisagreementMap,
  onCellPress,
}: {
  regions: (string | null)[][];
  holes?: boolean[][];
  rows: number;
  cols: number;
  cellDisagreementMap?: Map<string, CellDisagreement[]>;
  onCellPress?: (coordinate: CellCoordinate, disagreements: CellDisagreement[]) => void;
}) {
  return (
    <View style={styles.gridVisualization}>
      {regions.map((row, rowIndex) => (
        <View key={rowIndex} style={styles.gridRow}>
          {row.map((region, colIndex) => {
            const isHole = holes?.[rowIndex]?.[colIndex] || region === null;
            const bgColor = isHole ? '#333' : getRegionColor(region);

            const key = cellKey(rowIndex, colIndex);
            const disagreements = cellDisagreementMap?.get(key) || [];
            const severity = getHighestCellSeverity(disagreements);
            const hasDisagreement = disagreements.length > 0;
            const severityStyle = severity ? {
              borderColor: SEVERITY_COLORS[severity].border,
              borderWidth: 2,
            } : {};

            const cellContent = (
              <View
                style={[
                  styles.gridCell,
                  styles.gridCellRegion,
                  { backgroundColor: bgColor },
                  hasDisagreement && severityStyle,
                ]}
              >
                <Text style={[
                  styles.regionCellText,
                  isHole && styles.gridCellTextHole,
                ]}>
                  {isHole ? '#' : region || '.'}
                </Text>
                {hasDisagreement && (
                  <View style={[
                    styles.disagreementIndicator,
                    { backgroundColor: SEVERITY_COLORS[severity!].border },
                  ]} />
                )}
              </View>
            );

            if (hasDisagreement && onCellPress) {
              return (
                <Pressable
                  key={colIndex}
                  onPress={() => onCellPress({ row: rowIndex, col: colIndex }, disagreements)}
                >
                  {cellContent}
                </Pressable>
              );
            }

            return <View key={colIndex}>{cellContent}</View>;
          })}
        </View>
      ))}
    </View>
  );
}

/**
 * Format constraint for display
 */
function formatConstraint(constraint: ConstraintDef): string {
  if (constraint.type === 'none') return 'None';
  if (constraint.type === 'all_equal') return 'All Equal';
  if (constraint.type === 'all_different') return 'All Different';
  if (constraint.type === 'sum') {
    const op = constraint.op || '==';
    const value = constraint.value ?? '?';
    return `Sum ${op} ${value}`;
  }
  return constraint.type;
}

/**
 * Constraints list component
 */
function ConstraintsList({
  constraints,
}: {
  constraints: Record<string, ConstraintDef>;
}) {
  const entries = Object.entries(constraints);

  if (entries.length === 0) {
    return <Text style={styles.textSmall}>No constraints detected</Text>;
  }

  // Filter out 'none' type constraints
  const activeConstraints = entries.filter(([_, c]) => c.type !== 'none');

  if (activeConstraints.length === 0) {
    return <Text style={styles.textSmall}>No constraints detected</Text>;
  }

  return (
    <View style={styles.constraintList}>
      {activeConstraints.map(([region, constraint]) => (
        <View key={region} style={styles.constraintItem}>
          <View style={[styles.constraintRegionBadge, { backgroundColor: getRegionColor(region) }]}>
            <Text style={styles.constraintRegionText}>{region}</Text>
          </View>
          <Text style={styles.constraintValue}>{formatConstraint(constraint)}</Text>
        </View>
      ))}
    </View>
  );
}

/**
 * Dominoes list component with pip visualization
 */
function DominoesList({
  dominoes,
}: {
  dominoes: [number, number][];
}) {
  // Count pip frequency for summary
  const pipCounts = new Map<number, number>();
  for (const [a, b] of dominoes) {
    pipCounts.set(a, (pipCounts.get(a) || 0) + 1);
    pipCounts.set(b, (pipCounts.get(b) || 0) + 1);
  }

  return (
    <View style={styles.dominoListContainer}>
      {/* Pip count summary */}
      <View style={styles.pipSummary}>
        {Array.from(pipCounts.entries())
          .sort((a, b) => a[0] - b[0])
          .map(([pip, count]) => (
            <View key={pip} style={styles.pipBadge}>
              <Text style={styles.pipValue}>{pip}</Text>
              <Text style={styles.pipCount}>×{count}</Text>
            </View>
          ))}
      </View>

      {/* Domino tiles */}
      <View style={styles.dominoGrid}>
        {dominoes.map((domino, i) => (
          <View key={i} style={styles.dominoTile}>
            <View style={styles.dominoHalf}>
              <Text style={styles.dominoPipText}>{domino[0]}</Text>
            </View>
            <View style={styles.dominoDivider} />
            <View style={styles.dominoHalf}>
              <Text style={styles.dominoPipText}>{domino[1]}</Text>
            </View>
          </View>
        ))}
      </View>
    </View>
  );
}

/**
 * Confidence display component
 */
function ConfidenceDisplay({
  confidence,
}: {
  confidence: { board?: number; dominoes?: number };
}) {
  const items: { label: string; value: number | undefined }[] = [
    { label: 'Board', value: confidence.board },
    { label: 'Dominoes', value: confidence.dominoes },
  ];

  const validItems = items.filter(item => item.value !== undefined);

  if (validItems.length === 0) {
    return <Text style={styles.textSmall}>No confidence data</Text>;
  }

  return (
    <View style={styles.confidenceContainer}>
      {validItems.map(({ label, value }) => (
        <View key={label} style={styles.confidenceItem}>
          <Text style={styles.confidenceLabel}>{label}</Text>
          <View style={styles.confidenceBarContainer}>
            <View
              style={[
                styles.confidenceBar,
                { width: `${Math.round((value || 0) * 100)}%` },
                getConfidenceColor(value || 0),
              ]}
            />
          </View>
          <Text style={styles.confidenceValue}>{Math.round((value || 0) * 100)}%</Text>
        </View>
      ))}
    </View>
  );
}

/**
 * Get confidence bar color based on value
 */
function getConfidenceColor(value: number): { backgroundColor: string } {
  if (value >= 0.8) return { backgroundColor: '#4CAF50' };
  if (value >= 0.6) return { backgroundColor: '#8BC34A' };
  if (value >= 0.4) return { backgroundColor: '#f9a825' };
  return { backgroundColor: '#d32f2f' };
}

/**
 * Timing display component
 */
function TimingDisplay({
  timing,
}: {
  timing: ModelTiming;
}) {
  return (
    <View style={styles.timingContainer}>
      <View style={styles.timingItem}>
        <Text style={styles.timingLabel}>Response</Text>
        <Text style={styles.timingValue}>{timing.responseMs}ms</Text>
      </View>
      <View style={styles.timingItem}>
        <Text style={styles.timingLabel}>Parse</Text>
        <Text style={styles.timingValue}>{timing.parseMs}ms</Text>
      </View>
      <View style={styles.timingItem}>
        <Text style={styles.timingLabel}>Total</Text>
        <Text style={styles.timingValue}>{timing.responseMs + timing.parseMs}ms</Text>
      </View>
    </View>
  );
}

/**
 * Model-specific result view showing full extraction details with disagreement highlighting
 */
function ModelResultView({
  modelResult,
  timing,
  cellDisagreementMap,
  onCellPress,
}: {
  modelResult: NormalizedModelResult;
  timing?: ModelTiming;
  cellDisagreementMap?: Map<string, CellDisagreement[]>;
  onCellPress?: (coordinate: CellCoordinate, disagreements: CellDisagreement[]) => void;
}) {
  if (!modelResult.success) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorTitle}>Extraction Failed</Text>
        <Text style={styles.errorText}>{modelResult.error || 'Unknown error'}</Text>
      </View>
    );
  }

  const { dimensions, holes, regions, constraints, dominoes, confidence } = modelResult;
  const rows = dimensions?.rows || 0;
  const cols = dimensions?.cols || 0;

  // Count disagreements for this model's grid
  const disagreementCount = cellDisagreementMap ? cellDisagreementMap.size : 0;

  return (
    <View style={styles.modelResultContainer}>
      {/* Grid Dimensions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Grid Dimensions</Text>
        <Text style={styles.text}>
          {rows} rows × {cols} columns
        </Text>
      </View>

      {/* Disagreement Legend (if there are disagreements) */}
      {disagreementCount > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Disagreements</Text>
          <Text style={styles.textSmall}>
            {disagreementCount} cell{disagreementCount !== 1 ? 's' : ''} with disagreements. Tap to see details.
          </Text>
          <View style={styles.legendRow}>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: SEVERITY_COLORS.critical.border }]} />
              <Text style={styles.legendText}>Critical</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: SEVERITY_COLORS.warning.border }]} />
              <Text style={styles.legendText}>Warning</Text>
            </View>
            <View style={styles.legendItem}>
              <View style={[styles.legendDot, { backgroundColor: SEVERITY_COLORS.info.border }]} />
              <Text style={styles.legendText}>Info</Text>
            </View>
          </View>
        </View>
      )}

      {/* Grid with Holes */}
      {holes && holes.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Shape (# = hole)</Text>
          <GridVisualization
            holes={holes}
            rows={rows}
            cols={cols}
            cellDisagreementMap={cellDisagreementMap}
            onCellPress={onCellPress}
          />
        </View>
      )}

      {/* Regions */}
      {regions && regions.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Regions</Text>
          <RegionVisualization
            regions={regions}
            holes={holes}
            rows={rows}
            cols={cols}
            cellDisagreementMap={cellDisagreementMap}
            onCellPress={onCellPress}
          />
        </View>
      )}

      {/* Constraints */}
      {constraints && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Constraints</Text>
          <ConstraintsList constraints={constraints} />
        </View>
      )}

      {/* Dominoes */}
      {dominoes && dominoes.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Dominoes ({dominoes.length})</Text>
          <DominoesList dominoes={dominoes} />
        </View>
      )}

      {/* Confidence Scores */}
      {confidence && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Confidence</Text>
          <ConfidenceDisplay confidence={confidence} />
        </View>
      )}

      {/* Timing Info */}
      {timing && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Timing</Text>
          <TimingDisplay timing={timing} />
        </View>
      )}
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

  // State for cell detail popup
  const [selectedCell, setSelectedCell] = useState<{
    coordinate: CellCoordinate;
    disagreements: CellDisagreement[];
  } | null>(null);

  // State for severity filters (all enabled by default)
  const [severityFilters, setSeverityFilters] = useState<SeverityFilters>({
    critical: true,
    warning: true,
    info: true,
  });

  // Compute comparison result from raw responses
  const comparison = useMemo<ComparisonResult | null>(() => {
    if (!rawResponses) return null;

    return compareCellDetections(
      rawResponses.board,
      rawResponses.dominoes
    );
  }, [rawResponses]);

  // Handle cell press to show details popup
  const handleCellPress = useCallback((coordinate: CellCoordinate, disagreements: CellDisagreement[]) => {
    setSelectedCell({ coordinate, disagreements });
  }, []);

  // Close cell detail popup
  const handleCloseCellPopup = useCallback(() => {
    setSelectedCell(null);
  }, []);

  // Handle severity filter toggle
  const handleToggleSeverityFilter = useCallback((severity: DisagreementSeverity) => {
    setSeverityFilters(prev => ({
      ...prev,
      [severity]: !prev[severity],
    }));
  }, []);

  // Handle disagreement navigation - switch to first model tab and show cell popup
  const handleDisagreementPress = useCallback((disagreement: Disagreement) => {
    if ('coordinate' in disagreement) {
      const cellDisagreement = disagreement as CellDisagreement;
      // Switch to first model tab to see the grid
      if (comparison && comparison.modelsCompared.length > 0) {
        setSelectedTab(comparison.modelsCompared[0]);
      }
      // Show the cell detail popup for this disagreement
      const key = cellKey(cellDisagreement.coordinate.row, cellDisagreement.coordinate.col);
      const allDisagreementsAtCell = comparison?.cellDisagreementMap.get(key) || [cellDisagreement];
      setSelectedCell({
        coordinate: cellDisagreement.coordinate,
        disagreements: allDisagreementsAtCell,
      });
    }
  }, [comparison]);

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

  // Get timing data for current model
  const currentModelTiming = useMemo((): ModelTiming | undefined => {
    if (selectedTab === 'summary' || !rawResponses) return undefined;

    // Look for timing in board responses first
    const boardResponse = rawResponses.board.find(r => r.model === selectedTab);
    if (boardResponse?.timing) {
      return boardResponse.timing;
    }

    // Fall back to domino responses
    const dominoResponse = rawResponses.dominoes.find(r => r.model === selectedTab);
    if (dominoResponse?.timing) {
      return dominoResponse.timing;
    }

    return undefined;
  }, [selectedTab, rawResponses]);

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
            <SummaryView
              comparison={comparison}
              filters={severityFilters}
              onToggleFilter={handleToggleSeverityFilter}
              onDisagreementPress={handleDisagreementPress}
            />
          ) : currentModelResult ? (
            <ModelResultView
              modelResult={currentModelResult}
              timing={currentModelTiming}
              cellDisagreementMap={comparison.cellDisagreementMap}
              onCellPress={handleCellPress}
            />
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

        {/* Cell Detail Popup */}
        {selectedCell && (
          <CellDetailPopup
            coordinate={selectedCell.coordinate}
            disagreements={selectedCell.disagreements}
            onClose={handleCloseCellPopup}
          />
        )}
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

  // Grid Visualization
  gridVisualization: {
    backgroundColor: '#000',
    padding: 8,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  gridRow: {
    flexDirection: 'row',
  },
  gridCell: {
    width: 28,
    height: 28,
    justifyContent: 'center',
    alignItems: 'center',
    margin: 1,
    borderRadius: 2,
  },
  gridCellValid: {
    backgroundColor: '#2a2a2a',
  },
  gridCellHole: {
    backgroundColor: '#1a1a1a',
  },
  gridCellRegion: {
    // backgroundColor set dynamically
  },
  gridCellText: {
    fontSize: 14,
    fontFamily: 'Courier',
    color: '#fff',
  },
  gridCellTextHole: {
    color: '#555',
  },
  regionCellText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#fff',
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 1,
  },

  // Constraints List
  constraintList: {
    gap: 8,
  },
  constraintItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: '#2a2a2a',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  constraintRegionBadge: {
    width: 28,
    height: 28,
    borderRadius: 4,
    justifyContent: 'center',
    alignItems: 'center',
  },
  constraintRegionText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#fff',
  },
  constraintValue: {
    fontSize: 14,
    color: '#ccc',
  },

  // Dominoes List
  dominoListContainer: {
    gap: 12,
  },
  pipSummary: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 4,
  },
  pipBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  pipValue: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#fff',
  },
  pipCount: {
    fontSize: 11,
    color: '#888',
  },
  dominoGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  dominoTile: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#333',
    borderRadius: 6,
    overflow: 'hidden',
  },
  dominoHalf: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  dominoDivider: {
    width: 1,
    height: '100%',
    backgroundColor: '#555',
  },
  dominoPipText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#fff',
  },

  // Confidence Display
  confidenceContainer: {
    gap: 10,
  },
  confidenceItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  confidenceLabel: {
    width: 70,
    fontSize: 13,
    color: '#888',
  },
  confidenceBarContainer: {
    flex: 1,
    height: 8,
    backgroundColor: '#333',
    borderRadius: 4,
    overflow: 'hidden',
  },
  confidenceBar: {
    height: '100%',
    borderRadius: 4,
  },
  confidenceValue: {
    width: 40,
    fontSize: 13,
    fontWeight: 'bold',
    color: '#fff',
    textAlign: 'right',
  },

  // Timing Display
  timingContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  timingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: '#2a2a2a',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  timingLabel: {
    fontSize: 12,
    color: '#888',
  },
  timingValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#4CAF50',
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

  // Disagreement Legend
  legendRow: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 8,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  legendText: {
    fontSize: 12,
    color: '#888',
  },

  // Disagreement Indicator (small dot on cells)
  disagreementIndicator: {
    position: 'absolute',
    top: 2,
    right: 2,
    width: 6,
    height: 6,
    borderRadius: 3,
  },

  // Cell Detail Popup
  popupOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  popupContainer: {
    backgroundColor: '#2a2a2a',
    borderRadius: 12,
    width: '100%',
    maxWidth: 400,
    maxHeight: '80%',
    overflow: 'hidden',
  },
  popupHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 2,
  },
  popupHeaderContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  popupTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  popupSeverityBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  popupSeverityText: {
    fontSize: 11,
    fontWeight: 'bold',
  },
  popupCloseButton: {
    width: 32,
    height: 32,
    justifyContent: 'center',
    alignItems: 'center',
  },
  popupCloseText: {
    fontSize: 24,
    color: '#888',
    fontWeight: '300',
  },
  popupContent: {
    padding: 16,
    maxHeight: 300,
  },
  popupDisagreementItem: {
    marginBottom: 16,
  },
  popupDisagreementHeader: {
    borderLeftWidth: 3,
    paddingLeft: 12,
    marginBottom: 8,
  },
  popupDisagreementType: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  popupModelValues: {
    marginLeft: 15,
    gap: 6,
  },
  popupModelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  popupModelName: {
    fontSize: 13,
    color: '#888',
  },
  popupModelValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#fff',
    fontFamily: 'Courier',
  },

  // ══════════════════════════════════════════════════════════════════════════
  // Diff Summary Panel Styles
  // ══════════════════════════════════════════════════════════════════════════

  // Filter Bar
  filterBar: {
    flexDirection: 'row',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 12,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  filterLabel: {
    fontSize: 13,
    color: '#888',
    marginRight: 4,
  },
  filterChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#444',
  },
  filterChipInactive: {
    backgroundColor: 'transparent',
    opacity: 0.5,
  },
  filterChipText: {
    fontSize: 12,
    fontWeight: '500',
  },
  filterChipTextInactive: {
    color: '#888',
  },
  filterDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },

  // Diff Summary Panel
  diffSummaryPanel: {
    marginTop: 8,
  },
  diffSummaryResultCount: {
    fontSize: 12,
    color: '#888',
    marginBottom: 16,
    fontStyle: 'italic',
  },
  diffSummaryList: {
    gap: 20,
  },
  diffSummaryEmpty: {
    alignItems: 'center',
    padding: 24,
    backgroundColor: '#2a2a2a',
    borderRadius: 8,
  },
  diffSummaryEmptyText: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
  },
  diffSummaryEmptyHint: {
    fontSize: 12,
    color: '#666',
    textAlign: 'center',
    marginTop: 8,
  },

  // Disagreement Group
  disagreementGroup: {
    gap: 8,
  },
  disagreementGroupHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  disagreementGroupTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#aaa',
  },
  disagreementGroupCountBadge: {
    backgroundColor: '#444',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 10,
  },
  disagreementGroupCountText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#fff',
  },

  // Disagreement Item
  disagreementItem: {
    backgroundColor: '#2a2a2a',
    borderRadius: 8,
    padding: 12,
    borderLeftWidth: 3,
    marginBottom: 8,
  },
  disagreementItemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 10,
  },
  disagreementItemHeaderLeft: {
    flex: 1,
    marginRight: 8,
  },
  disagreementItemType: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 2,
  },
  disagreementItemLocation: {
    fontSize: 12,
    color: '#888',
    fontFamily: 'Courier',
  },
  disagreementItemSeverityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 10,
  },
  disagreementItemSeverityText: {
    fontSize: 10,
    fontWeight: 'bold',
  },
  disagreementItemValues: {
    gap: 4,
  },
  disagreementItemValueRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 4,
  },
  disagreementItemModelName: {
    fontSize: 12,
    color: '#888',
  },
  disagreementItemModelValue: {
    fontSize: 12,
    fontWeight: '600',
    color: '#fff',
    fontFamily: 'Courier',
  },
  disagreementItemNav: {
    marginTop: 8,
    alignItems: 'flex-end',
  },
  disagreementItemNavText: {
    fontSize: 11,
    color: '#4CAF50',
    fontStyle: 'italic',
  },
});
