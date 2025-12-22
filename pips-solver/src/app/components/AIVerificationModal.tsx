/**
 * AI Verification Modal
 * Shows what the AI extracted before applying it to the builder state
 * Enhanced with visual diff showing extraction vs image overlay
 */

import React, { useMemo, useState } from 'react';
import {
  Image,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  useWindowDimensions,
} from 'react-native';
import Svg, { Circle, G, Line, Rect, Text as SvgText } from 'react-native-svg';
import { BoardExtractionResult, DominoExtractionResult, ImageInfo } from '../../model/overlayTypes';
import { colors, radii, spacing } from '../../theme';

// View modes for the modal
type ViewMode = 'text' | 'visual';

interface Props {
  visible: boolean;
  boardResult: BoardExtractionResult;
  dominoResult: DominoExtractionResult;
  onAccept: () => void;
  onReject: () => void;
  /** Optional source image to show overlay comparison */
  sourceImage?: ImageInfo | null;
}

// Region color palette for visual diff (matches theme/tokens.ts)
const REGION_COLORS = [
  '#4A6670', // Teal Shadow
  '#8B6B5C', // Warm Stone
  '#5C4A6E', // Dusty Violet
  '#6B7A4A', // Olive Drab
  '#6E5A4A', // Umber
  '#4A5C6E', // Steel Blue
  '#6E4A5C', // Mauve
  '#5C6E4A', // Sage
  '#7A5C4A', // Sienna
  '#4A6E5C', // Sea Green
];

// Parse regions string to 2D array with letter-to-index mapping
function parseRegionsToGrid(regionsStr: string): { grid: (number | null)[][]; labels: string[] } {
  const lines = regionsStr.split('\\n').filter(line => line.length > 0);
  const labels: string[] = [];
  const labelToIndex: Record<string, number> = {};

  const grid = lines.map(line =>
    line.split('').map(char => {
      if (char === '#' || char === '.') return null;
      if (!(char in labelToIndex)) {
        labelToIndex[char] = labels.length;
        labels.push(char);
      }
      return labelToIndex[char];
    })
  );

  return { grid, labels };
}

// Parse shape string to 2D boolean array (true = hole)
function parseShapeToHoles(shapeStr: string): boolean[][] {
  const lines = shapeStr.split('\\n').filter(line => line.length > 0);
  return lines.map(line =>
    line.split('').map(char => char === '#')
  );
}

export default function AIVerificationModal({
  visible,
  boardResult,
  dominoResult,
  onAccept,
  onReject,
  sourceImage,
}: Props) {
  const { width: screenWidth } = useWindowDimensions();
  const [viewMode, setViewMode] = useState<ViewMode>('visual');
  const [showOverlay, setShowOverlay] = useState(true);

  // Parse extraction data for visual rendering
  const parsedData = useMemo(() => {
    const holes = parseShapeToHoles(boardResult.shape);
    const { grid: regionGrid, labels: regionLabels } = parseRegionsToGrid(boardResult.regions);
    return { holes, regionGrid, regionLabels };
  }, [boardResult.shape, boardResult.regions]);

  // Calculate grid dimensions for SVG
  const gridLayout = useMemo(() => {
    const padding = 16;
    const maxWidth = screenWidth - 40; // Account for modal padding
    const cellSize = Math.min(40, Math.floor((maxWidth - padding * 2) / boardResult.cols));
    const gridWidth = boardResult.cols * cellSize + padding * 2;
    const gridHeight = boardResult.rows * cellSize + padding * 2;
    return { cellSize, gridWidth, gridHeight, padding };
  }, [boardResult.rows, boardResult.cols, screenWidth]);

  // Format shape and regions for text display
  const formatGrid = (str: string) => {
    return str.split('\\n').map((line, i) => (
      <Text key={i} style={styles.gridLine}>
        {line.split('').map((char, j) => (
          <Text
            key={j}
            style={[
              styles.gridChar,
              char === '#' && styles.holeChar,
              char === '.' && styles.emptyChar,
            ]}
          >
            {char === '.' ? '·' : char}
          </Text>
        ))}
      </Text>
    ));
  };

  // Render visual grid with regions and holes
  const renderVisualGrid = () => {
    const { cellSize, gridWidth, gridHeight, padding } = gridLayout;
    const { holes, regionGrid, regionLabels } = parsedData;
    const elements: React.ReactElement[] = [];

    // Draw cells with region colors
    for (let row = 0; row < boardResult.rows; row++) {
      for (let col = 0; col < boardResult.cols; col++) {
        const x = padding + col * cellSize;
        const y = padding + row * cellSize;
        const isHole = holes[row]?.[col] ?? false;
        const regionIndex = regionGrid[row]?.[col];

        if (isHole) {
          // Render hole as dark void
          elements.push(
            <Rect
              key={`hole-${row}-${col}`}
              x={x + 2}
              y={y + 2}
              width={cellSize - 4}
              height={cellSize - 4}
              fill={colors.surface.obsidian}
              rx={radii.sm}
              opacity={0.9}
            />
          );
        } else {
          // Render cell with region color
          const regionColor = regionIndex !== null
            ? REGION_COLORS[regionIndex % REGION_COLORS.length]
            : colors.surface.slate;

          elements.push(
            <Rect
              key={`cell-${row}-${col}`}
              x={x + 1}
              y={y + 1}
              width={cellSize - 2}
              height={cellSize - 2}
              fill={regionColor}
              rx={radii.sm}
              opacity={0.85}
            />
          );

          // Add region label in cell center
          if (regionIndex !== null && regionLabels[regionIndex]) {
            elements.push(
              <SvgText
                key={`label-${row}-${col}`}
                x={x + cellSize / 2}
                y={y + cellSize / 2 + 4}
                fontSize={cellSize * 0.4}
                fontWeight="600"
                fill={colors.text.primary}
                textAnchor="middle"
                opacity={0.7}
              >
                {regionLabels[regionIndex]}
              </SvgText>
            );
          }
        }
      }
    }

    // Draw grid lines
    for (let row = 0; row <= boardResult.rows; row++) {
      const y = padding + row * cellSize;
      elements.push(
        <Line
          key={`h-line-${row}`}
          x1={padding}
          y1={y}
          x2={padding + boardResult.cols * cellSize}
          y2={y}
          stroke={colors.surface.ash}
          strokeWidth={1}
          opacity={0.4}
        />
      );
    }
    for (let col = 0; col <= boardResult.cols; col++) {
      const x = padding + col * cellSize;
      elements.push(
        <Line
          key={`v-line-${col}`}
          x1={x}
          y1={padding}
          x2={x}
          y2={padding + boardResult.rows * cellSize}
          stroke={colors.surface.ash}
          strokeWidth={1}
          opacity={0.4}
        />
      );
    }

    // Add constraint indicators (small circles at region centers)
    const regionCenters = calculateRegionCenters(regionGrid, boardResult.constraints);
    regionCenters.forEach((center, idx) => {
      if (center.constraint) {
        const cx = padding + center.col * cellSize + cellSize / 2;
        const cy = padding + center.row * cellSize + cellSize / 2;

        // Diamond shape for constraints (rotated square)
        const size = cellSize * 0.35;
        elements.push(
          <G key={`constraint-${idx}`}>
            <Rect
              x={cx - size / 2}
              y={cy - size / 2}
              width={size}
              height={size}
              fill={colors.accent.brass}
              transform={`rotate(45, ${cx}, ${cy})`}
              opacity={0.9}
            />
            <SvgText
              x={cx}
              y={cy + 3}
              fontSize={size * 0.6}
              fontWeight="700"
              fill={colors.text.inverse}
              textAnchor="middle"
            >
              {center.constraint.value ?? '='}
            </SvgText>
          </G>
        );
      }
    });

    return (
      <View style={styles.visualGridContainer}>
        <Svg width={gridWidth} height={gridHeight}>
          <Rect
            x={0}
            y={0}
            width={gridWidth}
            height={gridHeight}
            fill={colors.surface.charcoal}
            rx={radii.lg}
          />
          <G>{elements}</G>
        </Svg>
      </View>
    );
  };

  // Render dominoes visually
  const renderVisualDominoes = () => {
    const dominoWidth = 50;
    const dominoHeight = 24;
    const gap = 8;
    const cols = Math.floor((screenWidth - 60) / (dominoWidth + gap));

    return (
      <View style={styles.visualDominoContainer}>
        {dominoResult.dominoes.map((domino, i) => (
          <View key={i} style={styles.visualDomino}>
            <View style={styles.dominoHalf}>
              {renderPips(domino[0])}
            </View>
            <View style={styles.dominoDivider} />
            <View style={styles.dominoHalf}>
              {renderPips(domino[1])}
            </View>
          </View>
        ))}
      </View>
    );
  };

  // Render pip dots for a domino half
  const renderPips = (value: number) => {
    const pipPositions = getPipPositions(value);
    return (
      <View style={styles.pipContainer}>
        {pipPositions.map((pos, i) => (
          <View
            key={i}
            style={[
              styles.pip,
              { left: `${50 + pos.x * 35}%`, top: `${50 + pos.y * 35}%` },
            ]}
          />
        ))}
      </View>
    );
  };

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Verify AI Extraction</Text>
          <Text style={styles.subtitle}>Review before applying</Text>

          {/* View mode toggle */}
          <View style={styles.viewToggle}>
            <TouchableOpacity
              style={[styles.toggleButton, viewMode === 'visual' && styles.toggleButtonActive]}
              onPress={() => setViewMode('visual')}
            >
              <Text style={[styles.toggleText, viewMode === 'visual' && styles.toggleTextActive]}>
                Visual
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.toggleButton, viewMode === 'text' && styles.toggleButtonActive]}
              onPress={() => setViewMode('text')}
            >
              <Text style={[styles.toggleText, viewMode === 'text' && styles.toggleTextActive]}>
                Text
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
          {viewMode === 'visual' ? (
            <>
              {/* Visual Grid Preview with Image Overlay */}
              <View style={styles.section}>
                <View style={styles.sectionHeader}>
                  <Text style={styles.sectionTitle}>
                    Grid ({boardResult.rows}×{boardResult.cols})
                  </Text>
                  {sourceImage && (
                    <TouchableOpacity
                      style={styles.overlayToggle}
                      onPress={() => setShowOverlay(!showOverlay)}
                    >
                      <Text style={styles.overlayToggleText}>
                        {showOverlay ? 'Hide Overlay' : 'Show Overlay'}
                      </Text>
                    </TouchableOpacity>
                  )}
                </View>

                {/* Source image with overlay comparison */}
                {sourceImage && showOverlay ? (
                  <View style={styles.imageOverlayContainer}>
                    <Image
                      source={{ uri: sourceImage.uri }}
                      style={[
                        styles.sourceImage,
                        {
                          width: gridLayout.gridWidth,
                          height: gridLayout.gridHeight,
                        },
                      ]}
                      resizeMode="contain"
                    />
                    <View style={styles.overlayGrid}>
                      {renderVisualGrid()}
                    </View>
                  </View>
                ) : (
                  renderVisualGrid()
                )}

                {/* Region legend */}
                <View style={styles.legend}>
                  {parsedData.regionLabels.map((label, idx) => (
                    <View key={label} style={styles.legendItem}>
                      <View
                        style={[
                          styles.legendColor,
                          { backgroundColor: REGION_COLORS[idx % REGION_COLORS.length] },
                        ]}
                      />
                      <Text style={styles.legendText}>{label}</Text>
                    </View>
                  ))}
                </View>
              </View>

              {/* Visual Constraints */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Constraints</Text>
                {Object.entries(boardResult.constraints || {}).length === 0 ? (
                  <Text style={styles.textSmall}>No constraints detected</Text>
                ) : (
                  <View style={styles.constraintList}>
                    {Object.entries(boardResult.constraints).map(([label, constraint]) => (
                      <View key={label} style={styles.constraintItem}>
                        <View
                          style={[
                            styles.constraintBadge,
                            {
                              backgroundColor:
                                REGION_COLORS[
                                  parsedData.regionLabels.indexOf(label) % REGION_COLORS.length
                                ] || colors.surface.slate,
                            },
                          ]}
                        >
                          <Text style={styles.constraintLabel}>{label}</Text>
                        </View>
                        <Text style={styles.constraintValue}>
                          {constraint.type === 'sum'
                            ? `${constraint.op || '='} ${constraint.value}`
                            : constraint.type === 'all_equal'
                            ? 'all equal'
                            : constraint.type}
                        </Text>
                      </View>
                    ))}
                  </View>
                )}
              </View>

              {/* Visual Dominoes */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>
                  Dominoes ({dominoResult.dominoes.length})
                </Text>
                {renderVisualDominoes()}
              </View>

              {/* Confidence Indicators */}
              {boardResult.confidence && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Confidence</Text>
                  <View style={styles.confidenceGrid}>
                    <ConfidenceBar label="Grid" value={boardResult.confidence.grid} />
                    <ConfidenceBar label="Regions" value={boardResult.confidence.regions} />
                    <ConfidenceBar label="Constraints" value={boardResult.confidence.constraints} />
                    {dominoResult.confidence !== undefined && (
                      <ConfidenceBar label="Dominoes" value={dominoResult.confidence} />
                    )}
                  </View>
                </View>
              )}
            </>
          ) : (
            <>
              {/* Text View - Original display */}
              {/* Grid Info */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Grid Dimensions</Text>
                <Text style={styles.text}>
                  {boardResult.rows} rows × {boardResult.cols} columns
                </Text>
                {boardResult.gridLocation && (
                  <Text style={styles.textSmall}>
                    Location: ({boardResult.gridLocation.left}, {boardResult.gridLocation.top}) to (
                    {boardResult.gridLocation.right}, {boardResult.gridLocation.bottom})
                  </Text>
                )}
              </View>

              {/* Shape */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Shape (# = hole)</Text>
                <View style={styles.gridContainer}>{formatGrid(boardResult.shape)}</View>
              </View>

              {/* Regions */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Regions (· = unlabeled, # = hole)</Text>
                <View style={styles.gridContainer}>{formatGrid(boardResult.regions)}</View>
              </View>

              {/* Constraints */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Constraints</Text>
                {Object.entries(boardResult.constraints || {}).length === 0 ? (
                  <Text style={styles.textSmall}>No constraints detected</Text>
                ) : (
                  Object.entries(boardResult.constraints).map(([label, constraint]) => (
                    <Text key={label} style={styles.text}>
                      Region {label}:{' '}
                      {constraint.type === 'sum'
                        ? `sum ${constraint.op} ${constraint.value}`
                        : constraint.type}
                    </Text>
                  ))
                )}
              </View>

              {/* Dominoes */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Dominoes ({dominoResult.dominoes.length})</Text>
                <View style={styles.dominoContainer}>
                  {dominoResult.dominoes.map((domino, i) => (
                    <Text key={i} style={styles.domino}>
                      [{domino[0]},{domino[1]}]
                    </Text>
                  ))}
                </View>
              </View>

              {/* Confidence */}
              {boardResult.confidence && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Confidence Scores</Text>
                  <Text style={styles.text}>
                    Grid: {Math.round(boardResult.confidence.grid * 100)}%
                  </Text>
                  <Text style={styles.text}>
                    Regions: {Math.round(boardResult.confidence.regions * 100)}%
                  </Text>
                  <Text style={styles.text}>
                    Constraints: {Math.round(boardResult.confidence.constraints * 100)}%
                  </Text>
                  {dominoResult.confidence !== undefined && (
                    <Text style={styles.text}>
                      Dominoes: {Math.round(dominoResult.confidence * 100)}%
                    </Text>
                  )}
                </View>
              )}
            </>
          )}
        </ScrollView>

        <View style={styles.buttons}>
          <TouchableOpacity style={[styles.button, styles.rejectButton]} onPress={onReject}>
            <Text style={styles.buttonText}>Reject - I'll do it manually</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.acceptButton]} onPress={onAccept}>
            <Text style={styles.buttonText}>Accept - Apply this</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Helper Functions
// ════════════════════════════════════════════════════════════════════════════

/**
 * Calculate the visual center cell for each region (for constraint indicators)
 */
function calculateRegionCenters(
  regionGrid: (number | null)[][],
  constraints: Record<string, { type: string; op?: string; value?: number }>
): Array<{ label: string; row: number; col: number; constraint?: { type: string; value?: number } }> {
  const regionCells: Record<string, Array<{ row: number; col: number }>> = {};

  // Collect all cells for each region
  for (let row = 0; row < regionGrid.length; row++) {
    for (let col = 0; col < (regionGrid[row]?.length ?? 0); col++) {
      const regionIndex = regionGrid[row][col];
      if (regionIndex !== null) {
        const label = String.fromCharCode(65 + regionIndex); // Convert 0 -> 'A', 1 -> 'B', etc.
        if (!regionCells[label]) {
          regionCells[label] = [];
        }
        regionCells[label].push({ row, col });
      }
    }
  }

  // Calculate center cell for each region
  return Object.entries(regionCells).map(([label, cells]) => {
    const avgRow = Math.round(cells.reduce((sum, c) => sum + c.row, 0) / cells.length);
    const avgCol = Math.round(cells.reduce((sum, c) => sum + c.col, 0) / cells.length);

    // Find the cell closest to the center
    let minDist = Infinity;
    let centerCell = cells[0];
    for (const cell of cells) {
      const dist = Math.abs(cell.row - avgRow) + Math.abs(cell.col - avgCol);
      if (dist < minDist) {
        minDist = dist;
        centerCell = cell;
      }
    }

    return {
      label,
      row: centerCell.row,
      col: centerCell.col,
      constraint: constraints[label],
    };
  });
}

/**
 * Get pip positions for domino rendering (normalized -1 to 1)
 */
function getPipPositions(value: number): Array<{ x: number; y: number }> {
  const patterns: Record<number, Array<{ x: number; y: number }>> = {
    0: [],
    1: [{ x: 0, y: 0 }],
    2: [
      { x: -0.7, y: -0.7 },
      { x: 0.7, y: 0.7 },
    ],
    3: [
      { x: -0.7, y: -0.7 },
      { x: 0, y: 0 },
      { x: 0.7, y: 0.7 },
    ],
    4: [
      { x: -0.7, y: -0.7 },
      { x: 0.7, y: -0.7 },
      { x: -0.7, y: 0.7 },
      { x: 0.7, y: 0.7 },
    ],
    5: [
      { x: -0.7, y: -0.7 },
      { x: 0.7, y: -0.7 },
      { x: 0, y: 0 },
      { x: -0.7, y: 0.7 },
      { x: 0.7, y: 0.7 },
    ],
    6: [
      { x: -0.7, y: -0.7 },
      { x: -0.7, y: 0 },
      { x: -0.7, y: 0.7 },
      { x: 0.7, y: -0.7 },
      { x: 0.7, y: 0 },
      { x: 0.7, y: 0.7 },
    ],
  };
  return patterns[value] || [];
}

/**
 * Confidence bar component for visual display
 */
function ConfidenceBar({ label, value }: { label: string; value: number }) {
  const percentage = Math.round(value * 100);
  const barColor =
    percentage >= 90
      ? colors.semantic.jade
      : percentage >= 80
      ? colors.semantic.amber
      : colors.semantic.coral;

  return (
    <View style={confidenceBarStyles.container}>
      <Text style={confidenceBarStyles.label}>{label}</Text>
      <View style={confidenceBarStyles.barBackground}>
        <View
          style={[
            confidenceBarStyles.barFill,
            { width: `${percentage}%`, backgroundColor: barColor },
          ]}
        />
      </View>
      <Text style={[confidenceBarStyles.value, { color: barColor }]}>{percentage}%</Text>
    </View>
  );
}

const confidenceBarStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing[2],
  },
  label: {
    width: 80,
    fontSize: 12,
    color: colors.text.secondary,
  },
  barBackground: {
    flex: 1,
    height: 8,
    backgroundColor: colors.surface.slate,
    borderRadius: radii.full,
    overflow: 'hidden',
    marginHorizontal: spacing[2],
  },
  barFill: {
    height: '100%',
    borderRadius: radii.full,
  },
  value: {
    width: 40,
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'right',
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.surface.charcoal,
  },
  header: {
    padding: spacing[5],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  subtitle: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  viewToggle: {
    flexDirection: 'row',
    marginTop: spacing[3],
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    padding: spacing[1],
  },
  toggleButton: {
    flex: 1,
    paddingVertical: spacing[2],
    alignItems: 'center',
    borderRadius: radii.sm,
  },
  toggleButtonActive: {
    backgroundColor: colors.accent.brass,
  },
  toggleText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.secondary,
  },
  toggleTextActive: {
    color: colors.text.inverse,
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: spacing[5],
  },
  section: {
    marginBottom: spacing[6],
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[3],
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
    marginBottom: spacing[2],
  },
  overlayToggle: {
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[1],
    backgroundColor: colors.surface.slate,
    borderRadius: radii.sm,
  },
  overlayToggleText: {
    fontSize: 12,
    color: colors.text.secondary,
  },
  text: {
    fontSize: 14,
    color: colors.text.secondary,
    marginBottom: spacing[1],
  },
  textSmall: {
    fontSize: 12,
    color: colors.text.tertiary,
    marginTop: spacing[1],
  },
  // Text view styles
  gridContainer: {
    backgroundColor: colors.surface.obsidian,
    padding: spacing[3],
    borderRadius: radii.md,
    alignSelf: 'flex-start',
  },
  gridLine: {
    fontFamily: 'Courier',
    fontSize: 18,
    lineHeight: 24,
  },
  gridChar: {
    color: colors.text.primary,
    marginRight: spacing[2],
  },
  holeChar: {
    color: colors.text.tertiary,
  },
  emptyChar: {
    color: colors.text.secondary,
  },
  dominoContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
  },
  domino: {
    fontSize: 14,
    color: colors.text.secondary,
    backgroundColor: colors.surface.slate,
    paddingHorizontal: spacing[2],
    paddingVertical: spacing[1],
    borderRadius: radii.sm,
    fontFamily: 'Courier',
  },
  // Visual view styles
  visualGridContainer: {
    alignSelf: 'center',
    marginBottom: spacing[3],
  },
  imageOverlayContainer: {
    alignSelf: 'center',
    position: 'relative',
  },
  sourceImage: {
    borderRadius: radii.md,
    opacity: 0.6,
  },
  overlayGrid: {
    position: 'absolute',
    top: 0,
    left: 0,
    opacity: 0.8,
  },
  legend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
    marginTop: spacing[2],
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface.slate,
    paddingHorizontal: spacing[2],
    paddingVertical: spacing[1],
    borderRadius: radii.sm,
  },
  legendColor: {
    width: 16,
    height: 16,
    borderRadius: radii.sm,
    marginRight: spacing[1],
  },
  legendText: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.text.primary,
  },
  constraintList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
  },
  constraintItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface.slate,
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[2],
    borderRadius: radii.md,
  },
  constraintBadge: {
    width: 24,
    height: 24,
    borderRadius: radii.sm,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: spacing[2],
  },
  constraintLabel: {
    fontSize: 12,
    fontWeight: '700',
    color: colors.text.primary,
  },
  constraintValue: {
    fontSize: 14,
    color: colors.text.secondary,
  },
  visualDominoContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
  },
  visualDomino: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.domino.ivory,
    borderRadius: radii.sm,
    padding: spacing[1],
    borderWidth: 1,
    borderColor: colors.domino.border,
  },
  dominoHalf: {
    width: 22,
    height: 22,
    position: 'relative',
  },
  dominoDivider: {
    width: 1,
    height: 18,
    backgroundColor: colors.domino.ivoryDark,
    marginHorizontal: 2,
  },
  pipContainer: {
    width: '100%',
    height: '100%',
    position: 'relative',
  },
  pip: {
    position: 'absolute',
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.domino.pip,
    transform: [{ translateX: -2 }, { translateY: -2 }],
  },
  confidenceGrid: {
    marginTop: spacing[2],
  },
  buttons: {
    flexDirection: 'row',
    padding: spacing[4],
    gap: spacing[3],
    borderTopWidth: 1,
    borderTopColor: colors.surface.ash,
  },
  button: {
    flex: 1,
    paddingVertical: spacing[4],
    borderRadius: radii.md,
    alignItems: 'center',
  },
  rejectButton: {
    backgroundColor: colors.surface.graphite,
  },
  acceptButton: {
    backgroundColor: colors.semantic.jade,
  },
  buttonText: {
    color: colors.text.primary,
    fontSize: 16,
    fontWeight: '600',
  },
});
