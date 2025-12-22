/**
 * AI Verification Modal
 * Shows what the AI extracted before applying it to the builder state
 * Enhanced with visual diff showing extraction vs image overlay
 * Includes cell-by-cell correction UI for manual fixes
 */

import React, { useCallback, useMemo, useState } from 'react';
import {
  Image,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  useWindowDimensions,
} from 'react-native';
import Svg, { Circle, G, Line, Rect, Text as SvgText } from 'react-native-svg';
import {
  BoardExtractionResult,
  DominoExtractionResult,
  DominoPair,
  ImageInfo,
} from '../../model/overlayTypes';
import { colors, radii, spacing } from '../../theme';

// View modes for the modal
type ViewMode = 'text' | 'visual';

// Edit target type for tracking what's being edited
type EditTarget =
  | { type: 'cell'; row: number; col: number }
  | { type: 'domino'; index: number; half: 0 | 1 }
  | { type: 'constraint'; regionLabel: string }
  | null;

// Editable results that can be modified by the user
interface EditableBoardResult extends BoardExtractionResult {
  // We keep the same structure but allow modifications
}

interface EditableDominoResult extends DominoExtractionResult {
  // We keep the same structure but allow modifications
}

interface Props {
  visible: boolean;
  boardResult: BoardExtractionResult;
  dominoResult: DominoExtractionResult;
  onAccept: (editedBoard?: BoardExtractionResult, editedDominoes?: DominoExtractionResult) => void;
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

  // Editing state - track editable copies of the results
  const [editedBoard, setEditedBoard] = useState<EditableBoardResult>(() => ({
    ...boardResult,
  }));
  const [editedDominoes, setEditedDominoes] = useState<EditableDominoResult>(() => ({
    ...dominoResult,
  }));
  const [editTarget, setEditTarget] = useState<EditTarget>(null);
  const [hasEdits, setHasEdits] = useState(false);

  // Reset edited state when props change
  React.useEffect(() => {
    setEditedBoard({ ...boardResult });
    setEditedDominoes({ ...dominoResult });
    setHasEdits(false);
    setEditTarget(null);
  }, [boardResult, dominoResult]);

  // Parse extraction data for visual rendering (use edited versions)
  const parsedData = useMemo(() => {
    const holes = parseShapeToHoles(editedBoard.shape);
    const { grid: regionGrid, labels: regionLabels } = parseRegionsToGrid(editedBoard.regions);
    return { holes, regionGrid, regionLabels };
  }, [editedBoard.shape, editedBoard.regions]);

  // Calculate grid dimensions for SVG
  const gridLayout = useMemo(() => {
    const padding = 16;
    const maxWidth = screenWidth - 40; // Account for modal padding
    const cellSize = Math.min(40, Math.floor((maxWidth - padding * 2) / editedBoard.cols));
    const gridWidth = editedBoard.cols * cellSize + padding * 2;
    const gridHeight = editedBoard.rows * cellSize + padding * 2;
    return { cellSize, gridWidth, gridHeight, padding };
  }, [editedBoard.rows, editedBoard.cols, screenWidth]);

  // ════════════════════════════════════════════════════════════════════════════
  // Editing Callbacks
  // ════════════════════════════════════════════════════════════════════════════

  /**
   * Handle cell selection for editing
   */
  const handleCellPress = useCallback((row: number, col: number) => {
    setEditTarget({ type: 'cell', row, col });
  }, []);

  /**
   * Toggle a cell between hole and active cell
   */
  const toggleCellHole = useCallback((row: number, col: number) => {
    setEditedBoard(prev => {
      const shapeLines = prev.shape.split('\\n');
      const regionLines = prev.regions.split('\\n');

      // Get current character at position
      const currentShapeChar = shapeLines[row]?.[col];
      const isCurrentlyHole = currentShapeChar === '#';

      // Update shape string
      const newShapeLines = shapeLines.map((line, r) => {
        if (r !== row) return line;
        const chars = line.split('');
        chars[col] = isCurrentlyHole ? 'O' : '#';
        return chars.join('');
      });

      // Update regions string
      const newRegionLines = regionLines.map((line, r) => {
        if (r !== row) return line;
        const chars = line.split('');
        if (isCurrentlyHole) {
          // Converting from hole to cell - assign to first region
          chars[col] = 'A';
        } else {
          // Converting from cell to hole
          chars[col] = '#';
        }
        return chars.join('');
      });

      return {
        ...prev,
        shape: newShapeLines.join('\\n'),
        regions: newRegionLines.join('\\n'),
      };
    });
    setHasEdits(true);
    setEditTarget(null);
  }, []);

  /**
   * Change the region assignment for a cell
   */
  const changeCellRegion = useCallback((row: number, col: number, newRegionLabel: string) => {
    setEditedBoard(prev => {
      const regionLines = prev.regions.split('\\n');

      const newRegionLines = regionLines.map((line, r) => {
        if (r !== row) return line;
        const chars = line.split('');
        if (chars[col] !== '#') {
          chars[col] = newRegionLabel;
        }
        return chars.join('');
      });

      return {
        ...prev,
        regions: newRegionLines.join('\\n'),
      };
    });
    setHasEdits(true);
    setEditTarget(null);
  }, []);

  /**
   * Handle domino pip value editing
   */
  const handleDominoPipChange = useCallback((index: number, half: 0 | 1, newValue: number) => {
    if (newValue < 0 || newValue > 6) return;

    setEditedDominoes(prev => {
      const newDominoes = [...prev.dominoes];
      const domino = [...newDominoes[index]] as DominoPair;
      domino[half] = newValue;
      newDominoes[index] = domino;
      return { ...prev, dominoes: newDominoes };
    });
    setHasEdits(true);
    setEditTarget(null);
  }, []);

  /**
   * Increment/decrement a domino pip value
   */
  const cycleDominoPip = useCallback((index: number, half: 0 | 1, direction: 1 | -1) => {
    setEditedDominoes(prev => {
      const newDominoes = [...prev.dominoes];
      const domino = [...newDominoes[index]] as DominoPair;
      let newVal = domino[half] + direction;
      if (newVal < 0) newVal = 6;
      if (newVal > 6) newVal = 0;
      domino[half] = newVal;
      newDominoes[index] = domino;
      return { ...prev, dominoes: newDominoes };
    });
    setHasEdits(true);
  }, []);

  /**
   * Handle constraint value editing
   */
  const handleConstraintChange = useCallback(
    (regionLabel: string, type: string, value?: number, op?: string) => {
      setEditedBoard(prev => {
        const newConstraints = { ...prev.constraints };
        newConstraints[regionLabel] = { type, value, op };
        return { ...prev, constraints: newConstraints };
      });
      setHasEdits(true);
      setEditTarget(null);
    },
    []
  );

  /**
   * Delete a constraint
   */
  const deleteConstraint = useCallback((regionLabel: string) => {
    setEditedBoard(prev => {
      const newConstraints = { ...prev.constraints };
      delete newConstraints[regionLabel];
      return { ...prev, constraints: newConstraints };
    });
    setHasEdits(true);
    setEditTarget(null);
  }, []);

  /**
   * Handle accept with edited values
   */
  const handleAccept = useCallback(() => {
    if (hasEdits) {
      onAccept(editedBoard, editedDominoes);
    } else {
      onAccept();
    }
  }, [hasEdits, editedBoard, editedDominoes, onAccept]);

  /**
   * Reset all edits to original values
   */
  const resetEdits = useCallback(() => {
    setEditedBoard({ ...boardResult });
    setEditedDominoes({ ...dominoResult });
    setHasEdits(false);
    setEditTarget(null);
  }, [boardResult, dominoResult]);

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

  // Render visual grid with regions and holes (editable version)
  const renderVisualGrid = () => {
    const { cellSize, gridWidth, gridHeight, padding } = gridLayout;
    const { holes, regionGrid, regionLabels } = parsedData;
    const elements: React.ReactElement[] = [];

    // Check if a cell is currently selected
    const isSelectedCell = (row: number, col: number) =>
      editTarget?.type === 'cell' && editTarget.row === row && editTarget.col === col;

    // Draw cells with region colors
    for (let row = 0; row < editedBoard.rows; row++) {
      for (let col = 0; col < editedBoard.cols; col++) {
        const x = padding + col * cellSize;
        const y = padding + row * cellSize;
        const isHole = holes[row]?.[col] ?? false;
        const regionIndex = regionGrid[row]?.[col];
        const isSelected = isSelectedCell(row, col);

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
              stroke={isSelected ? colors.accent.brass : undefined}
              strokeWidth={isSelected ? 2 : 0}
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
              stroke={isSelected ? colors.accent.brass : undefined}
              strokeWidth={isSelected ? 3 : 0}
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
    for (let row = 0; row <= editedBoard.rows; row++) {
      const y = padding + row * cellSize;
      elements.push(
        <Line
          key={`h-line-${row}`}
          x1={padding}
          y1={y}
          x2={padding + editedBoard.cols * cellSize}
          y2={y}
          stroke={colors.surface.ash}
          strokeWidth={1}
          opacity={0.4}
        />
      );
    }
    for (let col = 0; col <= editedBoard.cols; col++) {
      const x = padding + col * cellSize;
      elements.push(
        <Line
          key={`v-line-${col}`}
          x1={x}
          y1={padding}
          x2={x}
          y2={padding + editedBoard.rows * cellSize}
          stroke={colors.surface.ash}
          strokeWidth={1}
          opacity={0.4}
        />
      );
    }

    // Add constraint indicators (small circles at region centers)
    const regionCenters = calculateRegionCenters(regionGrid, editedBoard.constraints);
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
        {/* Touch overlay for cell selection */}
        <View style={[styles.cellTouchOverlay, { width: gridWidth, height: gridHeight }]}>
          {Array.from({ length: editedBoard.rows }).map((_, row) =>
            Array.from({ length: editedBoard.cols }).map((_, col) => (
              <TouchableOpacity
                key={`touch-${row}-${col}`}
                style={[
                  styles.cellTouchTarget,
                  {
                    left: padding + col * cellSize,
                    top: padding + row * cellSize,
                    width: cellSize,
                    height: cellSize,
                  },
                ]}
                onPress={() => handleCellPress(row, col)}
                activeOpacity={0.7}
              />
            ))
          )}
        </View>
      </View>
    );
  };

  // Render dominoes visually (editable version)
  const renderVisualDominoes = () => {
    return (
      <View style={styles.visualDominoContainer}>
        <Text style={styles.editHint}>Tap halves to cycle pip values (0-6)</Text>
        {editedDominoes.dominoes.map((domino, i) => (
          <View key={i} style={styles.visualDomino}>
            <TouchableOpacity
              style={styles.dominoHalfTouchable}
              onPress={() => cycleDominoPip(i, 0, 1)}
              onLongPress={() => cycleDominoPip(i, 0, -1)}
              delayLongPress={300}
            >
              <View style={styles.dominoHalf}>{renderPips(domino[0])}</View>
            </TouchableOpacity>
            <View style={styles.dominoDivider} />
            <TouchableOpacity
              style={styles.dominoHalfTouchable}
              onPress={() => cycleDominoPip(i, 1, 1)}
              onLongPress={() => cycleDominoPip(i, 1, -1)}
              delayLongPress={300}
            >
              <View style={styles.dominoHalf}>{renderPips(domino[1])}</View>
            </TouchableOpacity>
            <Text style={styles.dominoIndex}>#{i + 1}</Text>
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
          <View style={styles.headerRow}>
            <View>
              <Text style={styles.title}>Verify AI Extraction</Text>
              <Text style={styles.subtitle}>
                {hasEdits ? 'Tap cells to edit • Review changes below' : 'Tap cells to correct errors'}
              </Text>
            </View>
            {hasEdits && (
              <TouchableOpacity style={styles.resetButton} onPress={resetEdits}>
                <Text style={styles.resetButtonText}>Reset</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Edit indicator badge */}
          {hasEdits && (
            <View style={styles.editBadge}>
              <Text style={styles.editBadgeText}>Modified</Text>
            </View>
          )}

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
                    Grid ({editedBoard.rows}×{editedBoard.cols})
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

                <Text style={styles.editHint}>Tap a cell to edit region or toggle hole</Text>

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

                {/* Cell Edit Panel - shows when a cell is selected */}
                {editTarget?.type === 'cell' && (
                  <CellEditPanel
                    row={editTarget.row}
                    col={editTarget.col}
                    isHole={parsedData.holes[editTarget.row]?.[editTarget.col] ?? false}
                    currentRegion={getRegionLabelAtCell(editedBoard.regions, editTarget.row, editTarget.col)}
                    availableRegions={parsedData.regionLabels}
                    onToggleHole={() => toggleCellHole(editTarget.row, editTarget.col)}
                    onChangeRegion={(label) => changeCellRegion(editTarget.row, editTarget.col, label)}
                    onClose={() => setEditTarget(null)}
                  />
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

              {/* Visual Constraints (Editable) */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Constraints</Text>
                <Text style={styles.editHint}>Tap constraint value to edit</Text>
                {Object.entries(editedBoard.constraints || {}).length === 0 ? (
                  <Text style={styles.textSmall}>No constraints detected</Text>
                ) : (
                  <View style={styles.constraintList}>
                    {Object.entries(editedBoard.constraints).map(([label, constraint]) => (
                      <TouchableOpacity
                        key={label}
                        style={styles.constraintItem}
                        onPress={() => setEditTarget({ type: 'constraint', regionLabel: label })}
                      >
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
                      </TouchableOpacity>
                    ))}
                  </View>
                )}

                {/* Constraint Edit Panel */}
                {editTarget?.type === 'constraint' && (
                  <ConstraintEditPanel
                    regionLabel={editTarget.regionLabel}
                    currentConstraint={editedBoard.constraints[editTarget.regionLabel]}
                    regionColor={
                      REGION_COLORS[
                        parsedData.regionLabels.indexOf(editTarget.regionLabel) % REGION_COLORS.length
                      ] || colors.surface.slate
                    }
                    onSave={(type, value, op) =>
                      handleConstraintChange(editTarget.regionLabel, type, value, op)
                    }
                    onDelete={() => deleteConstraint(editTarget.regionLabel)}
                    onClose={() => setEditTarget(null)}
                  />
                )}
              </View>

              {/* Visual Dominoes */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>
                  Dominoes ({editedDominoes.dominoes.length})
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
              {/* Text View - Shows current (possibly edited) values */}
              {/* Grid Info */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Grid Dimensions</Text>
                <Text style={styles.text}>
                  {editedBoard.rows} rows × {editedBoard.cols} columns
                  {hasEdits && ' (edited)'}
                </Text>
                {editedBoard.gridLocation && (
                  <Text style={styles.textSmall}>
                    Location: ({editedBoard.gridLocation.left}, {editedBoard.gridLocation.top}) to (
                    {editedBoard.gridLocation.right}, {editedBoard.gridLocation.bottom})
                  </Text>
                )}
              </View>

              {/* Shape */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Shape (# = hole)</Text>
                <View style={styles.gridContainer}>{formatGrid(editedBoard.shape)}</View>
              </View>

              {/* Regions */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Regions (· = unlabeled, # = hole)</Text>
                <View style={styles.gridContainer}>{formatGrid(editedBoard.regions)}</View>
              </View>

              {/* Constraints */}
              <View style={styles.section}>
                <Text style={styles.sectionTitle}>Constraints</Text>
                {Object.entries(editedBoard.constraints || {}).length === 0 ? (
                  <Text style={styles.textSmall}>No constraints detected</Text>
                ) : (
                  Object.entries(editedBoard.constraints).map(([label, constraint]) => (
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
                <Text style={styles.sectionTitle}>Dominoes ({editedDominoes.dominoes.length})</Text>
                <View style={styles.dominoContainer}>
                  {editedDominoes.dominoes.map((domino, i) => (
                    <Text key={i} style={styles.domino}>
                      [{domino[0]},{domino[1]}]
                    </Text>
                  ))}
                </View>
              </View>

              {/* Confidence (from original extraction) */}
              {boardResult.confidence && (
                <View style={styles.section}>
                  <Text style={styles.sectionTitle}>Original Confidence Scores</Text>
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
            <Text style={styles.buttonText}>Reject</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.button, hasEdits ? styles.acceptButtonEdited : styles.acceptButton]}
            onPress={handleAccept}
          >
            <Text style={styles.buttonText}>
              {hasEdits ? 'Accept with edits' : 'Accept'}
            </Text>
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
 * Get the region label at a specific cell position
 */
function getRegionLabelAtCell(regionsStr: string, row: number, col: number): string | null {
  const lines = regionsStr.split('\\n').filter(line => line.length > 0);
  const char = lines[row]?.[col];
  if (!char || char === '#' || char === '.') return null;
  return char;
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
// Cell Edit Panel Component
// ════════════════════════════════════════════════════════════════════════════

interface CellEditPanelProps {
  row: number;
  col: number;
  isHole: boolean;
  currentRegion: string | null;
  availableRegions: string[];
  onToggleHole: () => void;
  onChangeRegion: (label: string) => void;
  onClose: () => void;
}

function CellEditPanel({
  row,
  col,
  isHole,
  currentRegion,
  availableRegions,
  onToggleHole,
  onChangeRegion,
  onClose,
}: CellEditPanelProps) {
  return (
    <View style={cellEditStyles.container}>
      <View style={cellEditStyles.header}>
        <Text style={cellEditStyles.title}>
          Edit Cell ({row + 1}, {col + 1})
        </Text>
        <TouchableOpacity onPress={onClose} style={cellEditStyles.closeButton}>
          <Text style={cellEditStyles.closeText}>×</Text>
        </TouchableOpacity>
      </View>

      {/* Toggle hole/cell */}
      <TouchableOpacity style={cellEditStyles.actionButton} onPress={onToggleHole}>
        <Text style={cellEditStyles.actionText}>
          {isHole ? 'Convert to Cell' : 'Mark as Hole'}
        </Text>
      </TouchableOpacity>

      {/* Region selection (only if not a hole) */}
      {!isHole && (
        <View style={cellEditStyles.regionSection}>
          <Text style={cellEditStyles.sectionLabel}>Change Region:</Text>
          <View style={cellEditStyles.regionButtons}>
            {availableRegions.map((label, idx) => (
              <TouchableOpacity
                key={label}
                style={[
                  cellEditStyles.regionButton,
                  {
                    backgroundColor: REGION_COLORS[idx % REGION_COLORS.length],
                    borderWidth: currentRegion === label ? 3 : 0,
                    borderColor: colors.accent.brass,
                  },
                ]}
                onPress={() => onChangeRegion(label)}
              >
                <Text style={cellEditStyles.regionButtonText}>{label}</Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>
      )}
    </View>
  );
}

const cellEditStyles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface.graphite,
    borderRadius: radii.lg,
    padding: spacing[4],
    marginTop: spacing[3],
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[3],
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  closeButton: {
    width: 28,
    height: 28,
    borderRadius: radii.full,
    backgroundColor: colors.surface.ash,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: {
    fontSize: 20,
    color: colors.text.primary,
    marginTop: -2,
  },
  actionButton: {
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    padding: spacing[3],
    alignItems: 'center',
    marginBottom: spacing[3],
  },
  actionText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.primary,
  },
  regionSection: {
    marginTop: spacing[2],
  },
  sectionLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    marginBottom: spacing[2],
  },
  regionButtons: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
  },
  regionButton: {
    width: 36,
    height: 36,
    borderRadius: radii.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  regionButtonText: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.text.primary,
  },
});

// ════════════════════════════════════════════════════════════════════════════
// Constraint Edit Panel Component
// ════════════════════════════════════════════════════════════════════════════

interface ConstraintEditPanelProps {
  regionLabel: string;
  currentConstraint?: { type: string; op?: string; value?: number };
  regionColor: string;
  onSave: (type: string, value?: number, op?: string) => void;
  onDelete: () => void;
  onClose: () => void;
}

function ConstraintEditPanel({
  regionLabel,
  currentConstraint,
  regionColor,
  onSave,
  onDelete,
  onClose,
}: ConstraintEditPanelProps) {
  const [constraintType, setConstraintType] = useState(currentConstraint?.type || 'sum');
  const [constraintValue, setConstraintValue] = useState(
    currentConstraint?.value?.toString() || ''
  );
  const [constraintOp, setConstraintOp] = useState(currentConstraint?.op || '==');

  const handleSave = () => {
    const value = constraintValue ? parseInt(constraintValue, 10) : undefined;
    onSave(constraintType, value, constraintOp);
  };

  return (
    <View style={constraintEditStyles.container}>
      <View style={constraintEditStyles.header}>
        <View style={constraintEditStyles.headerLeft}>
          <View style={[constraintEditStyles.regionBadge, { backgroundColor: regionColor }]}>
            <Text style={constraintEditStyles.regionLabel}>{regionLabel}</Text>
          </View>
          <Text style={constraintEditStyles.title}>Edit Constraint</Text>
        </View>
        <TouchableOpacity onPress={onClose} style={constraintEditStyles.closeButton}>
          <Text style={constraintEditStyles.closeText}>×</Text>
        </TouchableOpacity>
      </View>

      {/* Constraint type selection */}
      <View style={constraintEditStyles.typeSection}>
        <Text style={constraintEditStyles.sectionLabel}>Type:</Text>
        <View style={constraintEditStyles.typeButtons}>
          {['sum', 'all_equal'].map(type => (
            <TouchableOpacity
              key={type}
              style={[
                constraintEditStyles.typeButton,
                constraintType === type && constraintEditStyles.typeButtonActive,
              ]}
              onPress={() => setConstraintType(type)}
            >
              <Text
                style={[
                  constraintEditStyles.typeButtonText,
                  constraintType === type && constraintEditStyles.typeButtonTextActive,
                ]}
              >
                {type === 'sum' ? 'Sum' : 'All Equal'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Value input for sum constraints */}
      {constraintType === 'sum' && (
        <View style={constraintEditStyles.valueSection}>
          <View style={constraintEditStyles.opSection}>
            <Text style={constraintEditStyles.sectionLabel}>Operator:</Text>
            <View style={constraintEditStyles.opButtons}>
              {['==', '<', '>', '!='].map(op => (
                <TouchableOpacity
                  key={op}
                  style={[
                    constraintEditStyles.opButton,
                    constraintOp === op && constraintEditStyles.opButtonActive,
                  ]}
                  onPress={() => setConstraintOp(op)}
                >
                  <Text
                    style={[
                      constraintEditStyles.opButtonText,
                      constraintOp === op && constraintEditStyles.opButtonTextActive,
                    ]}
                  >
                    {op}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          <View style={constraintEditStyles.inputSection}>
            <Text style={constraintEditStyles.sectionLabel}>Value:</Text>
            <TextInput
              style={constraintEditStyles.valueInput}
              value={constraintValue}
              onChangeText={setConstraintValue}
              keyboardType="number-pad"
              placeholder="0"
              placeholderTextColor={colors.text.tertiary}
            />
          </View>
        </View>
      )}

      {/* Action buttons */}
      <View style={constraintEditStyles.actions}>
        <TouchableOpacity style={constraintEditStyles.deleteButton} onPress={onDelete}>
          <Text style={constraintEditStyles.deleteButtonText}>Delete</Text>
        </TouchableOpacity>
        <TouchableOpacity style={constraintEditStyles.saveButton} onPress={handleSave}>
          <Text style={constraintEditStyles.saveButtonText}>Save</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const constraintEditStyles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface.graphite,
    borderRadius: radii.lg,
    padding: spacing[4],
    marginTop: spacing[3],
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[4],
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing[2],
  },
  regionBadge: {
    width: 28,
    height: 28,
    borderRadius: radii.sm,
    alignItems: 'center',
    justifyContent: 'center',
  },
  regionLabel: {
    fontSize: 14,
    fontWeight: '700',
    color: colors.text.primary,
  },
  title: {
    fontSize: 16,
    fontWeight: '600',
    color: colors.text.primary,
  },
  closeButton: {
    width: 28,
    height: 28,
    borderRadius: radii.full,
    backgroundColor: colors.surface.ash,
    alignItems: 'center',
    justifyContent: 'center',
  },
  closeText: {
    fontSize: 20,
    color: colors.text.primary,
    marginTop: -2,
  },
  typeSection: {
    marginBottom: spacing[4],
  },
  sectionLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    marginBottom: spacing[2],
  },
  typeButtons: {
    flexDirection: 'row',
    gap: spacing[2],
  },
  typeButton: {
    flex: 1,
    paddingVertical: spacing[2],
    paddingHorizontal: spacing[3],
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  typeButtonActive: {
    backgroundColor: colors.accent.brass,
  },
  typeButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.secondary,
  },
  typeButtonTextActive: {
    color: colors.text.inverse,
  },
  valueSection: {
    marginBottom: spacing[4],
  },
  opSection: {
    marginBottom: spacing[3],
  },
  opButtons: {
    flexDirection: 'row',
    gap: spacing[2],
  },
  opButton: {
    paddingVertical: spacing[2],
    paddingHorizontal: spacing[3],
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    minWidth: 44,
    alignItems: 'center',
  },
  opButtonActive: {
    backgroundColor: colors.accent.brass,
  },
  opButtonText: {
    fontSize: 14,
    fontWeight: '500',
    color: colors.text.secondary,
  },
  opButtonTextActive: {
    color: colors.text.inverse,
  },
  inputSection: {},
  valueInput: {
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
    padding: spacing[3],
    fontSize: 18,
    fontWeight: '600',
    color: colors.text.primary,
    textAlign: 'center',
  },
  actions: {
    flexDirection: 'row',
    gap: spacing[3],
    marginTop: spacing[2],
  },
  deleteButton: {
    flex: 1,
    paddingVertical: spacing[3],
    backgroundColor: colors.semantic.coral,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  deleteButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.primary,
  },
  saveButton: {
    flex: 2,
    paddingVertical: spacing[3],
    backgroundColor: colors.semantic.jade,
    borderRadius: radii.md,
    alignItems: 'center',
  },
  saveButtonText: {
    fontSize: 14,
    fontWeight: '600',
    color: colors.text.inverse,
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
  acceptButtonEdited: {
    backgroundColor: colors.accent.brass,
  },
  buttonText: {
    color: colors.text.primary,
    fontSize: 16,
    fontWeight: '600',
  },
  // Header editing mode styles
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  resetButton: {
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[2],
    backgroundColor: colors.surface.slate,
    borderRadius: radii.md,
  },
  resetButtonText: {
    fontSize: 12,
    fontWeight: '500',
    color: colors.text.secondary,
  },
  editBadge: {
    alignSelf: 'flex-start',
    marginTop: spacing[2],
    paddingHorizontal: spacing[2],
    paddingVertical: spacing[1],
    backgroundColor: colors.accent.brass,
    borderRadius: radii.sm,
  },
  editBadgeText: {
    fontSize: 11,
    fontWeight: '600',
    color: colors.text.inverse,
  },
  editHint: {
    fontSize: 12,
    color: colors.text.tertiary,
    fontStyle: 'italic',
    marginBottom: spacing[2],
  },
  // Cell touch overlay styles
  cellTouchOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
  },
  cellTouchTarget: {
    position: 'absolute',
    backgroundColor: 'transparent',
  },
  // Domino editing styles
  dominoHalfTouchable: {
    borderRadius: radii.sm,
  },
  dominoIndex: {
    fontSize: 8,
    color: colors.text.tertiary,
    marginLeft: spacing[1],
  },
});
