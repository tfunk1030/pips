/**
 * GridRenderer - Tactile Game Table Aesthetic
 *
 * Features:
 * - Ivory domino tiles with subtle 3D effect
 * - Pip dots rendered as circles (like real dominoes)
 * - Felt texture background with noise overlay
 * - Refined earth-tone region palette
 * - Smooth zoom/pan with gesture support
 */

import React, { useCallback, useMemo, useState } from 'react';
import { LayoutChangeEvent, StyleSheet, View } from 'react-native';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  Easing,
} from 'react-native-reanimated';
import Svg, {
  Circle,
  Defs,
  G,
  LinearGradient,
  Pattern,
  Rect,
  Stop,
  Text as SvgText,
} from 'react-native-svg';
import { Cell, NormalizedPuzzle, Solution } from '../../model/types';
import { colors, grid as gridTokens, radii, animation } from '../../theme';

// ════════════════════════════════════════════════════════════════════════════
// Types & Constants
// ════════════════════════════════════════════════════════════════════════════

interface GridRendererProps {
  puzzle: NormalizedPuzzle;
  solution?: Solution;
  onCellPress?: (cell: Cell) => void;
  highlightCell?: Cell;
  showPipDots?: boolean; // Use dot pattern instead of numbers
}

const CELL_SIZE = gridTokens.cellSize;
const GRID_PADDING = gridTokens.gridPadding;
const DOMINO_BORDER = gridTokens.dominoBorderWidth;
const PIP_SIZE = gridTokens.pipSize;
const PIP_SPACING = gridTokens.pipSpacing;
const CELL_INNER_PADDING = gridTokens.cellPadding;

// ════════════════════════════════════════════════════════════════════════════
// Pip Dot Patterns
// ════════════════════════════════════════════════════════════════════════════

// Pip positions for each value (0-6) relative to cell center
// Normalized to -1 to 1 range, multiplied by spacing at render time
const PIP_PATTERNS: Record<number, Array<{ x: number; y: number }>> = {
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

// ════════════════════════════════════════════════════════════════════════════
// Main Component
// ════════════════════════════════════════════════════════════════════════════

export default function GridRenderer({
  puzzle,
  solution,
  onCellPress,
  highlightCell,
  showPipDots = true,
}: GridRendererProps) {
  // Guard against undefined puzzle or spec
  if (!puzzle || !puzzle.spec) {
    return null;
  }

  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const gridWidth = puzzle.spec.cols * CELL_SIZE + GRID_PADDING * 2;
  const gridHeight = puzzle.spec.rows * CELL_SIZE + GRID_PADDING * 2;

  const handleLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setContainerSize({ width, height });
  }, []);

  // Pinch gesture for zoom
  const pinchGesture = Gesture.Pinch()
    .onUpdate(e => {
      scale.value = savedScale.value * e.scale;
    })
    .onEnd(() => {
      if (scale.value < 0.5) {
        scale.value = withSpring(0.5, { damping: 15 });
        savedScale.value = 0.5;
      } else if (scale.value > 3) {
        scale.value = withSpring(3, { damping: 15 });
        savedScale.value = 3;
      } else {
        savedScale.value = scale.value;
      }
    });

  // Pan gesture for panning
  const panGesture = Gesture.Pan()
    .onUpdate(e => {
      translateX.value = savedTranslateX.value + e.translationX;
      translateY.value = savedTranslateY.value + e.translationY;
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const tapGesture = Gesture.Tap()
    .runOnJS(true)
    .onEnd(e => {
      if (!onCellPress) return;
      if (containerSize.width <= 0 || containerSize.height <= 0) return;

      const localX = (e.x - translateX.value) / scale.value;
      const localY = (e.y - translateY.value) / scale.value;

      const offsetX = (containerSize.width - gridWidth) / 2;
      const offsetY = (containerSize.height - gridHeight) / 2;

      const svgX = localX - offsetX;
      const svgY = localY - offsetY;

      const col = Math.floor((svgX - GRID_PADDING) / CELL_SIZE);
      const row = Math.floor((svgY - GRID_PADDING) / CELL_SIZE);

      if (row < 0 || row >= puzzle.spec.rows || col < 0 || col >= puzzle.spec.cols) return;
      if (puzzle.spec.regions[row]?.[col] === -1) return;

      onCellPress({ row, col });
    });

  const composed = Gesture.Simultaneous(pinchGesture, panGesture, tapGesture);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  // Memoize SVG patterns to avoid re-renders
  const svgDefs = useMemo(() => <SvgDefinitions />, []);

  return (
    <GestureHandlerRootView style={styles.container} onLayout={handleLayout}>
      <GestureDetector gesture={composed}>
        <Animated.View style={[styles.svgContainer, animatedStyle]}>
          <Svg width={gridWidth} height={gridHeight}>
            {svgDefs}
            <G>
              {/* Felt background */}
              <Rect
                x={0}
                y={0}
                width={gridWidth}
                height={gridHeight}
                fill="url(#feltPattern)"
                rx={radii.lg}
              />

              {/* Draw regions (colored backgrounds) */}
              {renderRegions(puzzle)}

              {/* Draw subtle grid lines */}
              {renderGridLines(puzzle)}

              {/* Draw ivory domino tiles if solution exists */}
              {solution && renderDominoTiles(puzzle, solution)}

              {/* Draw pip values (dots or numbers) */}
              {solution && renderPipValues(puzzle, solution, highlightCell, showPipDots)}

              {/* Draw highlight */}
              {highlightCell && renderHighlight(highlightCell)}
            </G>
          </Svg>
        </Animated.View>
      </GestureDetector>
    </GestureHandlerRootView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// SVG Definitions (Patterns, Gradients)
// ════════════════════════════════════════════════════════════════════════════

function SvgDefinitions() {
  return (
    <Defs>
      {/* Felt texture pattern */}
      <Pattern
        id="feltPattern"
        patternUnits="userSpaceOnUse"
        width={100}
        height={100}
      >
        <Rect width={100} height={100} fill={colors.surface.charcoal} />
        {/* Subtle noise dots for felt texture */}
        {Array.from({ length: 50 }).map((_, i) => (
          <Circle
            key={`noise-${i}`}
            cx={Math.random() * 100}
            cy={Math.random() * 100}
            r={0.5 + Math.random() * 0.5}
            fill={colors.surface.slate}
            opacity={0.3 + Math.random() * 0.2}
          />
        ))}
      </Pattern>

      {/* Domino tile gradient (ivory with subtle 3D effect) */}
      <LinearGradient id="dominoGradient" x1="0%" y1="0%" x2="0%" y2="100%">
        <Stop offset="0%" stopColor={colors.domino.ivory} />
        <Stop offset="85%" stopColor={colors.domino.ivory} />
        <Stop offset="100%" stopColor={colors.domino.ivoryDark} />
      </LinearGradient>

      {/* Highlight glow gradient */}
      <LinearGradient id="highlightGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <Stop offset="0%" stopColor={colors.accent.brass} stopOpacity={0.8} />
        <Stop offset="100%" stopColor={colors.accent.brassLight} stopOpacity={0.6} />
      </LinearGradient>

      {/* Success glow gradient */}
      <LinearGradient id="successGradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <Stop offset="0%" stopColor={colors.semantic.jade} stopOpacity={0.8} />
        <Stop offset="100%" stopColor={colors.semantic.jadeLight} stopOpacity={0.6} />
      </LinearGradient>
    </Defs>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Render Functions
// ════════════════════════════════════════════════════════════════════════════

function renderRegions(puzzle: NormalizedPuzzle) {
  const elements: React.ReactElement[] = [];
  const regionColors = colors.regions;

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      const regionId = puzzle.spec.regions[row][col];
      const isHole = regionId === -1;

      if (isHole) {
        // Render hole as dark void
        const x = GRID_PADDING + col * CELL_SIZE;
        const y = GRID_PADDING + row * CELL_SIZE;
        elements.push(
          <Rect
            key={`hole-${row}-${col}`}
            x={x + 2}
            y={y + 2}
            width={CELL_SIZE - 4}
            height={CELL_SIZE - 4}
            fill={colors.surface.obsidian}
            rx={radii.sm}
            opacity={0.9}
          />
        );
        continue;
      }

      const color = regionColors[regionId % regionColors.length];
      const x = GRID_PADDING + col * CELL_SIZE;
      const y = GRID_PADDING + row * CELL_SIZE;

      elements.push(
        <Rect
          key={`region-${row}-${col}`}
          x={x + 1}
          y={y + 1}
          width={CELL_SIZE - 2}
          height={CELL_SIZE - 2}
          fill={color}
          rx={radii.sm}
          opacity={0.85}
        />
      );
    }
  }

  return elements;
}

function renderGridLines(puzzle: NormalizedPuzzle) {
  const elements: React.ReactElement[] = [];
  const strokeColor = colors.surface.ash;

  // Draw subtle internal grid lines
  for (let row = 0; row <= puzzle.spec.rows; row++) {
    const y = GRID_PADDING + row * CELL_SIZE;
    elements.push(
      <Rect
        key={`h-line-${row}`}
        x={GRID_PADDING}
        y={y - 0.5}
        width={puzzle.spec.cols * CELL_SIZE}
        height={1}
        fill={strokeColor}
        opacity={0.4}
      />
    );
  }

  for (let col = 0; col <= puzzle.spec.cols; col++) {
    const x = GRID_PADDING + col * CELL_SIZE;
    elements.push(
      <Rect
        key={`v-line-${col}`}
        x={x - 0.5}
        y={GRID_PADDING}
        width={1}
        height={puzzle.spec.rows * CELL_SIZE}
        fill={strokeColor}
        opacity={0.4}
      />
    );
  }

  return elements;
}

function renderDominoTiles(puzzle: NormalizedPuzzle, solution: Solution) {
  const elements: React.ReactElement[] = [];

  for (let i = 0; i < solution.dominoes.length; i++) {
    const domino = solution.dominoes[i];
    const { cell1, cell2 } = domino;

    const minRow = Math.min(cell1.row, cell2.row);
    const maxRow = Math.max(cell1.row, cell2.row);
    const minCol = Math.min(cell1.col, cell2.col);
    const maxCol = Math.max(cell1.col, cell2.col);

    const x = GRID_PADDING + minCol * CELL_SIZE + CELL_INNER_PADDING;
    const y = GRID_PADDING + minRow * CELL_SIZE + CELL_INNER_PADDING;
    const width = (maxCol - minCol + 1) * CELL_SIZE - CELL_INNER_PADDING * 2;
    const height = (maxRow - minRow + 1) * CELL_SIZE - CELL_INNER_PADDING * 2;

    // Outer shadow for depth
    elements.push(
      <Rect
        key={`domino-shadow-${i}`}
        x={x + 2}
        y={y + 2}
        width={width}
        height={height}
        fill={colors.surface.obsidian}
        rx={radii.md}
        opacity={0.3}
      />
    );

    // Main domino tile with ivory gradient
    elements.push(
      <Rect
        key={`domino-tile-${i}`}
        x={x}
        y={y}
        width={width}
        height={height}
        fill="url(#dominoGradient)"
        rx={radii.md}
        stroke={colors.domino.border}
        strokeWidth={DOMINO_BORDER}
      />
    );

    // Divider line between the two halves
    const isHorizontal = cell1.row === cell2.row;
    if (isHorizontal) {
      const dividerX = x + CELL_SIZE - CELL_INNER_PADDING;
      elements.push(
        <Rect
          key={`domino-divider-${i}`}
          x={dividerX - 0.5}
          y={y + 8}
          width={1}
          height={height - 16}
          fill={colors.domino.ivoryDark}
          opacity={0.6}
        />
      );
    } else {
      const dividerY = y + CELL_SIZE - CELL_INNER_PADDING;
      elements.push(
        <Rect
          key={`domino-divider-${i}`}
          x={x + 8}
          y={dividerY - 0.5}
          width={width - 16}
          height={1}
          fill={colors.domino.ivoryDark}
          opacity={0.6}
        />
      );
    }
  }

  return elements;
}

function renderPipValues(
  puzzle: NormalizedPuzzle,
  solution: Solution,
  highlightCell?: Cell,
  showPipDots: boolean = true
) {
  const elements: React.ReactElement[] = [];

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      if (puzzle.spec.regions[row]?.[col] === -1) {
        continue;
      }
      const value = solution.gridPips[row][col];
      if (value === null || value === undefined) {
        continue;
      }

      const centerX = GRID_PADDING + col * CELL_SIZE + CELL_SIZE / 2;
      const centerY = GRID_PADDING + row * CELL_SIZE + CELL_SIZE / 2;
      const isHighlighted = highlightCell && highlightCell.row === row && highlightCell.col === col;

      if (showPipDots) {
        // Render pip dots
        const pattern = PIP_PATTERNS[value] || [];
        const pipColor = isHighlighted ? colors.semantic.coral : colors.domino.pip;
        const pipRadius = isHighlighted ? PIP_SIZE + 1 : PIP_SIZE;

        pattern.forEach((pos, idx) => {
          elements.push(
            <Circle
              key={`pip-${row}-${col}-${idx}`}
              cx={centerX + pos.x * PIP_SPACING}
              cy={centerY + pos.y * PIP_SPACING}
              r={pipRadius / 2}
              fill={pipColor}
            />
          );
        });
      } else {
        // Render as text number (fallback)
        elements.push(
          <SvgText
            key={`pip-text-${row}-${col}`}
            x={centerX}
            y={centerY}
            fontSize={isHighlighted ? 26 : 22}
            fontWeight={isHighlighted ? '700' : '600'}
            fill={isHighlighted ? colors.semantic.coral : colors.domino.pip}
            textAnchor="middle"
            alignmentBaseline="central"
          >
            {value}
          </SvgText>
        );
      }
    }
  }

  return elements;
}

function renderHighlight(cell: Cell) {
  const x = GRID_PADDING + cell.col * CELL_SIZE + CELL_INNER_PADDING;
  const y = GRID_PADDING + cell.row * CELL_SIZE + CELL_INNER_PADDING;
  const size = CELL_SIZE - CELL_INNER_PADDING * 2;

  return (
    <G key="highlight-group">
      {/* Outer glow */}
      <Rect
        x={x - 3}
        y={y - 3}
        width={size + 6}
        height={size + 6}
        fill="none"
        stroke={colors.accent.brass}
        strokeWidth={2}
        rx={radii.md + 2}
        opacity={0.5}
      />
      {/* Inner highlight border */}
      <Rect
        x={x}
        y={y}
        width={size}
        height={size}
        fill="none"
        stroke={colors.accent.brass}
        strokeWidth={3}
        rx={radii.md}
      />
    </G>
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
  svgContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
