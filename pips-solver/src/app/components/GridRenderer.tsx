/**
 * Grid renderer with zoom/pan support using react-native-svg
 */

import React from 'react';
import { StyleSheet } from 'react-native';
import { Gesture, GestureDetector, GestureHandlerRootView } from 'react-native-gesture-handler';
import Animated, { useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import Svg, { G, Line, Rect, Text as SvgText } from 'react-native-svg';
import { Cell, NormalizedPuzzle, Solution } from '../../model/types';

interface GridRendererProps {
  puzzle: NormalizedPuzzle;
  solution?: Solution;
  onCellPress?: (cell: Cell) => void;
  highlightCell?: Cell;
}

const CELL_SIZE = 60;
const GRID_PADDING = 20;

// Region colors (cycling palette)
const REGION_COLORS = [
  '#FFE5E5',
  '#E5F3FF',
  '#E5FFE5',
  '#FFF3E5',
  '#FFE5FF',
  '#E5FFFF',
  '#FFE5F3',
  '#F3FFE5',
  '#E5E5FF',
  '#FFFFE5',
  '#FFE5E5',
  '#E5FFE5',
];

export default function GridRenderer({
  puzzle,
  solution,
  onCellPress,
  highlightCell,
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

  const gridWidth = puzzle.spec.cols * CELL_SIZE + GRID_PADDING * 2;
  const gridHeight = puzzle.spec.rows * CELL_SIZE + GRID_PADDING * 2;

  // Pinch gesture for zoom
  const pinchGesture = Gesture.Pinch()
    .onUpdate(e => {
      scale.value = savedScale.value * e.scale;
    })
    .onEnd(() => {
      // Clamp scale
      if (scale.value < 0.5) {
        scale.value = withSpring(0.5);
        savedScale.value = 0.5;
      } else if (scale.value > 3) {
        scale.value = withSpring(3);
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

  const composed = Gesture.Simultaneous(pinchGesture, panGesture);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translateX.value },
      { translateY: translateY.value },
      { scale: scale.value },
    ],
  }));

  return (
    <GestureHandlerRootView style={styles.container}>
      <GestureDetector gesture={composed}>
        <Animated.View style={[styles.svgContainer, animatedStyle]}>
          <Svg width={gridWidth} height={gridHeight}>
            <G>
              {/* Draw regions (colored backgrounds) */}
              {renderRegions(puzzle)}

              {/* Draw grid lines */}
              {renderGridLines(puzzle)}

              {/* Draw domino borders if solution exists */}
              {solution && renderDominoBorders(puzzle, solution)}

              {/* Draw pip values */}
              {solution && renderPipValues(puzzle, solution, highlightCell)}

              {/* Draw highlight */}
              {highlightCell && renderHighlight(highlightCell)}
            </G>
          </Svg>
        </Animated.View>
      </GestureDetector>
    </GestureHandlerRootView>
  );
}

function renderRegions(puzzle: NormalizedPuzzle) {
  const elements: React.ReactElement[] = [];

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      const regionId = puzzle.spec.regions[row][col];
      const color = REGION_COLORS[regionId % REGION_COLORS.length];

      const x = GRID_PADDING + col * CELL_SIZE;
      const y = GRID_PADDING + row * CELL_SIZE;

      elements.push(
        <Rect
          key={`region-${row}-${col}`}
          x={x}
          y={y}
          width={CELL_SIZE}
          height={CELL_SIZE}
          fill={color}
        />
      );
    }
  }

  return elements;
}

function renderGridLines(puzzle: NormalizedPuzzle) {
  const elements: React.ReactElement[] = [];

  // Horizontal lines
  for (let row = 0; row <= puzzle.spec.rows; row++) {
    const y = GRID_PADDING + row * CELL_SIZE;
    elements.push(
      <Line
        key={`h-line-${row}`}
        x1={GRID_PADDING}
        y1={y}
        x2={GRID_PADDING + puzzle.spec.cols * CELL_SIZE}
        y2={y}
        stroke="#999"
        strokeWidth={1}
      />
    );
  }

  // Vertical lines
  for (let col = 0; col <= puzzle.spec.cols; col++) {
    const x = GRID_PADDING + col * CELL_SIZE;
    elements.push(
      <Line
        key={`v-line-${col}`}
        x1={x}
        y1={GRID_PADDING}
        x2={x}
        y2={GRID_PADDING + puzzle.spec.rows * CELL_SIZE}
        stroke="#999"
        strokeWidth={1}
      />
    );
  }

  return elements;
}

function renderDominoBorders(puzzle: NormalizedPuzzle, solution: Solution) {
  const elements: React.ReactElement[] = [];

  for (let i = 0; i < solution.dominoes.length; i++) {
    const domino = solution.dominoes[i];
    const { cell1, cell2 } = domino;

    // Get bounding box
    const minRow = Math.min(cell1.row, cell2.row);
    const maxRow = Math.max(cell1.row, cell2.row);
    const minCol = Math.min(cell1.col, cell2.col);
    const maxCol = Math.max(cell1.col, cell2.col);

    const x = GRID_PADDING + minCol * CELL_SIZE;
    const y = GRID_PADDING + minRow * CELL_SIZE;
    const width = (maxCol - minCol + 1) * CELL_SIZE;
    const height = (maxRow - minRow + 1) * CELL_SIZE;

    elements.push(
      <Rect
        key={`domino-border-${i}`}
        x={x}
        y={y}
        width={width}
        height={height}
        fill="none"
        stroke="#000"
        strokeWidth={3}
      />
    );
  }

  return elements;
}

function renderPipValues(puzzle: NormalizedPuzzle, solution: Solution, highlightCell?: Cell) {
  const elements: React.ReactElement[] = [];

  for (let row = 0; row < puzzle.spec.rows; row++) {
    for (let col = 0; col < puzzle.spec.cols; col++) {
      const value = solution.gridPips[row][col];
      const x = GRID_PADDING + col * CELL_SIZE + CELL_SIZE / 2;
      const y = GRID_PADDING + row * CELL_SIZE + CELL_SIZE / 2;

      const isHighlighted = highlightCell && highlightCell.row === row && highlightCell.col === col;

      elements.push(
        <SvgText
          key={`pip-${row}-${col}`}
          x={x}
          y={y}
          fontSize={isHighlighted ? 28 : 24}
          fontWeight={isHighlighted ? 700 : 400}
          fill={isHighlighted ? '#FF0000' : '#000'}
          textAnchor="middle"
          alignmentBaseline="central"
        >
          {value}
        </SvgText>
      );
    }
  }

  return elements;
}

function renderHighlight(cell: Cell) {
  const x = GRID_PADDING + cell.col * CELL_SIZE;
  const y = GRID_PADDING + cell.row * CELL_SIZE;

  return (
    <Rect
      key="highlight"
      x={x}
      y={y}
      width={CELL_SIZE}
      height={CELL_SIZE}
      fill="none"
      stroke="#FF0000"
      strokeWidth={4}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  svgContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
