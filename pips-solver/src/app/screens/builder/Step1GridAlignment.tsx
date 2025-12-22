/**
 * Step 1: Grid Alignment
 * Allows user to position grid over the image, adjust rows/cols, and mark holes
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  Image,
  LayoutChangeEvent,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated from 'react-native-reanimated';
import Svg, { Circle, Line, Rect } from 'react-native-svg';
import { BuilderAction, GridBounds, OverlayBuilderState } from '../../../model/overlayTypes';
import { constrainBounds, hitTestCell } from '../../../utils/gridCalculations';
import ConfidenceIndicator from '../../components/ConfidenceIndicator';
import { Body, Button, Card, Label } from '../../components/ui';

interface StageConfidence {
  board?: number;
  dominoes?: number;
  currentStage?: string;
}

interface Props {
  state: OverlayBuilderState;
  dispatch: React.Dispatch<BuilderAction>;
  onPickNewImage: () => void;
  onAIExtract?: () => void;
  aiProgress?: string | null;
  stageConfidence?: StageConfidence;
}

export default function Step1GridAlignment({
  state,
  dispatch,
  onPickNewImage,
  onAIExtract,
  aiProgress,
  stageConfidence,
}: Props) {
  const { image, grid } = state;
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });

  const handleRowChange = (delta: number) => {
    dispatch({ type: 'SET_ROWS', rows: grid.rows + delta });
  };

  const handleColChange = (delta: number) => {
    dispatch({ type: 'SET_COLS', cols: grid.cols + delta });
  };

  const handleCellTap = useCallback(
    (row: number, col: number) => {
      dispatch({ type: 'TOGGLE_HOLE', row, col });
    },
    [dispatch]
  );

  const handleBoundsChange = useCallback(
    (newBounds: GridBounds) => {
      const constrained = constrainBounds(newBounds);
      dispatch({ type: 'SET_GRID_BOUNDS', bounds: constrained });
    },
    [dispatch]
  );

  const handleContainerLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setContainerSize({ width, height });
  }, []);

  if (!image) {
    return (
      <View style={styles.container}>
        <Body color="secondary" align="center" style={styles.emptyText}>No image selected</Body>
        <Button
          title="Select Image"
          variant="primary"
          onPress={onPickNewImage}
          style={styles.pickButton}
        />
      </View>
    );
  }

  // Calculate cell dimensions for rendering
  const imageAspect = image.width / image.height;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Image with grid overlay */}
      <View style={styles.imageContainer} onLayout={handleContainerLayout}>
        <Image
          source={{ uri: image.uri }}
          style={[styles.image, { aspectRatio: imageAspect }]}
          resizeMode="contain"
        />

        {/* Grid overlay with draggable edges */}
        {containerSize.width > 0 && containerSize.height > 0 && (
          <View style={StyleSheet.absoluteFill} pointerEvents="box-none">
            <DraggableGridOverlay
              bounds={grid.bounds}
              rows={grid.rows}
              cols={grid.cols}
              holes={grid.holes}
              containerSize={containerSize}
              onCellTap={handleCellTap}
              onBoundsChange={handleBoundsChange}
            />
          </View>
        )}
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <View style={styles.controlRow}>
          <Label color="secondary" style={styles.controlLabel}>Rows</Label>
          <Button
            title="−"
            variant="secondary"
            size="small"
            onPress={() => handleRowChange(-1)}
          />
          <Body style={styles.controlValue}>{grid.rows}</Body>
          <Button
            title="+"
            variant="secondary"
            size="small"
            onPress={() => handleRowChange(1)}
          />
        </View>

        <View style={styles.controlRow}>
          <Label color="secondary" style={styles.controlLabel}>Cols</Label>
          <Button
            title="−"
            variant="secondary"
            size="small"
            onPress={() => handleColChange(-1)}
          />
          <Body style={styles.controlValue}>{grid.cols}</Body>
          <Button
            title="+"
            variant="secondary"
            size="small"
            onPress={() => handleColChange(1)}
          />
        </View>
      </View>

      {/* AI Confidence Indicator */}
      {state.aiStatus === 'done' && state.aiConfidence?.grid && (
        <Card style={styles.confidenceSection}>
          <Label size="small" color="secondary" style={styles.confidenceTitle}>AI Extraction Confidence</Label>
          <ConfidenceIndicator label="Grid Layout" confidence={state.aiConfidence.grid} compact />
        </Card>
      )}

      {/* AI Extraction Button */}
      {onAIExtract && (
        <View style={styles.aiSection}>
          <Button
            title="Use AI to Extract Puzzle"
            variant="primary"
            onPress={onAIExtract}
            loading={!!aiProgress}
            style={styles.aiButton}
          />
          <Body size="small" color="secondary" align="center" style={styles.aiHint}>AI will detect grid, regions, constraints, and dominoes</Body>

          {/* Per-stage confidence indicators during extraction */}
          {aiProgress && stageConfidence && (stageConfidence.board !== undefined || stageConfidence.dominoes !== undefined) && (
            <View style={styles.stageConfidenceContainer}>
              <Label size="small" color="secondary" style={styles.stageConfidenceTitle}>Extraction Progress</Label>
              <View style={styles.stageConfidenceIndicators}>
                <ConfidenceIndicator
                  label="Board"
                  confidence={stageConfidence.board}
                  compact
                />
                <ConfidenceIndicator
                  label="Dominoes"
                  confidence={stageConfidence.dominoes}
                  compact
                />
              </View>
            </View>
          )}
        </View>
      )}

      <Body size="small" color="secondary" align="center" style={styles.hint}>Drag edges to align grid. Tap cells to mark holes.</Body>

      <Button
        title="Choose Different Image"
        variant="ghost"
        onPress={onPickNewImage}
      />
    </ScrollView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Draggable Grid Overlay Component
// ════════════════════════════════════════════════════════════════════════════

interface DraggableGridOverlayProps {
  bounds: GridBounds;
  rows: number;
  cols: number;
  holes: boolean[][];
  containerSize: { width: number; height: number };
  onCellTap: (row: number, col: number) => void;
  onBoundsChange: (bounds: GridBounds) => void;
}

function DraggableGridOverlay({
  bounds,
  rows,
  cols,
  holes,
  containerSize,
  onCellTap,
  onBoundsChange,
}: DraggableGridOverlayProps) {
  const cellWidth = (bounds.right - bounds.left) / cols;
  const cellHeight = (bounds.bottom - bounds.top) / rows;

  // Edge handle size (in pixels)
  const HANDLE_SIZE = 24;
  const HANDLE_HIT_SLOP = 20;

  // Convert percentage to pixels
  const toPixels = (pct: number, dimension: 'width' | 'height') => {
    const dim = containerSize[dimension];
    if (!dim || dim <= 0) return 0;
    return (pct / 100) * dim;
  };

  // Convert pixels to percentage
  const toPercent = (px: number, dimension: 'width' | 'height') => {
    const dim = containerSize[dimension];
    if (!dim || dim <= 0) return 0;
    return (px / dim) * 100;
  };

  // Calculate pixel positions
  const leftPx = toPixels(bounds.left, 'width');
  const rightPx = toPixels(bounds.right, 'width');
  const topPx = toPixels(bounds.top, 'height');
  const bottomPx = toPixels(bounds.bottom, 'height');

  // Refs to track starting values during drag
  const startBoundsRef = useRef(bounds);

  // Create gesture for left edge
  const leftGesture = Gesture.Pan()
    .runOnJS(true)
    .onBegin(() => {
      startBoundsRef.current = bounds;
    })
    .onUpdate(e => {
      const deltaPercent = toPercent(e.translationX, 'width');
      const start = startBoundsRef.current;
      const newLeft = Math.max(0, Math.min(start.right - 10, start.left + deltaPercent));
      onBoundsChange({ ...start, left: newLeft });
    })
    .hitSlop({ left: HANDLE_HIT_SLOP, right: HANDLE_HIT_SLOP, top: 0, bottom: 0 });

  // Create gesture for right edge
  const rightGesture = Gesture.Pan()
    .runOnJS(true)
    .onBegin(() => {
      startBoundsRef.current = bounds;
    })
    .onUpdate(e => {
      const deltaPercent = toPercent(e.translationX, 'width');
      const start = startBoundsRef.current;
      const newRight = Math.min(100, Math.max(start.left + 10, start.right + deltaPercent));
      onBoundsChange({ ...start, right: newRight });
    })
    .hitSlop({ left: HANDLE_HIT_SLOP, right: HANDLE_HIT_SLOP, top: 0, bottom: 0 });

  // Create gesture for top edge
  const topGesture = Gesture.Pan()
    .runOnJS(true)
    .onBegin(() => {
      startBoundsRef.current = bounds;
    })
    .onUpdate(e => {
      const deltaPercent = toPercent(e.translationY, 'height');
      const start = startBoundsRef.current;
      const newTop = Math.max(0, Math.min(start.bottom - 10, start.top + deltaPercent));
      onBoundsChange({ ...start, top: newTop });
    })
    .hitSlop({ left: 0, right: 0, top: HANDLE_HIT_SLOP, bottom: HANDLE_HIT_SLOP });

  // Create gesture for bottom edge
  const bottomGesture = Gesture.Pan()
    .runOnJS(true)
    .onBegin(() => {
      startBoundsRef.current = bounds;
    })
    .onUpdate(e => {
      const deltaPercent = toPercent(e.translationY, 'height');
      const start = startBoundsRef.current;
      const newBottom = Math.min(100, Math.max(start.top + 10, start.bottom + deltaPercent));
      onBoundsChange({ ...start, bottom: newBottom });
    })
    .hitSlop({ left: 0, right: 0, top: HANDLE_HIT_SLOP, bottom: HANDLE_HIT_SLOP });

  // Tap to toggle holes (cells that are not part of the puzzle)
  const tapGesture = Gesture.Tap()
    .runOnJS(true)
    .onEnd(e => {
      const cell = hitTestCell(e.x, e.y, bounds, rows, cols, containerSize);
      if (cell) {
        onCellTap(cell.row, cell.col);
      }
    });

  return (
    <View style={StyleSheet.absoluteFill}>
      {/* SVG Grid */}
      <Svg style={StyleSheet.absoluteFill}>
        {/* Grid border */}
        <Rect
          x={`${bounds.left}%`}
          y={`${bounds.top}%`}
          width={`${bounds.right - bounds.left}%`}
          height={`${bounds.bottom - bounds.top}%`}
          stroke="white"
          strokeWidth={3}
          fill="none"
        />

        {/* Vertical grid lines */}
        {Array.from({ length: cols + 1 }, (_, c) => (
          <Line
            key={`v${c}`}
            x1={`${bounds.left + c * cellWidth}%`}
            y1={`${bounds.top}%`}
            x2={`${bounds.left + c * cellWidth}%`}
            y2={`${bounds.bottom}%`}
            stroke="rgba(255,255,255,0.6)"
            strokeWidth={1.5}
          />
        ))}

        {/* Horizontal grid lines */}
        {Array.from({ length: rows + 1 }, (_, r) => (
          <Line
            key={`h${r}`}
            x1={`${bounds.left}%`}
            y1={`${bounds.top + r * cellHeight}%`}
            x2={`${bounds.right}%`}
            y2={`${bounds.top + r * cellHeight}%`}
            stroke="rgba(255,255,255,0.6)"
            strokeWidth={1.5}
          />
        ))}

        {/* Hole overlays */}
        {holes.map((row, r) =>
          row.map(
            (isHole, c) =>
              isHole && (
                <Rect
                  key={`hole-${r}-${c}`}
                  x={`${bounds.left + c * cellWidth}%`}
                  y={`${bounds.top + r * cellHeight}%`}
                  width={`${cellWidth}%`}
                  height={`${cellHeight}%`}
                  fill="rgba(0,0,0,0.7)"
                />
              )
          )
        )}

        {/* Edge handle indicators (circles at midpoint of each edge) */}
        <Circle
          cx={`${bounds.left}%`}
          cy={`${(bounds.top + bounds.bottom) / 2}%`}
          r={8}
          fill="#007AFF"
          stroke="white"
          strokeWidth={2}
        />
        <Circle
          cx={`${bounds.right}%`}
          cy={`${(bounds.top + bounds.bottom) / 2}%`}
          r={8}
          fill="#007AFF"
          stroke="white"
          strokeWidth={2}
        />
        <Circle
          cx={`${(bounds.left + bounds.right) / 2}%`}
          cy={`${bounds.top}%`}
          r={8}
          fill="#007AFF"
          stroke="white"
          strokeWidth={2}
        />
        <Circle
          cx={`${(bounds.left + bounds.right) / 2}%`}
          cy={`${bounds.bottom}%`}
          r={8}
          fill="#007AFF"
          stroke="white"
          strokeWidth={2}
        />
      </Svg>

      {/* Tap layer for toggling holes (kept under edge handles) */}
      <GestureDetector gesture={tapGesture}>
        <Animated.View style={StyleSheet.absoluteFill} />
      </GestureDetector>

      {/* Draggable edge handles (invisible touch targets) */}
      {/* Left edge */}
      <GestureDetector gesture={leftGesture}>
        <Animated.View
          style={[
            styles.edgeHandle,
            styles.verticalEdge,
            {
              left: leftPx - HANDLE_SIZE / 2,
              top: topPx,
              height: bottomPx - topPx,
              width: HANDLE_SIZE,
            },
          ]}
        />
      </GestureDetector>

      {/* Right edge */}
      <GestureDetector gesture={rightGesture}>
        <Animated.View
          style={[
            styles.edgeHandle,
            styles.verticalEdge,
            {
              left: rightPx - HANDLE_SIZE / 2,
              top: topPx,
              height: bottomPx - topPx,
              width: HANDLE_SIZE,
            },
          ]}
        />
      </GestureDetector>

      {/* Top edge */}
      <GestureDetector gesture={topGesture}>
        <Animated.View
          style={[
            styles.edgeHandle,
            styles.horizontalEdge,
            {
              left: leftPx,
              top: topPx - HANDLE_SIZE / 2,
              width: rightPx - leftPx,
              height: HANDLE_SIZE,
            },
          ]}
        />
      </GestureDetector>

      {/* Bottom edge */}
      <GestureDetector gesture={bottomGesture}>
        <Animated.View
          style={[
            styles.edgeHandle,
            styles.horizontalEdge,
            {
              left: leftPx,
              top: bottomPx - HANDLE_SIZE / 2,
              width: rightPx - leftPx,
              height: HANDLE_SIZE,
            },
          ]}
        />
      </GestureDetector>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  emptyText: {
    marginTop: 40,
  },
  pickButton: {
    alignSelf: 'center',
    marginTop: 16,
  },
  imageContainer: {
    width: '100%',
    backgroundColor: '#000',
    borderRadius: 12,
    overflow: 'hidden',
  },
  image: {
    width: '100%',
  },
  controls: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 24,
    marginTop: 16,
  },
  controlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  controlLabel: {
    marginRight: 4,
  },
  controlValue: {
    width: 30,
    textAlign: 'center',
    fontWeight: '600',
  },
  hint: {
    marginTop: 12,
  },
  aiSection: {
    marginTop: 16,
    alignItems: 'center',
  },
  aiButton: {
    backgroundColor: '#9C27B0',
    minWidth: 200,
  },
  aiHint: {
    marginTop: 8,
  },
  confidenceSection: {
    marginTop: 16,
  },
  confidenceTitle: {
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  stageConfidenceContainer: {
    marginTop: 12,
    paddingHorizontal: 16,
    paddingVertical: 10,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  stageConfidenceTitle: {
    marginBottom: 8,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  stageConfidenceIndicators: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    gap: 16,
  },
  edgeHandle: {
    position: 'absolute',
    // Touch targets are invisible - visual feedback via SVG circles
  },
  verticalEdge: {
    // Vertical edge (left/right)
  },
  horizontalEdge: {
    // Horizontal edge (top/bottom)
  },
});
