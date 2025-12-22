/**
 * Step 2: Region Painting
 * Allows user to paint cells with region colors using tap or drag gestures
 */

import React, { useCallback, useRef, useState } from 'react';
import {
  Image,
  LayoutChangeEvent,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import Animated from 'react-native-reanimated';
import Svg, { Line, Rect } from 'react-native-svg';
import { BuilderAction, OverlayBuilderState } from '../../../model/overlayTypes';
import { hitTestCell } from '../../../utils/gridCalculations';
import { triggerImpactLight, triggerPaintCell, triggerPaletteSelect } from '../../../utils/haptics';
import ConfidenceIndicator from '../../components/ConfidenceIndicator';

interface Props {
  state: OverlayBuilderState;
  dispatch: React.Dispatch<BuilderAction>;
}

export default function Step2RegionPainting({ state, dispatch }: Props) {
  const { image, grid, regions } = state;
  const { palette, regionGrid } = regions;
  const [containerSize, setContainerSize] = useState({ width: 0, height: 0 });
  const lastPaintedCellRef = useRef<{ row: number; col: number } | null>(null);

  const handleColorSelect = (index: number) => {
    triggerPaletteSelect(); // Haptic feedback when selecting palette color
    dispatch({ type: 'SELECT_PALETTE_COLOR', index });
  };

  const handleCellPaint = useCallback(
    (row: number, col: number) => {
      if (!grid.holes[row]?.[col]) {
        dispatch({ type: 'PAINT_CELL', row, col });
      }
    },
    [dispatch, grid.holes]
  );

  const handleContainerLayout = useCallback((e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setContainerSize({ width, height });
  }, []);

  // Paint cell and track to avoid repainting same cell during drag
  const paintCellIfNew = useCallback(
    (x: number, y: number) => {
      if (containerSize.width === 0 || containerSize.height === 0) return;

      const cell = hitTestCell(x, y, grid.bounds, grid.rows, grid.cols, containerSize);

      if (cell && !grid.holes[cell.row]?.[cell.col]) {
        const last = lastPaintedCellRef.current;
        if (!last || last.row !== cell.row || last.col !== cell.col) {
          lastPaintedCellRef.current = cell;
          triggerPaintCell(); // Haptic feedback when entering new cell
          dispatch({ type: 'PAINT_CELL', row: cell.row, col: cell.col });
        }
      }
    },
    [containerSize, grid.bounds, grid.rows, grid.cols, grid.holes, dispatch]
  );

  // Combined tap and pan gesture for painting
  const paintGesture = Gesture.Pan()
    // Keep this on the JS thread so we can safely mutate refs / call dispatch
    .runOnJS(true)
    .onBegin(e => {
      triggerImpactLight(); // Haptic feedback when painting starts
      lastPaintedCellRef.current = null;
      paintCellIfNew(e.x, e.y);
    })
    .onUpdate(e => {
      paintCellIfNew(e.x, e.y);
    })
    .onEnd(() => {
      lastPaintedCellRef.current = null;
    })
    .minDistance(0); // Allow tap to register

  if (!image) {
    return (
      <View style={styles.container}>
        <Text style={styles.emptyText}>No image available</Text>
      </View>
    );
  }

  const imageAspect = image.width / image.height;
  const cellWidth = (grid.bounds.right - grid.bounds.left) / grid.cols;
  const cellHeight = (grid.bounds.bottom - grid.bounds.top) / grid.rows;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* AI Confidence Indicator */}
      {state.aiStatus === 'done' && state.aiConfidence?.regions && (
        <View style={styles.confidenceSection}>
          <ConfidenceIndicator label="Region Boundaries" confidence={state.aiConfidence.regions} compact />
        </View>
      )}

      {/* Color Palette */}
      <View style={styles.paletteContainer}>
        <Text style={styles.paletteLabel}>Select Region:</Text>
        <View style={styles.palette}>
          {palette.colors.map((color, index) => (
            <TouchableOpacity
              key={index}
              style={[
                styles.paletteItem,
                { backgroundColor: color },
                palette.selectedIndex === index && styles.paletteItemSelected,
              ]}
              onPress={() => handleColorSelect(index)}
            >
              <Text style={styles.paletteItemText}>{palette.labels[index]}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Image with painted regions */}
      <View style={styles.imageContainer} onLayout={handleContainerLayout}>
        <Image
          source={{ uri: image.uri }}
          style={[styles.image, { aspectRatio: imageAspect }]}
          resizeMode="contain"
        />

        {/* Region overlay */}
        <Svg style={StyleSheet.absoluteFill} pointerEvents="none">
          {/* Painted cells */}
          {regionGrid.map((row, r) =>
            row.map((regionIndex, c) => {
              if (regionIndex === null || grid.holes[r]?.[c]) {
                return grid.holes[r]?.[c] ? (
                  <Rect
                    key={`cell-${r}-${c}`}
                    x={`${grid.bounds.left + c * cellWidth}%`}
                    y={`${grid.bounds.top + r * cellHeight}%`}
                    width={`${cellWidth}%`}
                    height={`${cellHeight}%`}
                    fill="rgba(0,0,0,0.8)"
                  />
                ) : null;
              }

              return (
                <Rect
                  key={`cell-${r}-${c}`}
                  x={`${grid.bounds.left + c * cellWidth}%`}
                  y={`${grid.bounds.top + r * cellHeight}%`}
                  width={`${cellWidth}%`}
                  height={`${cellHeight}%`}
                  fill={palette.colors[regionIndex]}
                  fillOpacity={0.4}
                />
              );
            })
          )}

          {/* Grid lines */}
          {Array.from({ length: grid.cols + 1 }, (_, c) => (
            <Line
              key={`v${c}`}
              x1={`${grid.bounds.left + c * cellWidth}%`}
              y1={`${grid.bounds.top}%`}
              x2={`${grid.bounds.left + c * cellWidth}%`}
              y2={`${grid.bounds.bottom}%`}
              stroke="rgba(255,255,255,0.8)"
              strokeWidth={2}
            />
          ))}
          {Array.from({ length: grid.rows + 1 }, (_, r) => (
            <Line
              key={`h${r}`}
              x1={`${grid.bounds.left}%`}
              y1={`${grid.bounds.top + r * cellHeight}%`}
              x2={`${grid.bounds.right}%`}
              y2={`${grid.bounds.top + r * cellHeight}%`}
              stroke="rgba(255,255,255,0.8)"
              strokeWidth={2}
            />
          ))}
        </Svg>

        {/* Cell labels (non-interactive, for visual feedback) */}
        <View style={StyleSheet.absoluteFill} pointerEvents="none">
          {Array.from({ length: grid.rows }, (_, r) =>
            Array.from(
              { length: grid.cols },
              (_, c) =>
                !grid.holes[r]?.[c] && (
                  <View
                    key={`label-${r}-${c}`}
                    style={{
                      position: 'absolute',
                      left: `${grid.bounds.left + c * cellWidth}%`,
                      top: `${grid.bounds.top + r * cellHeight}%`,
                      width: `${cellWidth}%`,
                      height: `${cellHeight}%`,
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <Text style={styles.cellLabelText}>
                      {palette.labels[regionGrid[r]?.[c] ?? 0]}
                    </Text>
                  </View>
                )
            )
          )}
        </View>

        {/* Gesture detector for tap and drag painting */}
        {containerSize.width > 0 && (
          <GestureDetector gesture={paintGesture}>
            <Animated.View style={StyleSheet.absoluteFill} />
          </GestureDetector>
        )}
      </View>

      <Text style={styles.hint}>
        Tap or drag to paint cells. Match regions to the puzzle image.
      </Text>
    </ScrollView>
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
    color: '#888',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 40,
  },
  paletteContainer: {
    marginBottom: 12,
  },
  paletteLabel: {
    color: '#888',
    fontSize: 13,
    marginBottom: 8,
  },
  palette: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    justifyContent: 'center',
  },
  paletteItem: {
    width: 36,
    height: 36,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  paletteItemSelected: {
    borderWidth: 3,
    borderColor: '#fff',
    transform: [{ scale: 1.1 }],
  },
  paletteItemText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
    textShadowColor: 'rgba(0,0,0,0.5)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
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
  cellLabel: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cellLabelText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 12,
    textShadowColor: 'rgba(0,0,0,0.8)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  hint: {
    color: '#888',
    fontSize: 13,
    textAlign: 'center',
    marginTop: 12,
  },
  confidenceSection: {
    marginBottom: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    backgroundColor: '#222',
    borderRadius: 8,
  },
});
