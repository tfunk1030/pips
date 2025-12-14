import * as React from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import Svg, { G, Line, Rect, Text as SvgText } from 'react-native-svg';
import { PanGestureHandler, PinchGestureHandler, State } from 'react-native-gesture-handler';
import type { Puzzle, Solution } from '../../model/types';
import { theme } from '../theme';

export function GridSvg({ puzzle, solution }: { puzzle: Puzzle; solution?: Solution }) {
  const cellSize = 44;
  const w = puzzle.cols * cellSize;
  const h = puzzle.rows * cellSize;

  const [selected, setSelected] = React.useState<number | null>(null);

  const baseScale = React.useRef(new Animated.Value(1)).current;
  const pinchScale = React.useRef(new Animated.Value(1)).current;
  const scale = Animated.multiply(baseScale, pinchScale);
  const translateX = React.useRef(new Animated.Value(0)).current;
  const translateY = React.useRef(new Animated.Value(0)).current;

  const lastScaleRef = React.useRef(1);
  const lastPanRef = React.useRef({ x: 0, y: 0 });

  const onPanEvent = Animated.event([{ nativeEvent: { translationX: translateX, translationY: translateY } }], {
    useNativeDriver: true,
  });

  const onPinchEvent = Animated.event([{ nativeEvent: { scale: pinchScale } }], { useNativeDriver: true });

  const onPanStateChange = (e: any) => {
    if (e.nativeEvent.oldState !== State.ACTIVE) return;
    lastPanRef.current = {
      x: lastPanRef.current.x + e.nativeEvent.translationX,
      y: lastPanRef.current.y + e.nativeEvent.translationY,
    };
    translateX.setOffset(lastPanRef.current.x);
    translateX.setValue(0);
    translateY.setOffset(lastPanRef.current.y);
    translateY.setValue(0);
  };

  const onPinchStateChange = (e: any) => {
    if (e.nativeEvent.oldState !== State.ACTIVE) return;
    const next = clamp(lastScaleRef.current * e.nativeEvent.scale, 0.5, 6);
    lastScaleRef.current = next;
    baseScale.setValue(next);
    pinchScale.setValue(1);
  };

  const selectedInfo = React.useMemo(() => {
    if (selected == null) return null;
    const cell = puzzle.cells[selected];
    const mate = solution?.mateCellIdByCellId?.[selected];
    const pip = solution?.gridPips?.[cell.r]?.[cell.c];
    return { cell, mate, pip };
  }, [selected, puzzle, solution]);

  return (
    <View style={styles.wrap}>
      {selectedInfo ? (
        <View style={styles.info}>
          <Text style={styles.infoText}>
            ({selectedInfo.cell.r},{selectedInfo.cell.c}) region={selectedInfo.cell.regionId}
            {selectedInfo.pip != null ? ` pip=${selectedInfo.pip}` : ''}
            {selectedInfo.mate != null && selectedInfo.mate >= 0 ? ` mateCellId=${selectedInfo.mate}` : ''}
          </Text>
        </View>
      ) : (
        <View style={styles.info}>
          <Text style={styles.infoText}>Tap a cell</Text>
        </View>
      )}

      <PanGestureHandler onGestureEvent={onPanEvent} onHandlerStateChange={onPanStateChange}>
        <Animated.View style={{ flex: 1 }}>
          <PinchGestureHandler onGestureEvent={onPinchEvent} onHandlerStateChange={onPinchStateChange}>
            <Animated.View
              style={[
                styles.canvas,
                {
                  transform: [
                    { translateX },
                    { translateY },
                    { scale },
                  ],
                },
              ]}
            >
              <Svg width={w} height={h}>
                <Rect x={0} y={0} width={w} height={h} fill={theme.colors.card} />

                {/* Domino links (underlay) */}
                {solution
                  ? puzzle.cells.map((cell) => {
                      const a = cell.id;
                      const b = solution.mateCellIdByCellId[a];
                      if (b == null || b < a) return null;
                      const cellB = puzzle.cells[b];
                      const x1 = cell.c * cellSize + cellSize / 2;
                      const y1 = cell.r * cellSize + cellSize / 2;
                      const x2 = cellB.c * cellSize + cellSize / 2;
                      const y2 = cellB.r * cellSize + cellSize / 2;
                      return (
                        <Line
                          key={`d-${a}-${b}`}
                          x1={x1}
                          y1={y1}
                          x2={x2}
                          y2={y2}
                          stroke="#E9EEF7"
                          strokeOpacity={0.45}
                          strokeWidth={10}
                          strokeLinecap="round"
                        />
                      );
                    })
                  : null}

                {/* Cells */}
                {puzzle.cells.map((cell) => {
                  const fill = regionColor(cell.regionId);
                  const x = cell.c * cellSize;
                  const y = cell.r * cellSize;
                  const isSel = selected === cell.id;
                  return (
                    <G key={`c-${cell.id}`}>
                      <Rect
                        x={x}
                        y={y}
                        width={cellSize}
                        height={cellSize}
                        fill={fill}
                        stroke={isSel ? theme.colors.accent : theme.colors.border}
                        strokeWidth={isSel ? 3 : 1}
                        onPress={() => setSelected(cell.id)}
                      />
                      {solution ? (
                        <SvgText
                          x={x + cellSize / 2}
                          y={y + cellSize / 2 + 6}
                          fontSize={16}
                          fontWeight="800"
                          fill="#08101F"
                          textAnchor="middle"
                        >
                          {String(solution.gridPips[cell.r][cell.c])}
                        </SvgText>
                      ) : null}
                    </G>
                  );
                })}
              </Svg>
            </Animated.View>
          </PinchGestureHandler>
        </Animated.View>
      </PanGestureHandler>
    </View>
  );
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function regionColor(id: string): string {
  // Deterministic pastel-ish palette via hash.
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  const sat = 55;
  const light = 68;
  return hslToHex(hue, sat, light);
}

function hslToHex(h: number, s: number, l: number): string {
  const ss = s / 100;
  const ll = l / 100;
  const c = (1 - Math.abs(2 * ll - 1)) * ss;
  const hh = (h % 360) / 60;
  const x = c * (1 - Math.abs((hh % 2) - 1));
  let r1 = 0,
    g1 = 0,
    b1 = 0;
  if (hh >= 0 && hh < 1) [r1, g1, b1] = [c, x, 0];
  else if (hh >= 1 && hh < 2) [r1, g1, b1] = [x, c, 0];
  else if (hh >= 2 && hh < 3) [r1, g1, b1] = [0, c, x];
  else if (hh >= 3 && hh < 4) [r1, g1, b1] = [0, x, c];
  else if (hh >= 4 && hh < 5) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  const m = ll - c / 2;
  const r = Math.round((r1 + m) * 255);
  const g = Math.round((g1 + m) * 255);
  const b = Math.round((b1 + m) * 255);
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function toHex(n: number): string {
  const s = Math.max(0, Math.min(255, n)).toString(16);
  return s.length === 1 ? `0${s}` : s;
}

const styles = StyleSheet.create({
  wrap: { flex: 1 },
  canvas: { alignSelf: 'flex-start' },
  info: { position: 'absolute', top: 8, left: 8, right: 8, zIndex: 10 },
  infoText: {
    color: theme.colors.text,
    backgroundColor: 'rgba(10,12,18,0.75)',
    borderColor: theme.colors.border,
    borderWidth: 1,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: 10,
    fontSize: 12,
    fontWeight: '700',
  },
});


