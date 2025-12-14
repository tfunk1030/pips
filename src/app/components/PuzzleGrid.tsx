import React, { useMemo, useState } from 'react';
import { Text, View, PanResponder, GestureResponderEvent, PanResponderGestureState } from 'react-native';
import Svg, { Rect, G, Text as SvgText, Line } from 'react-native-svg';
import { PuzzleSpec, SolutionGrid } from '../../model/types';

interface Props {
  puzzle: PuzzleSpec;
  solution?: SolutionGrid;
}

const regionPalette = ['#2a2f4f', '#4a60c4', '#519872', '#b48b7d', '#c44569', '#f2c14e', '#2c699a'];

const cellSize = 48;

export default function PuzzleGrid({ puzzle, solution }: Props) {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });

  const panResponder = useMemo(
    () =>
      PanResponder.create({
        onMoveShouldSetPanResponder: () => true,
        onPanResponderMove: (_: GestureResponderEvent, gesture: PanResponderGestureState) => {
          setOffset({ x: gesture.dx, y: gesture.dy });
        },
        onPanResponderRelease: (_evt, gesture) => {
          setOffset((prev) => ({ x: prev.x + gesture.dx, y: prev.y + gesture.dy }));
        },
        onPanResponderTerminationRequest: () => true,
        onPanResponderGrant: () => {},
        onPanResponderEnd: () => {},
      }),
    [],
  );

  const width = puzzle.cols * cellSize * scale;
  const height = puzzle.rows * cellSize * scale;

  return (
    <View {...panResponder.panHandlers}>
      <Svg width={width} height={height}>
        <G transform={`translate(${offset.x},${offset.y}) scale(${scale})`}>
          {puzzle.regions.map((row, r) =>
            row.map((region, c) => {
              const color = regionPalette[region % regionPalette.length];
              return (
                <Rect
                  key={`${r}-${c}`}
                  x={c * cellSize}
                  y={r * cellSize}
                  width={cellSize}
                  height={cellSize}
                  fill={color}
                  stroke="#111"
                  strokeWidth={1}
                  opacity={0.8}
                />
              );
            }),
          )}
          {solution &&
            solution.gridPips.map((row, r) =>
              row.map((pip, c) => (
                <SvgText key={`pip-${r}-${c}`} x={c * cellSize + cellSize / 2} y={r * cellSize + cellSize / 1.6} fill="#fff" fontSize={18} textAnchor="middle">
                  {pip}
                </SvgText>
              )),
            )}
          {solution &&
            solution.dominoes.map((domino) => {
              const [[r1, c1], [r2, c2]] = domino.cells;
              return (
                <Line
                  key={domino.id}
                  x1={c1 * cellSize + cellSize / 2}
                  y1={r1 * cellSize + cellSize / 2}
                  x2={c2 * cellSize + cellSize / 2}
                  y2={r2 * cellSize + cellSize / 2}
                  stroke="#f2c14e"
                  strokeWidth={4}
                />
              );
            })}
        </G>
      </Svg>
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', paddingHorizontal: 12, marginTop: 6 }}>
        <Text style={{ color: '#9aa5ce' }}>Pinch via OS zoom; drag to pan.</Text>
        <Text style={{ color: '#9aa5ce' }}>Scale: {scale.toFixed(1)}</Text>
      </View>
    </View>
  );
}
