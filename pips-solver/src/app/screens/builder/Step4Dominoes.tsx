/**
 * Step 4: Dominoes
 * Enter domino pairs for the puzzle
 */

import React, { useState } from 'react';
import { ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import { BuilderAction, DominoPair, OverlayBuilderState } from '../../../model/overlayTypes';
import {
  getPipPositions,
  parseDominoShorthand,
  validateDominoCount,
} from '../../../utils/dominoUtils';

interface Props {
  state: OverlayBuilderState;
  dispatch: React.Dispatch<BuilderAction>;
}

export default function Step4Dominoes({ state, dispatch }: Props) {
  const { grid, dominoes } = state;
  const [quickInput, setQuickInput] = useState('');

  // Count valid cells (non-holes)
  const cellCount = grid.holes.flat().filter(h => !h).length;
  const validation = validateDominoCount(dominoes.dominoes, cellCount);

  const handleQuickApply = () => {
    if (quickInput.trim()) {
      const parsed = parseDominoShorthand(quickInput);
      if (parsed.length > 0) {
        dispatch({ type: 'SET_DOMINOES', dominoes: parsed });
      }
      setQuickInput('');
    }
  };

  const handleAutoFill = () => {
    dispatch({ type: 'AUTO_FILL_DOMINOES' });
  };

  const handleAddDomino = () => {
    dispatch({ type: 'ADD_DOMINO', pip1: 0, pip2: 0 });
  };

  const handleRemoveDomino = (index: number) => {
    dispatch({ type: 'REMOVE_DOMINO', index });
  };

  const handleCyclePip = (dominoIndex: number, half: 0 | 1, direction: 1 | -1) => {
    dispatch({ type: 'CYCLE_DOMINO_PIP', dominoIndex, half, direction });
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Quick entry */}
      <View style={styles.quickContainer}>
        <TextInput
          style={styles.quickInput}
          value={quickInput}
          onChangeText={setQuickInput}
          placeholder="Type: 61 33 36 43 14"
          placeholderTextColor="#666"
          // Use a keyboard that has a space bar so users can type "61 33 36..."
          // We still validate by stripping non 0-6 digits in parseDominoShorthand.
          keyboardType="default"
          autoCapitalize="none"
          autoCorrect={false}
          returnKeyType="done"
          blurOnSubmit
          onSubmitEditing={handleQuickApply}
        />
        <TouchableOpacity style={styles.quickButton} onPress={handleQuickApply}>
          <Text style={styles.quickButtonText}>Set</Text>
        </TouchableOpacity>
      </View>

      <Text style={styles.hint}>Type digit pairs (61 = [6,1]) or tap dominoes to adjust</Text>

      {/* Validation status */}
      <View style={[styles.validationBanner, validation.valid && styles.validationBannerValid]}>
        <Text style={[styles.validationText, validation.valid && styles.validationTextValid]}>
          {validation.message}
        </Text>
        {!validation.valid && (
          <TouchableOpacity style={styles.autoFillButton} onPress={handleAutoFill}>
            <Text style={styles.autoFillButtonText}>Auto-fill {Math.floor(cellCount / 2)}</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Domino list */}
      <View style={styles.dominoList}>
        {dominoes.dominoes.map((domino, index) => (
          <View key={index} style={styles.dominoItem}>
            <Text style={styles.dominoIndex}>{index + 1}</Text>
            <DominoVisual
              domino={domino}
              onTapLeft={() => handleCyclePip(index, 0, 1)}
              onTapRight={() => handleCyclePip(index, 1, 1)}
            />
            <TouchableOpacity style={styles.removeButton} onPress={() => handleRemoveDomino(index)}>
              <Text style={styles.removeButtonText}>✕</Text>
            </TouchableOpacity>
          </View>
        ))}
      </View>

      {/* Add domino button */}
      <TouchableOpacity style={styles.addButton} onPress={handleAddDomino}>
        <Text style={styles.addButtonText}>+ Add Domino</Text>
      </TouchableOpacity>
    </ScrollView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Domino Visual Component
// ════════════════════════════════════════════════════════════════════════════

interface DominoVisualProps {
  domino: DominoPair;
  onTapLeft?: () => void;
  onTapRight?: () => void;
  size?: number;
}

function DominoVisual({ domino, onTapLeft, onTapRight, size = 36 }: DominoVisualProps) {
  return (
    <View style={dominoStyles.container}>
      <TouchableOpacity
        style={[dominoStyles.half, { width: size, height: size }]}
        onPress={onTapLeft}
        activeOpacity={0.7}
      >
        <PipDisplay value={domino[0]} size={size} />
      </TouchableOpacity>
      <View style={dominoStyles.divider} />
      <TouchableOpacity
        style={[dominoStyles.half, { width: size, height: size }]}
        onPress={onTapRight}
        activeOpacity={0.7}
      >
        <PipDisplay value={domino[1]} size={size} />
      </TouchableOpacity>
    </View>
  );
}

interface PipDisplayProps {
  value: number;
  size: number;
}

function PipDisplay({ value, size }: PipDisplayProps) {
  const positions = getPipPositions(value);
  const pipSize = size * 0.18;

  return (
    <View style={{ width: size, height: size, position: 'relative' }}>
      {positions.map(([x, y], i) => (
        <View
          key={i}
          style={{
            position: 'absolute',
            left: x * size - pipSize / 2,
            top: y * size - pipSize / 2,
            width: pipSize,
            height: pipSize,
            borderRadius: pipSize / 2,
            backgroundColor: '#1a1a2e',
          }}
        />
      ))}
    </View>
  );
}

const dominoStyles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 6,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
    elevation: 3,
  },
  half: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#fff',
  },
  divider: {
    width: 1,
    backgroundColor: '#ccc',
  },
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  quickContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 8,
  },
  quickInput: {
    flex: 1,
    backgroundColor: '#222',
    color: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    fontSize: 16,
    fontFamily: 'monospace',
  },
  quickButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    borderRadius: 8,
    justifyContent: 'center',
  },
  quickButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  hint: {
    color: '#888',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 12,
  },
  validationBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#332200',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  validationBannerValid: {
    backgroundColor: '#003322',
  },
  validationText: {
    color: '#ffcc00',
    fontSize: 14,
    fontWeight: '500',
    flex: 1,
  },
  validationTextValid: {
    color: '#00cc66',
  },
  autoFillButton: {
    backgroundColor: '#444',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
    marginLeft: 8,
  },
  autoFillButtonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: '600',
  },
  dominoList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    justifyContent: 'center',
    marginBottom: 16,
  },
  dominoItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#222',
    padding: 8,
    borderRadius: 8,
    gap: 6,
  },
  dominoIndex: {
    color: '#666',
    fontSize: 12,
    width: 16,
    textAlign: 'center',
  },
  removeButton: {
    width: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  removeButtonText: {
    color: '#ff4444',
    fontSize: 14,
  },
  addButton: {
    backgroundColor: '#333',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  addButtonText: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
});
