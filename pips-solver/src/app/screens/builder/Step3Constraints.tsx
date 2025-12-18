/**
 * Step 3: Constraints
 * Set constraints for each region
 */

import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Modal,
} from 'react-native';
import Svg, { Rect, Text as SvgText } from 'react-native-svg';
import {
  OverlayBuilderState,
  BuilderAction,
  ConstraintDef,
  ConstraintType,
  ConstraintOp,
} from '../../../model/overlayTypes';
import { getConstraintLabel } from '../../../utils/constraintParser';

interface Props {
  state: OverlayBuilderState;
  dispatch: React.Dispatch<BuilderAction>;
}

export default function Step3Constraints({ state, dispatch }: Props) {
  const { grid, regions, constraints } = state;
  const { palette, regionGrid } = regions;
  const { regionConstraints, selectedRegion } = constraints;

  const [shorthandInput, setShorthandInput] = useState('');
  const [showEditor, setShowEditor] = useState(false);

  // Get unique regions used in the grid
  const usedRegions = new Set<number>();
  regionGrid.forEach(row =>
    row.forEach(cell => {
      if (cell !== null) usedRegions.add(cell);
    })
  );
  const sortedRegions = Array.from(usedRegions).sort((a, b) => a - b);

  const handleRegionSelect = (index: number) => {
    dispatch({ type: 'SELECT_REGION', regionIndex: index });
    setShowEditor(true);
  };

  const handleShorthandApply = () => {
    if (shorthandInput.trim()) {
      dispatch({ type: 'APPLY_CONSTRAINT_SHORTHAND', shorthand: shorthandInput });
      setShorthandInput('');
    }
  };

  const handleConstraintChange = (constraint: ConstraintDef) => {
    if (selectedRegion !== null) {
      dispatch({
        type: 'SET_CONSTRAINT',
        regionIndex: selectedRegion,
        constraint,
      });
    }
  };

  const cellWidth = (grid.bounds.right - grid.bounds.left) / grid.cols;
  const cellHeight = (grid.bounds.bottom - grid.bounds.top) / grid.rows;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Shorthand input */}
      <View style={styles.shorthandContainer}>
        <TextInput
          style={styles.shorthandInput}
          value={shorthandInput}
          onChangeText={setShorthandInput}
          placeholder="Quick: A=8 B>4 C= Dx"
          placeholderTextColor="#666"
          autoCapitalize="characters"
          onSubmitEditing={handleShorthandApply}
        />
        <TouchableOpacity style={styles.shorthandButton} onPress={handleShorthandApply}>
          <Text style={styles.shorthandButtonText}>Apply</Text>
        </TouchableOpacity>
      </View>

      {/* Grid preview */}
      <View style={styles.gridContainer}>
        <Svg width="100%" height={200} viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
          {/* Cells */}
          {regionGrid.map((row, r) =>
            row.map((regionIndex, c) => {
              if (regionIndex === null || grid.holes[r]?.[c]) {
                return grid.holes[r]?.[c] ? (
                  <Rect
                    key={`cell-${r}-${c}`}
                    x={10 + (c / grid.cols) * 80}
                    y={10 + (r / grid.rows) * 80}
                    width={80 / grid.cols}
                    height={80 / grid.rows}
                    fill="#333"
                  />
                ) : null;
              }

              const isSelected = selectedRegion === regionIndex;
              return (
                <Rect
                  key={`cell-${r}-${c}`}
                  x={10 + (c / grid.cols) * 80}
                  y={10 + (r / grid.rows) * 80}
                  width={80 / grid.cols}
                  height={80 / grid.rows}
                  fill={palette.colors[regionIndex]}
                  stroke={isSelected ? '#fff' : 'rgba(255,255,255,0.3)'}
                  strokeWidth={isSelected ? 2 : 0.5}
                  onPress={() => handleRegionSelect(regionIndex)}
                />
              );
            })
          )}
        </Svg>
      </View>

      {/* Region list */}
      <View style={styles.regionList}>
        <Text style={styles.sectionTitle}>Region Constraints:</Text>
        {sortedRegions.map(regionIndex => {
          const constraint = regionConstraints[regionIndex];
          const label = getConstraintLabel(constraint);
          const isSelected = selectedRegion === regionIndex;

          return (
            <TouchableOpacity
              key={regionIndex}
              style={[styles.regionItem, isSelected && styles.regionItemSelected]}
              onPress={() => handleRegionSelect(regionIndex)}
            >
              <View
                style={[
                  styles.regionColor,
                  { backgroundColor: palette.colors[regionIndex] },
                ]}
              />
              <Text style={styles.regionLabel}>{palette.labels[regionIndex]}</Text>
              <Text style={styles.constraintLabel}>{label}</Text>
            </TouchableOpacity>
          );
        })}
      </View>

      <Text style={styles.hint}>
        Tap a region to edit. Use shorthand: A=8 (sum), A= (equal), Ax (different)
      </Text>

      {/* Constraint Editor Modal */}
      <Modal visible={showEditor} animationType="slide" transparent>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                Edit Region {selectedRegion !== null ? palette.labels[selectedRegion] : ''}
              </Text>
              <TouchableOpacity onPress={() => setShowEditor(false)}>
                <Text style={styles.modalClose}>✕</Text>
              </TouchableOpacity>
            </View>

            <ConstraintEditor
              constraint={selectedRegion !== null ? regionConstraints[selectedRegion] : undefined}
              onChange={handleConstraintChange}
            />

            <TouchableOpacity
              style={styles.modalDoneButton}
              onPress={() => setShowEditor(false)}
            >
              <Text style={styles.modalDoneButtonText}>Done</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </ScrollView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Constraint Editor Component
// ════════════════════════════════════════════════════════════════════════════

interface ConstraintEditorProps {
  constraint?: ConstraintDef;
  onChange: (constraint: ConstraintDef) => void;
}

function ConstraintEditor({ constraint, onChange }: ConstraintEditorProps) {
  const currentType = constraint?.type || 'none';
  const currentOp = constraint?.op || '==';
  const currentValue = constraint?.value ?? 0;

  const handleTypeChange = (type: ConstraintType) => {
    if (type === 'sum') {
      onChange({ type, op: '==', value: 6 });
    } else {
      onChange({ type });
    }
  };

  const handleOpChange = (op: ConstraintOp) => {
    onChange({ ...constraint, type: 'sum', op, value: currentValue });
  };

  const handleValueChange = (value: number) => {
    onChange({ ...constraint, type: 'sum', op: currentOp, value });
  };

  return (
    <View style={editorStyles.container}>
      {/* Type selector */}
      <Text style={editorStyles.label}>Type:</Text>
      <View style={editorStyles.typeRow}>
        {(['none', 'sum', 'all_equal', 'all_different'] as ConstraintType[]).map(type => (
          <TouchableOpacity
            key={type}
            style={[
              editorStyles.typeButton,
              currentType === type && editorStyles.typeButtonSelected,
            ]}
            onPress={() => handleTypeChange(type)}
          >
            <Text
              style={[
                editorStyles.typeButtonText,
                currentType === type && editorStyles.typeButtonTextSelected,
              ]}
            >
              {type === 'none' ? 'None' : type === 'sum' ? 'Sum' : type === 'all_equal' ? '=' : '✕'}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Sum options */}
      {currentType === 'sum' && (
        <>
          <Text style={editorStyles.label}>Operator:</Text>
          <View style={editorStyles.typeRow}>
            {(['==', '<', '>', '!='] as ConstraintOp[]).map(op => (
              <TouchableOpacity
                key={op}
                style={[
                  editorStyles.typeButton,
                  currentOp === op && editorStyles.typeButtonSelected,
                ]}
                onPress={() => handleOpChange(op)}
              >
                <Text
                  style={[
                    editorStyles.typeButtonText,
                    currentOp === op && editorStyles.typeButtonTextSelected,
                  ]}
                >
                  {op === '==' ? '=' : op === '!=' ? '≠' : op}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={editorStyles.label}>Value:</Text>
          <View style={editorStyles.valueRow}>
            <TouchableOpacity
              style={editorStyles.valueButton}
              onPress={() => handleValueChange(Math.max(0, currentValue - 1))}
            >
              <Text style={editorStyles.valueButtonText}>−</Text>
            </TouchableOpacity>
            <Text style={editorStyles.valueDisplay}>{currentValue}</Text>
            <TouchableOpacity
              style={editorStyles.valueButton}
              onPress={() => handleValueChange(Math.min(99, currentValue + 1))}
            >
              <Text style={editorStyles.valueButtonText}>+</Text>
            </TouchableOpacity>
          </View>
        </>
      )}
    </View>
  );
}

const editorStyles = StyleSheet.create({
  container: {
    padding: 16,
  },
  label: {
    color: '#888',
    fontSize: 13,
    marginBottom: 8,
    marginTop: 12,
  },
  typeRow: {
    flexDirection: 'row',
    gap: 8,
  },
  typeButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    backgroundColor: '#333',
    borderRadius: 8,
    alignItems: 'center',
  },
  typeButtonSelected: {
    backgroundColor: '#007AFF',
  },
  typeButtonText: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
  typeButtonTextSelected: {
    color: '#fff',
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  valueButton: {
    width: 48,
    height: 48,
    backgroundColor: '#333',
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  valueButtonText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '600',
  },
  valueDisplay: {
    color: '#fff',
    fontSize: 32,
    fontWeight: 'bold',
    minWidth: 60,
    textAlign: 'center',
  },
});

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  shorthandContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  shorthandInput: {
    flex: 1,
    backgroundColor: '#222',
    color: '#fff',
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: 8,
    fontSize: 14,
  },
  shorthandButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    borderRadius: 8,
    justifyContent: 'center',
  },
  shorthandButtonText: {
    color: '#fff',
    fontWeight: '600',
  },
  gridContainer: {
    backgroundColor: '#111',
    borderRadius: 12,
    padding: 8,
    marginBottom: 16,
  },
  regionList: {
    marginBottom: 16,
  },
  sectionTitle: {
    color: '#888',
    fontSize: 13,
    marginBottom: 8,
  },
  regionItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#222',
    padding: 12,
    borderRadius: 8,
    marginBottom: 6,
  },
  regionItemSelected: {
    backgroundColor: '#333',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  regionColor: {
    width: 24,
    height: 24,
    borderRadius: 6,
    marginRight: 12,
  },
  regionLabel: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    flex: 1,
  },
  constraintLabel: {
    color: '#888',
    fontSize: 16,
  },
  hint: {
    color: '#888',
    fontSize: 13,
    textAlign: 'center',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a2e',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 30,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  modalTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  modalClose: {
    color: '#888',
    fontSize: 24,
  },
  modalDoneButton: {
    backgroundColor: '#007AFF',
    margin: 16,
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalDoneButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});
