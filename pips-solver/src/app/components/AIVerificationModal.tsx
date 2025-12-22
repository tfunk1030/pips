/**
 * AI Verification Modal
 * Shows what the AI extracted before applying it to the builder state
 */

import React, { useState, useMemo } from 'react';
import { Modal, ScrollView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { BoardExtractionResult, DominoExtractionResult, RawResponses } from '../../model/overlayTypes';
import ExtractionComparisonModal from './ExtractionComparisonModal';

interface Props {
  visible: boolean;
  boardResult: BoardExtractionResult;
  dominoResult: DominoExtractionResult;
  onAccept: () => void;
  onReject: () => void;
  /** Optional raw responses from multiple models for comparison */
  rawResponses?: RawResponses | null;
}

export default function AIVerificationModal({
  visible,
  boardResult,
  dominoResult,
  onAccept,
  onReject,
  rawResponses,
}: Props) {
  // State for comparison modal visibility
  const [showComparisonModal, setShowComparisonModal] = useState(false);

  // Determine if multiple models were used (at least 2 different models in responses)
  const hasMultipleModels = useMemo(() => {
    if (!rawResponses) return false;

    // Collect unique model names from board and domino responses
    const modelNames = new Set<string>();

    for (const response of rawResponses.board) {
      modelNames.add(response.model);
    }
    for (const response of rawResponses.dominoes) {
      modelNames.add(response.model);
    }

    return modelNames.size >= 2;
  }, [rawResponses]);

  // Format shape and regions for display
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

  return (
    <Modal visible={visible} animationType="slide" presentationStyle="pageSheet">
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Verify AI Extraction</Text>
          <Text style={styles.subtitle}>Review before applying</Text>
        </View>

        <ScrollView style={styles.content} contentContainerStyle={styles.contentContainer}>
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
        </ScrollView>

        {/* Compare Models button - only shown when multiple models were used */}
        {hasMultipleModels && (
          <View style={styles.compareButtonContainer}>
            <TouchableOpacity
              style={[styles.button, styles.compareButton]}
              onPress={() => setShowComparisonModal(true)}
            >
              <Text style={styles.buttonText}>Compare Models</Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.buttons}>
          <TouchableOpacity style={[styles.button, styles.rejectButton]} onPress={onReject}>
            <Text style={styles.buttonText}>Reject - I'll do it manually</Text>
          </TouchableOpacity>
          <TouchableOpacity style={[styles.button, styles.acceptButton]} onPress={onAccept}>
            <Text style={styles.buttonText}>Accept - Apply this</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* Extraction Comparison Modal */}
      <ExtractionComparisonModal
        visible={showComparisonModal}
        rawResponses={rawResponses ?? null}
        onClose={() => setShowComparisonModal(false)}
        onAccept={() => {
          setShowComparisonModal(false);
          onAccept();
        }}
      />
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a1a',
  },
  header: {
    padding: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: 20,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
    marginBottom: 8,
  },
  text: {
    fontSize: 14,
    color: '#ccc',
    marginBottom: 4,
  },
  textSmall: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  gridContainer: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  gridLine: {
    fontFamily: 'Courier',
    fontSize: 18,
    lineHeight: 24,
  },
  gridChar: {
    color: '#fff',
    marginRight: 8,
  },
  holeChar: {
    color: '#666',
  },
  emptyChar: {
    color: '#888',
  },
  dominoContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  domino: {
    fontSize: 14,
    color: '#ccc',
    backgroundColor: '#333',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    fontFamily: 'Courier',
  },
  buttons: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  button: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  rejectButton: {
    backgroundColor: '#666',
  },
  acceptButton: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  compareButtonContainer: {
    paddingHorizontal: 16,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  compareButton: {
    backgroundColor: '#2196F3',
    flex: 0,
  },
});
