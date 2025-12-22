import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  LayoutAnimation,
  Platform,
  UIManager,
  ActivityIndicator,
} from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import Svg, { Rect, Text as SvgText, G } from 'react-native-svg';
import { RootStackParamList } from '../navigation/RootNavigator';
import ConfidenceSummary from '../components/ui/ConfidenceSummary';
import { ExtractionResult, StageConfidence } from '../../extraction/types';
import { PuzzleSpec } from '../../model/types';
import { useExtraction } from '../hooks';

// Enable LayoutAnimation for Android
if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

type Props = NativeStackScreenProps<RootStackParamList, 'ExtractionResult'>;

const CELL_SIZE = 40;

/**
 * Get color based on confidence level for stage breakdown.
 */
function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) {
    return '#22c55e'; // Green
  } else if (confidence >= 0.6) {
    return '#eab308'; // Yellow
  } else {
    return '#dc2626'; // Red
  }
}

/**
 * Get human-readable stage name.
 */
function getStageName(stage: StageConfidence['stage']): string {
  const stageNames: Record<StageConfidence['stage'], string> = {
    BOARD_DETECTION: 'Board Detection',
    GRID_ALIGNMENT: 'Grid Alignment',
    CELL_EXTRACTION: 'Cell Extraction',
    PIP_RECOGNITION: 'Pip Recognition',
  };
  return stageNames[stage];
}

/**
 * Component displaying a preview of the extracted puzzle grid.
 */
function ExtractionGridPreview({ extractionResult }: { extractionResult: ExtractionResult }) {
  const { rows, cols, cells } = extractionResult;

  if (cells.length === 0) {
    return (
      <View style={styles.emptyGrid}>
        <Text style={styles.emptyGridText}>No cells extracted</Text>
      </View>
    );
  }

  // Create a 2D array for efficient cell lookup
  const cellGrid: (number | null)[][] = Array.from({ length: rows }, () =>
    Array(cols).fill(null)
  );
  const confidenceGrid: number[][] = Array.from({ length: rows }, () =>
    Array(cols).fill(0)
  );

  cells.forEach((cell) => {
    if (cell.row < rows && cell.col < cols) {
      cellGrid[cell.row][cell.col] = cell.value;
      confidenceGrid[cell.row][cell.col] = cell.confidence;
    }
  });

  const width = cols * CELL_SIZE;
  const height = rows * CELL_SIZE;

  return (
    <View style={styles.gridContainer}>
      <ScrollView horizontal showsHorizontalScrollIndicator={false}>
        <ScrollView showsVerticalScrollIndicator={false}>
          <Svg width={width} height={height}>
            <G>
              {cellGrid.map((row, r) =>
                row.map((value, c) => {
                  const cellConfidence = confidenceGrid[r][c];
                  const opacity = 0.5 + cellConfidence * 0.5;
                  return (
                    <React.Fragment key={`${r}-${c}`}>
                      <Rect
                        x={c * CELL_SIZE}
                        y={r * CELL_SIZE}
                        width={CELL_SIZE}
                        height={CELL_SIZE}
                        fill="#1a1b26"
                        stroke="#4a60c4"
                        strokeWidth={1}
                        opacity={opacity}
                      />
                      <SvgText
                        x={c * CELL_SIZE + CELL_SIZE / 2}
                        y={r * CELL_SIZE + CELL_SIZE / 1.5}
                        fill={value !== null ? '#e6e6e6' : '#555'}
                        fontSize={16}
                        fontWeight="bold"
                        textAnchor="middle"
                      >
                        {value !== null ? value : '?'}
                      </SvgText>
                    </React.Fragment>
                  );
                })
              )}
            </G>
          </Svg>
        </ScrollView>
      </ScrollView>
      <Text style={styles.gridDimensions}>
        {rows} x {cols} grid ({cells.length} cells)
      </Text>
    </View>
  );
}

/**
 * Component displaying individual stage confidence breakdown.
 */
function StageConfidenceBreakdown({
  stageConfidences,
}: {
  stageConfidences: StageConfidence[];
}) {
  return (
    <View style={styles.breakdownContainer}>
      {stageConfidences.map((stage) => {
        const percentage = Math.round(stage.confidence * 100);
        const color = getConfidenceColor(stage.confidence);
        return (
          <View key={stage.stage} style={styles.stageRow}>
            <Text style={styles.stageName}>{getStageName(stage.stage)}</Text>
            <View style={styles.stageConfidence}>
              <View style={styles.stageBar}>
                <View
                  style={[
                    styles.stageBarFill,
                    {
                      width: `${percentage}%`,
                      backgroundColor: color,
                    },
                  ]}
                />
              </View>
              <Text style={[styles.stagePercentage, { color }]}>{percentage}%</Text>
            </View>
          </View>
        );
      })}
    </View>
  );
}

/**
 * Screen displaying extraction results including puzzle grid preview and confidence summary.
 *
 * Features:
 * - Displays extracted puzzle grid preview with pip values
 * - Shows ConfidenceSummary component with all hints
 * - Provides 'Accept & Solve' and 'Retry Extraction' actions
 * - Shows individual stage confidence breakdown on request
 * - Loading state shown during extraction retry
 */
export default function ExtractionResultScreen({ route, navigation }: Props) {
  const { extractionResult: initialResult, sourceImageUri } = route.params;
  const [extractionResult, setExtractionResult] = useState<ExtractionResult>(initialResult);
  const [showStageBreakdown, setShowStageBreakdown] = useState(false);
  const { isExtracting, extractFromImage } = useExtraction();

  const toggleStageBreakdown = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setShowStageBreakdown(!showStageBreakdown);
  };

  /**
   * Convert extraction result to PuzzleSpec for solving.
   * Creates a simple grid where each cell is its own region.
   */
  const createPuzzleFromExtraction = (): PuzzleSpec => {
    const { rows, cols, cells } = extractionResult;

    // Create regions where each cell is its own region (basic case)
    const regions: number[][] = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => r * cols + c)
    );

    // Create region constraints (sum = pip value for each cell)
    const regionConstraints: PuzzleSpec['regionConstraints'] = {};
    cells.forEach((cell) => {
      const regionId = cell.row * cols + cell.col;
      if (cell.value !== null) {
        regionConstraints[regionId] = { type: 'sum', value: cell.value, size: 1 };
      }
    });

    return {
      name: 'Extracted Puzzle',
      rows,
      cols,
      regions,
      regionConstraints,
      maxPip: 6,
      allowDuplicates: false,
    };
  };

  const handleAcceptAndSolve = () => {
    const puzzle = createPuzzleFromExtraction();
    navigation.navigate('Solve', {
      puzzle,
      sourceText: `Extracted from image\nProcessing time: ${extractionResult.processingTimeMs}ms\nConfidence: ${Math.round(extractionResult.overallConfidence * 100)}%`,
    });
  };

  const handleRetryExtraction = useCallback(async () => {
    if (!sourceImageUri) {
      // Navigate back if no source image available
      navigation.goBack();
      return;
    }

    // Re-run extraction with the same image and update local state
    // This allows the user to retry without leaving the screen
    const newResult = await extractFromImage(sourceImageUri);
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExtractionResult(newResult);
  }, [sourceImageUri, extractFromImage, navigation]);

  // Handle failed extraction
  if (!extractionResult.success) {
    return (
      <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
        <View style={styles.errorCard}>
          <Text style={styles.errorTitle}>Extraction Failed</Text>
          <Text style={styles.errorMessage}>
            {extractionResult.error || 'Unable to extract puzzle from image.'}
          </Text>
        </View>

        <ConfidenceSummary
          confidence={extractionResult.overallConfidence}
          hints={extractionResult.hints}
          initiallyExpanded
          style={styles.confidenceSummary}
        />

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={handleRetryExtraction}
          disabled={isExtracting}
          accessibilityRole="button"
          accessibilityLabel="Retry extraction"
          accessibilityState={{ disabled: isExtracting }}
        >
          {isExtracting ? (
            <View style={styles.buttonLoadingContent}>
              <ActivityIndicator size="small" color="#fff" />
              <Text style={styles.primaryButtonText}>Extracting...</Text>
            </View>
          ) : (
            <Text style={styles.primaryButtonText}>Retry Extraction</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={() => navigation.goBack()}
          disabled={isExtracting}
        >
          <Text style={styles.secondaryButtonText}>Cancel</Text>
        </TouchableOpacity>
      </ScrollView>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      {/* Header */}
      <Text style={styles.header}>Extraction Results</Text>
      <Text style={styles.subheader}>
        Processed in {extractionResult.processingTimeMs}ms
      </Text>

      {/* Grid Preview */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Extracted Grid</Text>
        <ExtractionGridPreview extractionResult={extractionResult} />
      </View>

      {/* Confidence Summary */}
      <ConfidenceSummary
        confidence={extractionResult.overallConfidence}
        hints={extractionResult.hints}
        initiallyExpanded={extractionResult.hints.length > 0}
        style={styles.confidenceSummary}
      />

      {/* Stage Breakdown Toggle */}
      <TouchableOpacity
        style={styles.breakdownToggle}
        onPress={toggleStageBreakdown}
        accessibilityRole="button"
        accessibilityLabel={`${showStageBreakdown ? 'Hide' : 'Show'} stage confidence breakdown`}
        accessibilityState={{ expanded: showStageBreakdown }}
      >
        <Text style={styles.breakdownToggleText}>
          {showStageBreakdown ? 'Hide' : 'Show'} Stage Breakdown
        </Text>
        <Text style={styles.breakdownToggleIcon}>
          {showStageBreakdown ? '▲' : '▼'}
        </Text>
      </TouchableOpacity>

      {/* Stage Confidence Breakdown */}
      {showStageBreakdown && (
        <StageConfidenceBreakdown stageConfidences={extractionResult.stageConfidences} />
      )}

      {/* Action Buttons */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[
            styles.primaryButton,
            extractionResult.overallConfidence < 0.6 && styles.warningButton,
          ]}
          onPress={handleAcceptAndSolve}
          accessibilityRole="button"
          accessibilityLabel="Accept extraction and solve puzzle"
        >
          <Text style={styles.primaryButtonText}>Accept & Solve</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={handleRetryExtraction}
          disabled={isExtracting}
          accessibilityRole="button"
          accessibilityLabel="Retry extraction with a new image"
          accessibilityState={{ disabled: isExtracting }}
        >
          {isExtracting ? (
            <View style={styles.buttonLoadingContent}>
              <ActivityIndicator size="small" color="#4a60c4" />
              <Text style={styles.secondaryButtonText}>Extracting...</Text>
            </View>
          ) : (
            <Text style={styles.secondaryButtonText}>Retry Extraction</Text>
          )}
        </TouchableOpacity>
      </View>

      {/* Loading Overlay */}
      {isExtracting && (
        <View style={styles.loadingOverlay}>
          <View style={styles.loadingCard}>
            <ActivityIndicator size="large" color="#4a60c4" />
            <Text style={styles.loadingText}>Re-extracting puzzle...</Text>
          </View>
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c1021',
  },
  contentContainer: {
    padding: 16,
    paddingBottom: 32,
  },
  header: {
    color: '#ffffff',
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  subheader: {
    color: '#9aa5ce',
    fontSize: 14,
    marginBottom: 20,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    color: '#e6e6e6',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  gridContainer: {
    backgroundColor: '#1a1b26',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  emptyGrid: {
    backgroundColor: '#1a1b26',
    borderRadius: 12,
    padding: 32,
    alignItems: 'center',
    justifyContent: 'center',
  },
  emptyGridText: {
    color: '#9aa5ce',
    fontSize: 14,
  },
  gridDimensions: {
    color: '#9aa5ce',
    fontSize: 12,
    marginTop: 12,
  },
  confidenceSummary: {
    marginBottom: 16,
  },
  breakdownToggle: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#1a1b26',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
  },
  breakdownToggleText: {
    color: '#9aa5ce',
    fontSize: 14,
    fontWeight: '500',
  },
  breakdownToggleIcon: {
    color: '#9aa5ce',
    fontSize: 12,
  },
  breakdownContainer: {
    backgroundColor: '#1a1b26',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  stageRow: {
    marginBottom: 12,
  },
  stageName: {
    color: '#e6e6e6',
    fontSize: 14,
    marginBottom: 6,
  },
  stageConfidence: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  stageBar: {
    flex: 1,
    height: 8,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  stageBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  stagePercentage: {
    fontSize: 14,
    fontWeight: '600',
    width: 45,
    textAlign: 'right',
  },
  buttonContainer: {
    marginTop: 8,
    gap: 12,
  },
  primaryButton: {
    backgroundColor: '#4a60c4',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  warningButton: {
    backgroundColor: '#b45309',
  },
  primaryButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: '#4a60c4',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#4a60c4',
    fontSize: 16,
    fontWeight: 'bold',
  },
  errorCard: {
    backgroundColor: 'rgba(220, 38, 38, 0.15)',
    borderRadius: 12,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: 'rgba(220, 38, 38, 0.3)',
  },
  errorTitle: {
    color: '#dc2626',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  errorMessage: {
    color: '#e6e6e6',
    fontSize: 14,
    lineHeight: 20,
  },
  buttonLoadingContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(12, 16, 33, 0.85)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingCard: {
    backgroundColor: '#1a1b26',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
    gap: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
  loadingText: {
    color: '#e6e6e6',
    fontSize: 16,
    fontWeight: '500',
  },
});
