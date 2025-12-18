/**
 * OverlayBuilder Screen
 * Main coordinator for the 4-step puzzle creation wizard
 */

import React, { useReducer, useEffect, useCallback, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Alert,
  SafeAreaView,
  TextInput,
  Modal,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import {
  OverlayBuilderState,
  BuilderStep,
  createInitialBuilderState,
} from '../../model/overlayTypes';
import { saveDraft, loadDraft, deleteDraft } from '../../storage/drafts';
import { buildPuzzleSpec, validateBuilderState, getBuilderStats } from '../../utils/specBuilder';
import { savePuzzle, getSettings } from '../../storage/puzzles';
import { validatePuzzleSpec } from '../../validator/validateSpec';
import { extractPuzzleFromImage, convertAIResultToBuilderState, ExtractionProgress } from '../../services/aiExtraction';
import { builderReducer, countValidCells } from '../../state/builderReducer';

// Step components
import Step1GridAlignment from './builder/Step1GridAlignment';
import Step2RegionPainting from './builder/Step2RegionPainting';
import Step3Constraints from './builder/Step3Constraints';
import Step4Dominoes from './builder/Step4Dominoes';

// ════════════════════════════════════════════════════════════════════════════
// Component
// ════════════════════════════════════════════════════════════════════════════

interface Props {
  navigation: any;
  route: {
    params?: {
      draftId?: string;
    };
  };
}

export default function OverlayBuilderScreen({ navigation, route }: Props) {
  const [state, dispatch] = useReducer(builderReducer, null, createInitialBuilderState);
  const saveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const hasImageRef = useRef(false);
  const [showNameModal, setShowNameModal] = useState(false);
  const [puzzleName, setPuzzleName] = useState('');
  const [aiProgress, setAIProgress] = useState<string | null>(null);

  // Load draft if provided
  useEffect(() => {
    const loadExistingDraft = async () => {
      if (route.params?.draftId) {
        const draft = await loadDraft(route.params.draftId);
        if (draft) {
          dispatch({ type: 'LOAD_DRAFT', state: draft });
          hasImageRef.current = !!draft.image;
        }
      }
    };
    loadExistingDraft();
  }, [route.params?.draftId]);

  // Auto-save on state changes (debounced)
  useEffect(() => {
    if (!state.image) return; // Don't save until image is selected

    if (saveTimeoutRef.current) {
      clearTimeout(saveTimeoutRef.current);
    }

    saveTimeoutRef.current = setTimeout(() => {
      saveDraft(state);
    }, 500);

    return () => {
      if (saveTimeoutRef.current) {
        clearTimeout(saveTimeoutRef.current);
      }
    };
  }, [state]);

  // Pick image on mount if no draft
  useEffect(() => {
    if (!route.params?.draftId && !hasImageRef.current) {
      pickImage();
    }
  }, []);

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: false,
        quality: 0.8,
        base64: true,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        dispatch({
          type: 'SET_IMAGE',
          image: {
            uri: asset.uri,
            width: asset.width,
            height: asset.height,
            base64: asset.base64 || undefined,
          },
        });
        hasImageRef.current = true;
      } else if (!hasImageRef.current) {
        // User cancelled without selecting - go back
        navigation.goBack();
      }
    } catch (error) {
      console.error('Image picker error:', error);
      Alert.alert('Error', 'Failed to pick image');
      if (!hasImageRef.current) {
        navigation.goBack();
      }
    }
  };

  const handleAIExtract = useCallback(async () => {
    if (!state.image?.base64) {
      Alert.alert('Error', 'No image available for AI extraction');
      return;
    }

    // Get API key from settings
    const settings = await getSettings();
    if (!settings.anthropicApiKey) {
      Alert.alert(
        'API Key Required',
        'Please add your Anthropic API key in Settings to use AI extraction.',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Go to Settings',
            onPress: () => navigation.navigate('Settings'),
          },
        ]
      );
      return;
    }

    dispatch({ type: 'AI_START' });
    setAIProgress('Starting extraction...');

    const result = await extractPuzzleFromImage(
      state.image.base64,
      settings.anthropicApiKey,
      (progress: ExtractionProgress) => {
        setAIProgress(progress.message);
      }
    );

    if (result.success && result.result) {
      const converted = convertAIResultToBuilderState(result.result);
      dispatch({
        type: 'AI_SUCCESS',
        grid: converted.grid,
        regions: converted.regions,
        constraints: converted.constraints,
        dominoes: converted.dominoes,
        reasoning: result.result.reasoning,
      });
      setAIProgress(null);

      // Show appropriate message based on partial vs full success
      if (result.partial) {
        Alert.alert(
          'Partial Success',
          'Board structure extracted successfully, but domino extraction failed. Please add dominoes manually in Step 4.',
          [{ text: 'OK' }]
        );
      } else {
        Alert.alert('Success', 'AI extraction complete! Review and adjust the results.');
      }
    } else {
      dispatch({ type: 'AI_ERROR', error: result.error || 'Unknown error' });
      setAIProgress(null);
      Alert.alert('Extraction Failed', result.error || 'Failed to extract puzzle data');
    }
  }, [state.image, navigation]);

  const goToStep = (step: BuilderStep) => {
    dispatch({ type: 'SET_STEP', step });
  };

  const handleNext = () => {
    if (state.step < 4) {
      // Update expected domino count when moving to step 4
      if (state.step === 3) {
        const cellCount = countValidCells(state.grid.holes);
        dispatch({ type: 'AUTO_FILL_DOMINOES' });
      }
      goToStep((state.step + 1) as BuilderStep);
    }
  };

  const handleBack = () => {
    if (state.step > 1) {
      goToStep((state.step - 1) as BuilderStep);
    } else {
      // Prompt to save draft before leaving
      Alert.alert(
        'Save Progress?',
        'Do you want to save your progress?',
        [
          {
            text: 'Discard',
            style: 'destructive',
            onPress: async () => {
              await deleteDraft(state.draftId);
              navigation.goBack();
            },
          },
          {
            text: 'Save & Exit',
            onPress: async () => {
              await saveDraft(state);
              navigation.goBack();
            },
          },
          { text: 'Cancel', style: 'cancel' },
        ]
      );
    }
  };

  const handleFinish = useCallback(() => {
    // Validate before finishing
    const validation = validateBuilderState(state);

    if (!validation.valid) {
      Alert.alert(
        'Invalid Puzzle',
        validation.errors.join('\n'),
        [{ text: 'OK' }]
      );
      return;
    }

    if (validation.warnings.length > 0) {
      Alert.alert(
        'Warnings',
        `${validation.warnings.join('\n')}\n\nContinue anyway?`,
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Continue',
            onPress: () => setShowNameModal(true),
          },
        ]
      );
      return;
    }

    // Show name prompt
    setShowNameModal(true);
  }, [state]);

  const handleSavePuzzle = useCallback(async () => {
    const name = puzzleName.trim() || `Puzzle ${new Date().toLocaleDateString()}`;

    // Build the puzzle spec
    const result = buildPuzzleSpec(state, name);

    if (!result.success || !result.spec) {
      Alert.alert('Error', result.errors.join('\n'));
      return;
    }

    // Validate the spec
    const specValidation = validatePuzzleSpec(result.spec);
    if (!specValidation.valid) {
      Alert.alert(
        'Invalid Puzzle Spec',
        specValidation.errors.join('\n'),
        [{ text: 'OK' }]
      );
      return;
    }

    try {
      // Save the puzzle
      const savedPuzzle = await savePuzzle(result.spec, result.yaml || '');

      // Delete the draft
      await deleteDraft(state.draftId);

      // Close modal
      setShowNameModal(false);
      setPuzzleName('');

      // Navigate to the puzzle viewer
      navigation.replace('Viewer', { puzzleId: savedPuzzle.id });
    } catch (error) {
      Alert.alert('Error', `Failed to save puzzle: ${error}`);
    }
  }, [state, puzzleName, navigation]);

  const renderStepContent = () => {
    switch (state.step) {
      case 1:
        return (
          <Step1GridAlignment
            state={state}
            dispatch={dispatch}
            onPickNewImage={pickImage}
            onAIExtract={handleAIExtract}
            aiProgress={aiProgress}
          />
        );
      case 2:
        return <Step2RegionPainting state={state} dispatch={dispatch} />;
      case 3:
        return <Step3Constraints state={state} dispatch={dispatch} />;
      case 4:
        return <Step4Dominoes state={state} dispatch={dispatch} />;
      default:
        return null;
    }
  };

  const stepTitles = ['Grid', 'Regions', 'Constraints', 'Dominoes'];

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack} style={styles.backButton}>
          <Text style={styles.backButtonText}>
            {state.step === 1 ? '✕' : '←'}
          </Text>
        </TouchableOpacity>
        <Text style={styles.title}>{stepTitles[state.step - 1]}</Text>
        <Text style={styles.stepIndicator}>{state.step}/4</Text>
      </View>

      {/* Step Progress */}
      <View style={styles.progressContainer}>
        {[1, 2, 3, 4].map(step => (
          <View
            key={step}
            style={[
              styles.progressDot,
              state.step === step && styles.progressDotActive,
              state.step > step && styles.progressDotComplete,
            ]}
          />
        ))}
      </View>

      {/* Step Content */}
      <View style={styles.content}>{renderStepContent()}</View>

      {/* Footer */}
      <View style={styles.footer}>
        {state.step > 1 && (
          <TouchableOpacity
            style={[styles.footerButton, styles.footerButtonSecondary]}
            onPress={() => goToStep((state.step - 1) as BuilderStep)}
          >
            <Text style={styles.footerButtonTextSecondary}>← Back</Text>
          </TouchableOpacity>
        )}

        {state.step < 4 ? (
          <TouchableOpacity
            style={[styles.footerButton, styles.footerButtonPrimary]}
            onPress={handleNext}
          >
            <Text style={styles.footerButtonText}>Next →</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.footerButton, styles.footerButtonSuccess]}
            onPress={handleFinish}
          >
            <Text style={styles.footerButtonText}>Create Puzzle</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Name Prompt Modal */}
      <Modal
        visible={showNameModal}
        animationType="fade"
        transparent
        onRequestClose={() => setShowNameModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Name Your Puzzle</Text>
            <TextInput
              style={styles.modalInput}
              value={puzzleName}
              onChangeText={setPuzzleName}
              placeholder="Enter puzzle name..."
              placeholderTextColor="#666"
              autoFocus
              onSubmitEditing={handleSavePuzzle}
            />
            <View style={styles.modalStats}>
              {(() => {
                const stats = getBuilderStats(state);
                return (
                  <Text style={styles.modalStatsText}>
                    {stats.cellCount} cells • {stats.regionCount} regions • {stats.dominoCount} dominoes
                  </Text>
                );
              })()}
            </View>
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalButtonCancel]}
                onPress={() => {
                  setShowNameModal(false);
                  setPuzzleName('');
                }}
              >
                <Text style={styles.modalButtonTextCancel}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButton, styles.modalButtonSave]}
                onPress={handleSavePuzzle}
              >
                <Text style={styles.modalButtonText}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#1a1a2e',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 20,
    color: '#fff',
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
  },
  stepIndicator: {
    fontSize: 14,
    color: '#888',
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
    gap: 8,
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#444',
  },
  progressDotActive: {
    backgroundColor: '#007AFF',
    width: 24,
  },
  progressDotComplete: {
    backgroundColor: '#34C759',
  },
  content: {
    flex: 1,
  },
  footer: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  footerButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  footerButtonPrimary: {
    backgroundColor: '#007AFF',
  },
  footerButtonSecondary: {
    backgroundColor: '#333',
  },
  footerButtonSuccess: {
    backgroundColor: '#34C759',
  },
  footerButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  footerButtonTextSecondary: {
    color: '#ccc',
    fontSize: 16,
    fontWeight: '600',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#1a1a2e',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 360,
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#fff',
    textAlign: 'center',
    marginBottom: 20,
  },
  modalInput: {
    backgroundColor: '#222',
    color: '#fff',
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderRadius: 8,
    fontSize: 16,
    marginBottom: 12,
  },
  modalStats: {
    marginBottom: 20,
  },
  modalStatsText: {
    color: '#888',
    fontSize: 13,
    textAlign: 'center',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  modalButton: {
    flex: 1,
    paddingVertical: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalButtonCancel: {
    backgroundColor: '#333',
  },
  modalButtonSave: {
    backgroundColor: '#34C759',
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  modalButtonTextCancel: {
    color: '#ccc',
    fontSize: 16,
    fontWeight: '600',
  },
});
