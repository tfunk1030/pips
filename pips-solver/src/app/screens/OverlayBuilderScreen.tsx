/**
 * OverlayBuilder Screen
 * Main coordinator for the 4-step puzzle creation wizard
 */

import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import React, { useCallback, useEffect, useReducer, useRef, useState } from 'react';
import { Alert, Modal, StyleSheet, TextInput, TouchableOpacity, View } from 'react-native';
import Animated, { FadeIn, FadeInDown, FadeInUp } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  AIExtractionResult,
  BuilderStep,
  createInitialBuilderState,
} from '../../model/overlayTypes';
import {
  convertAIResultToBuilderState,
  extractPuzzleMultiModel,
} from '../../services/aiExtraction';
import { ExtractionProgress as EnsembleProgress } from '../../services/ensembleExtraction';
import { builderReducer, countValidCells } from '../../state/builderReducer';
import { deleteDraft, loadDraft, saveDraft } from '../../storage/drafts';
import { getSettings, savePuzzle } from '../../storage/puzzles';
import { buildPuzzleSpec, getBuilderStats, validateBuilderState } from '../../utils/specBuilder';
import { validatePuzzleSpec } from '../../validator/validateSpec';
import { colors, spacing, radii } from '../../theme';
import { fontFamilies } from '../../theme/fonts';
import { Button, Heading, Body, Label, Mono } from '../components/ui';
import AIVerificationModal from '../components/AIVerificationModal';

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
  const [pendingAIResult, setPendingAIResult] = useState<AIExtractionResult | null>(null);

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
    if (!state.image) return;

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
        mediaTypes: ['images'] as any,
        allowsEditing: false,
        quality: 0.7,
        base64: true,
      });

      if (!result.canceled && result.assets[0]) {
        const asset = result.assets[0];
        let manipulated: { uri: string; width: number; height: number; base64?: string } | null =
          null;

        let base64FromFile: string | undefined = asset.base64 || undefined;
        if (!base64FromFile) {
          try {
            base64FromFile = await FileSystem.readAsStringAsync(asset.uri, {
              encoding: 'base64' as any,
            });
          } catch (e) {
            console.warn('Failed to read image base64 from file system:', e);
          }
        }

        const isLikelyHEIC =
          asset.uri.toLowerCase().includes('.heic') || asset.uri.toLowerCase().includes('.heif');

        if (isLikelyHEIC) {
          console.log('[DEBUG] Detected HEIC format - must convert to JPEG for AI processing');
        }

        const tryManipulate = async (targetWidth: number) => {
          return await ImageManipulator.manipulateAsync(
            asset.uri,
            [{ resize: { width: targetWidth } }],
            {
              compress: 0.85,
              format: ImageManipulator.SaveFormat.JPEG,
              base64: true,
            }
          );
        };

        try {
          manipulated = await tryManipulate(1600);
        } catch (e1) {
          try {
            manipulated = await tryManipulate(1024);
          } catch (e2) {
            console.warn('Image manipulation failed, falling back to picker base64:', e2);
            manipulated = null;

            if (!base64FromFile || isLikelyHEIC) {
              const reason = isLikelyHEIC
                ? 'This image is in HEIC format which must be converted to JPEG for AI processing, but the conversion failed.'
                : 'Failed to process the selected image.';

              Alert.alert(
                'Image Processing Error',
                `${reason}\n\nThis can happen with certain image formats or if the image is too large. Please try:\n\n1. Taking a screenshot of the puzzle (screenshots are usually PNG)\n2. Converting the image to JPEG first\n3. Using a different image`,
                [{ text: 'OK' }]
              );
              if (!hasImageRef.current) {
                navigation.goBack();
              }
              return;
            }
          }
        }

        const finalUri = manipulated?.uri || asset.uri;
        const finalWidth = manipulated?.width || asset.width;
        const finalHeight = manipulated?.height || asset.height;
        const finalBase64 = manipulated?.base64 || base64FromFile || undefined;

        if (!finalBase64 || finalBase64.length < 100) {
          Alert.alert(
            'Image Data Error',
            'The selected image could not be processed. Please try selecting a different image or taking a screenshot.',
            [{ text: 'OK' }]
          );
          if (!hasImageRef.current) {
            navigation.goBack();
          }
          return;
        }

        console.log(
          `[DEBUG] Image prepared - size: ${finalWidth}x${finalHeight}, base64 length: ${finalBase64.length}`
        );

        dispatch({
          type: 'SET_IMAGE',
          image: {
            uri: finalUri,
            width: finalWidth,
            height: finalHeight,
            base64: finalBase64,
          },
        });
        hasImageRef.current = true;
      } else if (!hasImageRef.current) {
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

    const settings = await getSettings();

    const hasOpenRouterKey = !!settings.openrouterApiKey?.trim();
    const hasGoogleKey = !!settings.googleApiKey?.trim();
    const hasAnthropicKey = !!settings.anthropicApiKey?.trim();
    const hasOpenAIKey = !!settings.openaiApiKey?.trim();

    // OpenRouter provides access to all models, so only need one key
    if (!hasOpenRouterKey && !hasGoogleKey && !hasAnthropicKey && !hasOpenAIKey) {
      Alert.alert(
        'API Key Required',
        'Please add an API key in Settings to use AI extraction.\n\n' +
          'OpenRouter (Recommended) - Access to all models\n' +
          'Or individual provider keys:\n' +
          '• Google (Gemini)\n' +
          '• Anthropic (Claude)\n' +
          '• OpenAI (GPT)',
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

    const strategy = settings.extractionStrategy || 'accurate';
    const strategyNames: Record<string, string> = {
      fast: 'Fast',
      balanced: 'Balanced',
      accurate: 'Accurate',
      ensemble: 'Maximum Accuracy',
    };
    setAIProgress(`Starting ${strategyNames[strategy]} extraction...`);

    const hasCVService = !!settings.cvServiceUrl?.trim();
    const result = await extractPuzzleMultiModel(state.image.base64, {
      strategy,
      apiKeys: {
        openrouter: settings.openrouterApiKey,
        google: settings.googleApiKey,
        anthropic: settings.anthropicApiKey,
        openai: settings.openaiApiKey,
      },
      useHybridCV: hasCVService,
      cvServiceUrl: settings.cvServiceUrl || undefined,
      onProgress: (progress: EnsembleProgress) => {
        const modelInfo = progress.modelsUsed?.length ? ` (${progress.modelsUsed.join(', ')})` : '';
        setAIProgress(`${progress.message}${modelInfo}`);
      },
    });

    if (result.success && result.result) {
      const converted = convertAIResultToBuilderState(result.result);

      console.log('[DEBUG] AI extraction result:', {
        boardRows: result.result.board.rows,
        boardCols: result.result.board.cols,
        boardShape: result.result.board.shape,
        boardRegions: result.result.board.regions,
        gridLocation: result.result.board.gridLocation,
        constraintCount: Object.keys(result.result.board.constraints || {}).length,
        dominoCount: result.result.dominoes.dominoes.length,
        modelsUsed: result.modelsUsed,
        timing: result.timing,
      });

      console.log('[DEBUG] Converted state:', {
        gridRows: converted.grid.rows,
        gridCols: converted.grid.cols,
        gridHoles: converted.grid.holes?.length,
        regionGridSize: converted.regions.regionGrid?.length,
        constraintCount: Object.keys(converted.constraints.regionConstraints || {}).length,
        dominoCount: converted.dominoes.length,
      });

      if (result.timing) {
        console.log(
          `[DEBUG] Extraction timing: board=${result.timing.boardMs}ms, dominoes=${result.timing.dominoesMs}ms, total=${result.timing.totalMs}ms`
        );
      }

      setAIProgress(null);
      setPendingAIResult(result.result);
    } else {
      dispatch({ type: 'AI_ERROR', error: result.error || 'Unknown error' });
      setAIProgress(null);

      let errorMsg = result.error || 'Failed to extract puzzle data';
      let helpText = '';

      if (errorMsg.includes('API key')) {
        helpText = '\n\nPlease check your API key in Settings.';
      } else if (errorMsg.includes('Could not process image') || errorMsg.includes('400')) {
        errorMsg = 'Board extraction failed: Could not process image';
        helpText =
          '\n\nThe image format may not be compatible with the AI service. This can happen when:\n\n• The image is in an unsupported format (e.g., HEIC)\n• The image failed to convert to JPEG\n• The image is corrupted\n\nPlease try:\n1. Taking a screenshot of the puzzle instead\n2. Converting the image to JPEG/PNG first\n3. Using a different image';
      } else if (errorMsg.includes('model')) {
        helpText = '\n\nThere may be an issue with the AI model. Please try again later.';
      } else if (errorMsg.includes('JSON') || errorMsg.includes('parsing')) {
        helpText = '\n\nThe AI response was malformed. Try again, or extract manually.';
      } else {
        helpText = '\n\nYou can still build the puzzle manually through the steps.';
      }

      Alert.alert('Extraction Failed', errorMsg + helpText);
    }
  }, [state.image, navigation]);

  const handleAcceptAIResult = useCallback(() => {
    if (!pendingAIResult) return;

    const converted = convertAIResultToBuilderState(pendingAIResult);
    const boardConf = pendingAIResult.board.confidence;
    const dominoConf = pendingAIResult.dominoes.confidence;

    dispatch({
      type: 'AI_SUCCESS',
      grid: converted.grid,
      regions: converted.regions,
      constraints: converted.constraints,
      dominoes: converted.dominoes,
      reasoning: pendingAIResult.reasoning,
      confidence: {
        grid: boardConf?.grid,
        regions: boardConf?.regions,
        constraints: boardConf?.constraints,
        dominoes: dominoConf,
      },
    });

    console.log('[DEBUG] Accepted and applied AI result');
    setPendingAIResult(null);
    dispatch({ type: 'SET_STEP', step: 2 });
  }, [pendingAIResult]);

  const handleRejectAIResult = useCallback(() => {
    console.log('[DEBUG] User rejected AI result');
    setPendingAIResult(null);
    Alert.alert(
      'Extraction Rejected',
      'You can manually build the puzzle using the step-by-step wizard.'
    );
  }, []);

  const goToStep = (step: BuilderStep) => {
    dispatch({ type: 'SET_STEP', step });
  };

  const handleNext = () => {
    if (state.step < 4) {
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
      Alert.alert('Save Progress?', 'Do you want to save your progress?', [
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
      ]);
    }
  };

  const handleFinish = useCallback(() => {
    const validation = validateBuilderState(state);

    if (!validation.valid) {
      Alert.alert('Invalid Puzzle', validation.errors.join('\n'), [{ text: 'OK' }]);
      return;
    }

    if (validation.warnings.length > 0) {
      Alert.alert('Warnings', `${validation.warnings.join('\n')}\n\nContinue anyway?`, [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Continue',
          onPress: () => setShowNameModal(true),
        },
      ]);
      return;
    }

    setShowNameModal(true);
  }, [state]);

  const handleSavePuzzle = useCallback(async () => {
    const name = puzzleName.trim() || `Puzzle ${new Date().toLocaleDateString()}`;
    const result = buildPuzzleSpec(state, name);

    if (!result.success || !result.spec) {
      Alert.alert('Error', result.errors.join('\n'));
      return;
    }

    const specValidation = validatePuzzleSpec(result.spec);
    if (!specValidation.valid) {
      Alert.alert('Invalid Puzzle Spec', specValidation.errors.join('\n'), [{ text: 'OK' }]);
      return;
    }

    try {
      const savedPuzzle = await savePuzzle(result.spec, result.yaml || '');
      await deleteDraft(state.draftId);
      setShowNameModal(false);
      setPuzzleName('');
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
      <Animated.View entering={FadeInDown.duration(400)} style={styles.header}>
        <TouchableOpacity onPress={handleBack} style={styles.backButton}>
          <Mono style={styles.backButtonText}>{state.step === 1 ? '×' : '←'}</Mono>
        </TouchableOpacity>
        <Heading size="small" style={styles.title}>{stepTitles[state.step - 1]}</Heading>
        <Label size="small" color="secondary">{state.step}/4</Label>
      </Animated.View>

      {/* Step Progress */}
      <Animated.View entering={FadeIn.duration(400)} style={styles.progressContainer}>
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
      </Animated.View>

      {/* Step Content */}
      <View style={styles.content}>{renderStepContent()}</View>

      {/* Footer */}
      <Animated.View entering={FadeInUp.duration(400)} style={styles.footer}>
        {state.step > 1 && (
          <Button
            variant="secondary"
            size="medium"
            title="Back"
            onPress={() => goToStep((state.step - 1) as BuilderStep)}
            style={styles.footerButton}
          />
        )}

        {state.step < 4 ? (
          <Button
            variant="primary"
            size="medium"
            title="Next"
            onPress={handleNext}
            style={state.step === 1 ? styles.footerButtonFull : styles.footerButton}
          />
        ) : (
          <Button
            variant="success"
            size="medium"
            title="Create Puzzle"
            onPress={handleFinish}
            style={styles.footerButton}
          />
        )}
      </Animated.View>

      {/* Name Prompt Modal */}
      <Modal
        visible={showNameModal}
        animationType="fade"
        transparent
        onRequestClose={() => setShowNameModal(false)}
      >
        <View style={styles.modalOverlay}>
          <Animated.View entering={FadeIn.duration(300)} style={styles.modalContent}>
            <Heading size="medium" style={styles.modalTitle}>Name Your Puzzle</Heading>
            <TextInput
              style={styles.modalInput}
              value={puzzleName}
              onChangeText={setPuzzleName}
              placeholder="Enter puzzle name..."
              placeholderTextColor={colors.text.tertiary}
              autoFocus
              onSubmitEditing={handleSavePuzzle}
            />
            <View style={styles.modalStats}>
              {(() => {
                const stats = getBuilderStats(state);
                return (
                  <Body size="small" color="secondary" style={styles.modalStatsText}>
                    {stats.cellCount} cells • {stats.regionCount} regions • {stats.dominoCount} dominoes
                  </Body>
                );
              })()}
            </View>
            <View style={styles.modalButtons}>
              <Button
                variant="secondary"
                size="medium"
                title="Cancel"
                onPress={() => {
                  setShowNameModal(false);
                  setPuzzleName('');
                }}
                style={styles.modalButton}
              />
              <Button
                variant="success"
                size="medium"
                title="Save"
                onPress={handleSavePuzzle}
                style={styles.modalButton}
              />
            </View>
          </Animated.View>
        </View>
      </Modal>

      {/* AI Verification Modal */}
      {pendingAIResult && (
        <AIVerificationModal
          visible={true}
          boardResult={pendingAIResult.board}
          dominoResult={pendingAIResult.dominoes}
          onAccept={handleAcceptAIResult}
          onReject={handleRejectAIResult}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.surface.obsidian,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[3],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.slate,
  },
  backButton: {
    padding: spacing[2],
  },
  backButtonText: {
    fontSize: 20,
    color: colors.text.primary,
  },
  title: {
    flex: 1,
    textAlign: 'center',
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: spacing[3],
    gap: spacing[2],
  },
  progressDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.surface.graphite,
  },
  progressDotActive: {
    backgroundColor: colors.accent.brass,
    width: 24,
  },
  progressDotComplete: {
    backgroundColor: colors.semantic.jade,
  },
  content: {
    flex: 1,
  },
  footer: {
    flexDirection: 'row',
    padding: spacing[4],
    gap: spacing[3],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
  },
  footerButton: {
    flex: 1,
  },
  footerButtonFull: {
    flex: 1,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing[5],
  },
  modalContent: {
    backgroundColor: colors.surface.charcoal,
    borderRadius: radii.xl,
    padding: spacing[6],
    width: '100%',
    maxWidth: 360,
    borderWidth: 1,
    borderColor: colors.surface.slate,
  },
  modalTitle: {
    textAlign: 'center',
    marginBottom: spacing[5],
  },
  modalInput: {
    backgroundColor: colors.surface.obsidian,
    color: colors.text.primary,
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[4],
    borderRadius: radii.md,
    fontSize: 16,
    fontFamily: fontFamilies.bodyRegular,
    marginBottom: spacing[3],
    borderWidth: 1,
    borderColor: colors.surface.slate,
  },
  modalStats: {
    marginBottom: spacing[5],
  },
  modalStatsText: {
    textAlign: 'center',
  },
  modalButtons: {
    flexDirection: 'row',
    gap: spacing[3],
  },
  modalButton: {
    flex: 1,
  },
});
