/**
 * Home / Library Screen
 * Lists saved puzzles and allows importing new ones
 *
 * Redesigned with "Tactile Game Table" aesthetic
 */

import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  Modal,
  ScrollView,
  Alert,
  Pressable,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withDelay,
  withTiming,
  FadeIn,
  FadeInDown,
  FadeInUp,
  Layout,
} from 'react-native-reanimated';
import { useFocusEffect } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { StoredPuzzle } from '../../model/types';
import {
  getAllPuzzles,
  deletePuzzle,
  importPuzzle,
} from '../../storage/puzzles';
import { parsePuzzle } from '../../model/parser';
import { validatePuzzleSpec } from '../../validator/validateSpec';
import { SAMPLE_PUZZLES } from '../../samples';
import { listDrafts, deleteDraft, cleanExpiredDrafts } from '../../storage/drafts';
import { DraftMeta } from '../../model/overlayTypes';

// Theme imports
import { colors, spacing, radii, shadows } from '../../theme';
import { fontFamilies, textStyles } from '../../theme/fonts';
import { springConfigs, timingConfigs } from '../../theme/animations';
import {
  Button,
  Card,
  CardFooter,
  Badge,
  Text,
  DisplayText,
  Heading,
  Body,
  Label,
  Mono,
  Divider,
} from '../components/ui';

// ════════════════════════════════════════════════════════════════════════════
// Animated Components
// ════════════════════════════════════════════════════════════════════════════

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

// ════════════════════════════════════════════════════════════════════════════
// Main Component
// ════════════════════════════════════════════════════════════════════════════

export default function HomeScreen({ navigation }: any) {
  const [puzzles, setPuzzles] = useState<StoredPuzzle[]>([]);
  const [showImportModal, setShowImportModal] = useState(false);
  const [showSamplesModal, setShowSamplesModal] = useState(false);
  const [yamlInput, setYamlInput] = useState('');
  const [drafts, setDrafts] = useState<DraftMeta[]>([]);

  useFocusEffect(
    useCallback(() => {
      loadPuzzles();
      loadDrafts();
    }, [])
  );

  useEffect(() => {
    cleanExpiredDrafts();
  }, []);

  const loadDrafts = async () => {
    const loaded = await listDrafts();
    setDrafts(loaded);
  };

  const handleResumeDraft = (draft: DraftMeta) => {
    navigation.navigate('OverlayBuilder', { draftId: draft.draftId });
  };

  const handleDeleteDraft = async (draft: DraftMeta) => {
    Alert.alert('Delete Draft', 'Are you sure you want to delete this draft?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteDraft(draft.draftId);
          loadDrafts();
        },
      },
    ]);
  };

  const handleCreateFromPhoto = () => {
    navigation.navigate('OverlayBuilder');
  };

  const loadPuzzles = async () => {
    const loaded = await getAllPuzzles();
    setPuzzles(loaded.sort((a, b) => b.updatedAt - a.updatedAt));
  };

  const handleImport = async () => {
    try {
      const parseResult = parsePuzzle(yamlInput);

      if (!parseResult.success || !parseResult.spec) {
        Alert.alert('Parse Error', parseResult.error || 'Failed to parse YAML');
        return;
      }

      const validation = validatePuzzleSpec(parseResult.spec);

      if (!validation.valid) {
        Alert.alert(
          'Validation Error',
          `Puzzle has errors:\n${validation.errors.join('\n')}`
        );
        return;
      }

      if (validation.warnings.length > 0) {
        Alert.alert(
          'Warnings',
          `Puzzle has warnings:\n${validation.warnings.join('\n')}\n\nContinue?`,
          [
            { text: 'Cancel', style: 'cancel' },
            {
              text: 'Import',
              onPress: async () => {
                await importPuzzle(yamlInput, parseResult.spec!);
                setShowImportModal(false);
                setYamlInput('');
                loadPuzzles();
              },
            },
          ]
        );
        return;
      }

      await importPuzzle(yamlInput, parseResult.spec);
      setShowImportModal(false);
      setYamlInput('');
      loadPuzzles();
      Alert.alert('Success', 'Puzzle imported successfully');
    } catch (error) {
      Alert.alert('Error', `Failed to import: ${error}`);
    }
  };

  const handleLoadSample = async (sample: any) => {
    try {
      const parseResult = parsePuzzle(sample.yaml);
      if (parseResult.success && parseResult.spec) {
        await importPuzzle(sample.yaml, parseResult.spec);
        setShowSamplesModal(false);
        loadPuzzles();
        Alert.alert('Success', `Loaded sample: ${sample.name}`);
      }
    } catch (error) {
      Alert.alert('Error', `Failed to load sample: ${error}`);
    }
  };

  const handleDelete = (puzzle: StoredPuzzle) => {
    Alert.alert(
      'Delete Puzzle',
      `Are you sure you want to delete "${puzzle.name}"?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deletePuzzle(puzzle.id);
            loadPuzzles();
          },
        },
      ]
    );
  };

  // ══════════════════════════════════════════════════════════════════════════
  // Render Functions
  // ══════════════════════════════════════════════════════════════════════════

  const renderPuzzleCard = ({ item, index }: { item: StoredPuzzle; index: number }) => (
    <Animated.View
      entering={FadeInDown.delay(index * 50).springify()}
      layout={Layout.springify()}
    >
      <PuzzleCard
        puzzle={item}
        onPress={() => navigation.navigate('Viewer', { puzzleId: item.id })}
        onDelete={() => handleDelete(item)}
      />
    </Animated.View>
  );

  return (
    <SafeAreaView style={styles.container} edges={['top']}>
      {/* Header */}
      <Animated.View entering={FadeInDown.delay(0)} style={styles.header}>
        <View>
          <Label size="small" style={styles.headerLabel}>
            PUZZLE SOLVER
          </Label>
          <DisplayText size="medium">Pips</DisplayText>
        </View>
        <TouchableOpacity
          style={styles.settingsButton}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.settingsIcon}>⚙</Text>
        </TouchableOpacity>
      </Animated.View>

      {/* Draft Recovery Banner */}
      {drafts.length > 0 && (
        <Animated.View entering={FadeIn.delay(200)}>
          <Card variant="accent" style={styles.draftBanner}>
            <View style={styles.draftContent}>
              <Badge label="DRAFT" variant="warning" size="small" />
              <Heading size="small" style={styles.draftTitle}>
                Continue where you left off?
              </Heading>
              <Body size="small" color="secondary">
                Step {drafts[0].step}/4 • {drafts[0].rows}×{drafts[0].cols} grid
              </Body>
            </View>
            <View style={styles.draftActions}>
              <Button
                title="Resume"
                variant="primary"
                size="small"
                onPress={() => handleResumeDraft(drafts[0])}
              />
              <TouchableOpacity
                style={styles.draftDeleteButton}
                onPress={() => handleDeleteDraft(drafts[0])}
              >
                <Text style={styles.draftDeleteIcon}>✕</Text>
              </TouchableOpacity>
            </View>
          </Card>
        </Animated.View>
      )}

      {/* Action Buttons */}
      <Animated.View entering={FadeInUp.delay(100)} style={styles.actions}>
        <Button
          title="From Photo"
          variant="primary"
          size="large"
          fullWidth
          onPress={handleCreateFromPhoto}
          leftIcon={<Text style={styles.buttonIcon}>📷</Text>}
        />

        <View style={styles.actionRow}>
          <View style={styles.actionHalf}>
            <Button
              title="Import YAML"
              variant="secondary"
              size="medium"
              fullWidth
              onPress={() => setShowImportModal(true)}
            />
          </View>
          <View style={styles.actionHalf}>
            <Button
              title="Load Sample"
              variant="secondary"
              size="medium"
              fullWidth
              onPress={() => setShowSamplesModal(true)}
            />
          </View>
        </View>
      </Animated.View>

      <Divider spacing={4} />

      {/* Puzzle List */}
      {puzzles.length === 0 ? (
        <Animated.View entering={FadeIn.delay(300)} style={styles.emptyState}>
          <Text style={styles.emptyIcon}>🎲</Text>
          <Heading size="medium" color="secondary">
            No puzzles yet
          </Heading>
          <Body size="small" color="tertiary" align="center">
            Import a puzzle or load a sample to get started
          </Body>
        </Animated.View>
      ) : (
        <FlatList
          data={puzzles}
          renderItem={renderPuzzleCard}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* Import Modal */}
      <Modal
        visible={showImportModal}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowImportModal(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Heading size="large">Import YAML</Heading>
            <TouchableOpacity onPress={() => setShowImportModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <TextInput
            style={styles.yamlInput}
            multiline
            placeholder="Paste YAML puzzle here..."
            placeholderTextColor={colors.text.tertiary}
            value={yamlInput}
            onChangeText={setYamlInput}
            autoCapitalize="none"
            autoCorrect={false}
          />

          <View style={styles.modalFooter}>
            <Button title="Import Puzzle" variant="primary" size="large" fullWidth onPress={handleImport} />
          </View>
        </SafeAreaView>
      </Modal>

      {/* Samples Modal */}
      <Modal
        visible={showSamplesModal}
        animationType="slide"
        presentationStyle="pageSheet"
        onRequestClose={() => setShowSamplesModal(false)}
      >
        <SafeAreaView style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Heading size="large">Sample Puzzles</Heading>
            <TouchableOpacity onPress={() => setShowSamplesModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.samplesList}>
            {SAMPLE_PUZZLES.map((sample, index) => (
              <Animated.View
                key={sample.id}
                entering={FadeInDown.delay(index * 30)}
              >
                <Card
                  variant="default"
                  onPress={() => handleLoadSample(sample)}
                  style={styles.sampleCard}
                >
                  <Heading size="small">{sample.name}</Heading>
                  <Mono small>{sample.id}</Mono>
                </Card>
              </Animated.View>
            ))}
          </ScrollView>
        </SafeAreaView>
      </Modal>
    </SafeAreaView>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Puzzle Card Component
// ════════════════════════════════════════════════════════════════════════════

interface PuzzleCardProps {
  puzzle: StoredPuzzle;
  onPress: () => void;
  onDelete: () => void;
}

function PuzzleCard({ puzzle, onPress, onDelete }: PuzzleCardProps) {
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePressIn = () => {
    scale.value = withTiming(0.98, timingConfigs.fast);
  };

  const handlePressOut = () => {
    scale.value = withSpring(1, springConfigs.snappy);
  };

  return (
    <AnimatedPressable
      onPress={onPress}
      onPressIn={handlePressIn}
      onPressOut={handlePressOut}
      style={[styles.puzzleCard, animatedStyle]}
    >
      <View style={styles.puzzleCardInner}>
        {/* Mini grid preview */}
        <View style={styles.puzzlePreview}>
          <View style={styles.puzzlePreviewGrid}>
            {Array.from({ length: 4 }).map((_, i) => (
              <View
                key={i}
                style={[
                  styles.puzzlePreviewCell,
                  { backgroundColor: colors.regions[i % colors.regions.length] },
                ]}
              />
            ))}
          </View>
        </View>

        {/* Info */}
        <View style={styles.puzzleInfo}>
          <Heading size="small" numberOfLines={1}>
            {puzzle.name}
          </Heading>
          <View style={styles.puzzleMeta}>
            <Mono small>
              {puzzle.spec.rows}×{puzzle.spec.cols}
            </Mono>
            <Badge
              label={puzzle.solved ? 'Solved' : 'Unsolved'}
              variant={puzzle.solved ? 'success' : 'default'}
              size="small"
            />
          </View>
          <Label size="small" color="tertiary">
            {new Date(puzzle.updatedAt).toLocaleDateString()}
          </Label>
        </View>

        {/* Delete button */}
        <TouchableOpacity style={styles.deleteButton} onPress={onDelete}>
          <Text style={styles.deleteIcon}>🗑</Text>
        </TouchableOpacity>
      </View>
    </AnimatedPressable>
  );
}

// ════════════════════════════════════════════════════════════════════════════
// Styles
// ════════════════════════════════════════════════════════════════════════════

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.surface.obsidian,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    paddingHorizontal: spacing[4],
    paddingTop: spacing[4],
    paddingBottom: spacing[3],
  },
  headerLabel: {
    color: colors.accent.brass,
    marginBottom: spacing[1],
  },
  settingsButton: {
    padding: spacing[2],
  },
  settingsIcon: {
    fontSize: 24,
    color: colors.text.secondary,
  },
  draftBanner: {
    marginHorizontal: spacing[4],
    marginBottom: spacing[3],
  },
  draftContent: {
    gap: spacing[1],
  },
  draftTitle: {
    marginTop: spacing[2],
  },
  draftActions: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing[3],
    gap: spacing[2],
  },
  draftDeleteButton: {
    padding: spacing[2],
  },
  draftDeleteIcon: {
    color: colors.text.tertiary,
    fontSize: 18,
  },
  actions: {
    paddingHorizontal: spacing[4],
    gap: spacing[3],
  },
  actionRow: {
    flexDirection: 'row',
    gap: spacing[3],
  },
  actionHalf: {
    flex: 1,
  },
  buttonIcon: {
    fontSize: 18,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing[10],
    gap: spacing[2],
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: spacing[2],
  },
  list: {
    padding: spacing[4],
    gap: spacing[3],
  },
  puzzleCard: {
    backgroundColor: colors.surface.slate,
    borderRadius: radii.lg,
    ...shadows.md,
    marginBottom: spacing[3],
  },
  puzzleCardInner: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing[4],
    gap: spacing[4],
  },
  puzzlePreview: {
    width: 56,
    height: 56,
    backgroundColor: colors.surface.charcoal,
    borderRadius: radii.md,
    padding: spacing[1],
    justifyContent: 'center',
    alignItems: 'center',
  },
  puzzlePreviewGrid: {
    width: 40,
    height: 40,
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  puzzlePreviewCell: {
    width: 18,
    height: 18,
    margin: 1,
    borderRadius: 2,
  },
  puzzleInfo: {
    flex: 1,
    gap: spacing[1],
  },
  puzzleMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing[2],
  },
  deleteButton: {
    padding: spacing[2],
  },
  deleteIcon: {
    fontSize: 18,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: colors.surface.charcoal,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing[4],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.ash,
  },
  modalClose: {
    fontSize: 28,
    color: colors.text.secondary,
  },
  yamlInput: {
    flex: 1,
    padding: spacing[4],
    fontSize: 14,
    fontFamily: fontFamilies.monoRegular,
    color: colors.text.primary,
    backgroundColor: colors.surface.slate,
    margin: spacing[4],
    borderRadius: radii.lg,
    textAlignVertical: 'top',
  },
  modalFooter: {
    padding: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.ash,
  },
  samplesList: {
    flex: 1,
    padding: spacing[4],
  },
  sampleCard: {
    marginBottom: spacing[3],
  },
});
