/**
 * Home / Library Screen
 * Lists saved puzzles and allows importing new ones
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  Modal,
  ScrollView,
  Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { StoredPuzzle } from '../../model/types';
import {
  getAllPuzzles,
  deletePuzzle,
  importPuzzle,
} from '../../storage/puzzles';
import { parsePuzzle } from '../../model/parser';
import { validatePuzzleSpec } from '../../validator/validateSpec';
import { SAMPLE_PUZZLES } from '../../samples';

export default function HomeScreen({ navigation }: any) {
  const [puzzles, setPuzzles] = useState<StoredPuzzle[]>([]);
  const [showImportModal, setShowImportModal] = useState(false);
  const [showSamplesModal, setShowSamplesModal] = useState(false);
  const [yamlInput, setYamlInput] = useState('');

  useFocusEffect(
    useCallback(() => {
      loadPuzzles();
    }, [])
  );

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

  const renderPuzzle = ({ item }: { item: StoredPuzzle }) => (
    <TouchableOpacity
      style={styles.puzzleCard}
      onPress={() => navigation.navigate('Viewer', { puzzleId: item.id })}
    >
      <View style={styles.puzzleInfo}>
        <Text style={styles.puzzleName}>{item.name}</Text>
        <Text style={styles.puzzleMeta}>
          {item.spec.rows}x{item.spec.cols} • {item.solved ? 'Solved ✓' : 'Unsolved'}
        </Text>
        <Text style={styles.puzzleDate}>
          {new Date(item.updatedAt).toLocaleDateString()}
        </Text>
      </View>
      <TouchableOpacity
        style={styles.deleteButton}
        onPress={() => handleDelete(item)}
      >
        <Text style={styles.deleteButtonText}>Delete</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>Pips Puzzles</Text>
        <TouchableOpacity
          style={styles.settingsButton}
          onPress={() => navigation.navigate('Settings')}
        >
          <Text style={styles.settingsButtonText}>⚙</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => setShowImportModal(true)}
        >
          <Text style={styles.actionButtonText}>Import YAML</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.actionButton}
          onPress={() => setShowSamplesModal(true)}
        >
          <Text style={styles.actionButtonText}>Load Sample</Text>
        </TouchableOpacity>
      </View>

      {puzzles.length === 0 ? (
        <View style={styles.emptyState}>
          <Text style={styles.emptyText}>No puzzles yet</Text>
          <Text style={styles.emptySubtext}>
            Import a puzzle or load a sample to get started
          </Text>
        </View>
      ) : (
        <FlatList
          data={puzzles}
          renderItem={renderPuzzle}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
        />
      )}

      {/* Import Modal */}
      <Modal
        visible={showImportModal}
        animationType="slide"
        onRequestClose={() => setShowImportModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Import YAML</Text>
            <TouchableOpacity onPress={() => setShowImportModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <TextInput
            style={styles.yamlInput}
            multiline
            placeholder="Paste YAML puzzle here..."
            value={yamlInput}
            onChangeText={setYamlInput}
            autoCapitalize="none"
            autoCorrect={false}
          />

          <TouchableOpacity style={styles.importButton} onPress={handleImport}>
            <Text style={styles.importButtonText}>Import</Text>
          </TouchableOpacity>
        </View>
      </Modal>

      {/* Samples Modal */}
      <Modal
        visible={showSamplesModal}
        animationType="slide"
        onRequestClose={() => setShowSamplesModal(false)}
      >
        <View style={styles.modalContainer}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>Sample Puzzles</Text>
            <TouchableOpacity onPress={() => setShowSamplesModal(false)}>
              <Text style={styles.modalClose}>✕</Text>
            </TouchableOpacity>
          </View>

          <ScrollView style={styles.samplesList}>
            {SAMPLE_PUZZLES.map((sample) => (
              <TouchableOpacity
                key={sample.id}
                style={styles.sampleCard}
                onPress={() => handleLoadSample(sample)}
              >
                <Text style={styles.sampleName}>{sample.name}</Text>
                <Text style={styles.sampleId}>{sample.id}</Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
  },
  settingsButton: {
    padding: 8,
  },
  settingsButtonText: {
    fontSize: 24,
  },
  actions: {
    flexDirection: 'row',
    padding: 16,
    gap: 12,
  },
  actionButton: {
    flex: 1,
    backgroundColor: '#007AFF',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  list: {
    padding: 16,
  },
  puzzleCard: {
    backgroundColor: '#fff',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  puzzleInfo: {
    flex: 1,
  },
  puzzleName: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  puzzleMeta: {
    fontSize: 14,
    color: '#666',
    marginBottom: 2,
  },
  puzzleDate: {
    fontSize: 12,
    color: '#999',
  },
  deleteButton: {
    padding: 8,
    paddingHorizontal: 12,
  },
  deleteButtonText: {
    color: '#FF3B30',
    fontSize: 14,
    fontWeight: '600',
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyText: {
    fontSize: 20,
    fontWeight: '600',
    color: '#999',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#fff',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingTop: 60,
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
  },
  modalClose: {
    fontSize: 28,
    color: '#666',
  },
  yamlInput: {
    flex: 1,
    padding: 16,
    fontSize: 14,
    fontFamily: 'monospace',
    textAlignVertical: 'top',
  },
  importButton: {
    backgroundColor: '#007AFF',
    padding: 16,
    margin: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  importButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  samplesList: {
    flex: 1,
    padding: 16,
  },
  sampleCard: {
    backgroundColor: '#f9f9f9',
    padding: 16,
    borderRadius: 8,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#ddd',
  },
  sampleName: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  sampleId: {
    fontSize: 12,
    color: '#666',
  },
});
