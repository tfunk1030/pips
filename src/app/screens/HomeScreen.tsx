import React, { useEffect, useState } from 'react';
import { View, Text, FlatList, TouchableOpacity, StyleSheet, ActivityIndicator } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import { loadPuzzles, savePuzzle } from '../../storage/puzzles';
import { parsePuzzle } from '../../model/parser';
import { useExtraction } from '../hooks';
import sampleText from '../../samples/sample1.yaml?raw';

// @ts-ignore bundler raw import fallback
const sampleYaml = typeof sampleText === 'string' ? sampleText : '';

type Props = NativeStackScreenProps<RootStackParamList, 'Home'>;

interface PuzzleEntry {
  id: string;
  name: string;
  text: string;
}

export default function HomeScreen({ navigation }: Props) {
  const [library, setLibrary] = useState<PuzzleEntry[]>([]);
  const { isExtracting, extractFromImage } = useExtraction();

  useEffect(() => {
    loadPuzzles().then(setLibrary);
  }, []);

  const handleLoadSample = async () => {
    const { puzzle } = parsePuzzle(sampleYaml);
    const entry: PuzzleEntry = { id: `sample-${Date.now()}`, name: puzzle.name ?? 'Sample', text: sampleYaml };
    await savePuzzle(entry);
    const updated = await loadPuzzles();
    setLibrary(updated);
    navigation.navigate('Viewer', { puzzle, sourceText: sampleYaml });
  };

  const handleScanPuzzle = async () => {
    // In production, this would open the camera or image picker
    // For now, we use a placeholder URI to trigger the extraction
    const mockImageUri = 'mock://captured-image';
    const result = await extractFromImage(mockImageUri);
    navigation.navigate('ExtractionResult', {
      extractionResult: result,
      sourceImageUri: mockImageUri,
    });
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Pips Library</Text>
      <Text style={styles.sub}>Saved puzzles stay on device for offline solving.</Text>
      <TouchableOpacity style={styles.primary} onPress={() => navigation.navigate('Editor')}>
        <Text style={styles.primaryText}>Import YAML</Text>
      </TouchableOpacity>
      <TouchableOpacity
        style={[styles.primary, styles.scanButton]}
        onPress={handleScanPuzzle}
        disabled={isExtracting}
        accessibilityRole="button"
        accessibilityLabel="Scan puzzle from image"
        accessibilityState={{ disabled: isExtracting }}
      >
        {isExtracting ? (
          <View style={styles.scanButtonContent}>
            <ActivityIndicator size="small" color="#fff" />
            <Text style={styles.primaryText}>Extracting...</Text>
          </View>
        ) : (
          <Text style={styles.primaryText}>Scan Puzzle</Text>
        )}
      </TouchableOpacity>
      <TouchableOpacity style={styles.secondary} onPress={handleLoadSample}>
        <Text style={styles.secondaryText}>Load Example Puzzle</Text>
      </TouchableOpacity>
      <FlatList
        data={library}
        keyExtractor={(item) => item.id}
        contentContainerStyle={{ paddingVertical: 8 }}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.card}
            onPress={() => {
              const { puzzle } = parsePuzzle(item.text);
              navigation.navigate('Viewer', { puzzle, sourceText: item.text });
            }}
          >
            <Text style={styles.cardTitle}>{item.name}</Text>
            <Text style={styles.cardMeta}>{item.text.split('\n')[0]}</Text>
          </TouchableOpacity>
        )}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0c1021', padding: 16 },
  header: { fontSize: 24, fontWeight: 'bold', color: '#fff', marginBottom: 4 },
  sub: { color: '#9aa5ce', marginBottom: 12 },
  primary: { backgroundColor: '#4a60c4', padding: 12, borderRadius: 8, marginBottom: 8 },
  primaryText: { color: '#fff', textAlign: 'center', fontWeight: 'bold' },
  scanButton: { backgroundColor: '#2d8659' },
  scanButtonContent: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  secondary: { borderWidth: 1, borderColor: '#4a60c4', padding: 12, borderRadius: 8, marginBottom: 16 },
  secondaryText: { color: '#4a60c4', textAlign: 'center', fontWeight: 'bold' },
  card: { backgroundColor: '#141a32', padding: 12, borderRadius: 8, marginBottom: 8 },
  cardTitle: { color: '#fff', fontSize: 16, marginBottom: 4 },
  cardMeta: { color: '#9aa5ce', fontSize: 12 },
});
