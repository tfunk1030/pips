import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, Alert, ScrollView } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import { parsePuzzle } from '../../model/parser';
import { savePuzzle } from '../../storage/puzzles';
import sampleText from '../../samples/sample1.yaml?raw';

// @ts-ignore bundler raw import fallback
const sampleYaml = typeof sampleText === 'string' ? sampleText : '';

type Props = NativeStackScreenProps<RootStackParamList, 'Editor'>;

export default function PuzzleEditorScreen({ navigation, route }: Props) {
  const [text, setText] = useState(route.params?.initialText ?? sampleYaml);

  const handleParse = async () => {
    try {
      const { puzzle } = parsePuzzle(text);
      await savePuzzle({ id: `puzzle-${Date.now()}`, name: puzzle.name ?? 'Imported Puzzle', text });
      navigation.navigate('Viewer', { puzzle, sourceText: text });
    } catch (err: any) {
      Alert.alert('Parse error', err.message);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Paste YAML or JSON</Text>
      <ScrollView style={styles.editorWrapper}>
        <TextInput
          style={styles.editor}
          multiline
          value={text}
          onChangeText={setText}
          placeholder="rows: 2\ncols: 2\nregions: ..."
          placeholderTextColor="#6c7391"
        />
      </ScrollView>
      <View style={styles.row}>
        <TouchableOpacity style={styles.secondary} onPress={() => setText(sampleYaml)}>
          <Text style={styles.secondaryText}>Load Example</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primary} onPress={handleParse}>
          <Text style={styles.primaryText}>Parse & Save</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0c1021', padding: 16 },
  header: { color: '#fff', fontSize: 18, marginBottom: 8 },
  editorWrapper: { flex: 1, borderColor: '#1f2c4d', borderWidth: 1, borderRadius: 8, marginBottom: 12 },
  editor: { color: '#e6e6e6', padding: 12, minHeight: 300 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  primary: { backgroundColor: '#4a60c4', padding: 12, borderRadius: 8, flex: 1, marginLeft: 8 },
  primaryText: { color: '#fff', textAlign: 'center', fontWeight: 'bold' },
  secondary: { borderWidth: 1, borderColor: '#4a60c4', padding: 12, borderRadius: 8, flex: 1, marginRight: 8 },
  secondaryText: { color: '#4a60c4', textAlign: 'center', fontWeight: 'bold' },
});
