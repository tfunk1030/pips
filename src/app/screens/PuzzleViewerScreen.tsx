import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import { NativeStackScreenProps } from '@react-navigation/native-stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import PuzzleGrid from '../components/PuzzleGrid';

import { validateSpec } from '../../validator/validateSpec';

import { PuzzleSpec } from '../../model/types';

type Props = NativeStackScreenProps<RootStackParamList, 'Viewer'>;

export default function PuzzleViewerScreen({ navigation, route }: Props) {
  const { puzzle, sourceText } = route.params;
  const specReport = validateSpec(puzzle);

  const renderSpecIssues = () =>
    specReport.issues.map((issue, idx) => (
      <Text key={idx} style={{ color: issue.level === 'error' ? '#f78c6c' : '#9aa5ce' }}>
        • {issue.message}
      </Text>
    ));

  return (
    <ScrollView style={styles.container} contentContainerStyle={{ paddingBottom: 24 }}>
      <Text style={styles.header}>{puzzle.name ?? 'Puzzle'}</Text>
      <PuzzleGrid puzzle={puzzle} />
      <Text style={styles.section}>Spec validation</Text>
      {renderSpecIssues()}
      <View style={styles.buttonRow}>
        <TouchableOpacity style={styles.secondary} onPress={() => navigation.navigate('Editor', { initialText: sourceText })}>
          <Text style={styles.secondaryText}>Edit</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.primary} onPress={() => navigation.navigate('Solve', { puzzle, sourceText })}>
          <Text style={styles.primaryText}>Solve</Text>
        </TouchableOpacity>
      </View>
      <Text style={styles.section}>YAML</Text>
      <Text selectable style={styles.code}>
        {sourceText}
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0c1021', padding: 16 },
  header: { color: '#fff', fontSize: 22, marginBottom: 12 },
  section: { color: '#9aa5ce', marginTop: 16, marginBottom: 4 },
  buttonRow: { flexDirection: 'row', marginTop: 12 },
  primary: { backgroundColor: '#4a60c4', padding: 12, borderRadius: 8, flex: 1, marginLeft: 8 },
  primaryText: { color: '#fff', textAlign: 'center', fontWeight: 'bold' },
  secondary: { borderWidth: 1, borderColor: '#4a60c4', padding: 12, borderRadius: 8, flex: 1, marginRight: 8 },
  secondaryText: { color: '#4a60c4', textAlign: 'center', fontWeight: 'bold' },
  code: { backgroundColor: '#11162a', color: '#e6e6e6', padding: 12, borderRadius: 8, fontFamily: 'monospace' },
});
