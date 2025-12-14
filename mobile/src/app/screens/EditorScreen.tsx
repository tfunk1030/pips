import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as React from 'react';
import { Alert, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import type { RootStackParamList } from '../navigation/RootNavigator';
import { theme } from '../theme';
import { savePuzzle } from '../../storage/puzzles';

type Props = NativeStackScreenProps<RootStackParamList, 'Editor'>;

export function EditorScreen({ navigation, route }: Props) {
  const [name, setName] = React.useState('Untitled');
  const [text, setText] = React.useState(route.params?.initialText ?? '');

  const onSave = async () => {
    const trimmed = text.trim();
    if (!trimmed) {
      Alert.alert('Nothing to save', 'Paste YAML/JSON first.');
      return;
    }
    const id = await savePuzzle({ name: name.trim() || 'Untitled', text: trimmed });
    navigation.replace('Puzzle', { puzzleId: id });
  };

  return (
    <View style={styles.screen}>
      <Text style={styles.label}>Name</Text>
      <TextInput
        value={name}
        onChangeText={setName}
        placeholder="Puzzle name"
        placeholderTextColor={theme.colors.muted}
        style={styles.input}
      />

      <View style={styles.actions}>
        <Pressable style={styles.primaryBtn} onPress={onSave}>
          <Text style={styles.primaryBtnText}>Save + Open</Text>
        </Pressable>
        <Pressable
          style={styles.secondaryBtn}
          onPress={() => navigation.navigate('Puzzle', { puzzleText: text })}
        >
          <Text style={styles.secondaryBtnText}>Open (no save)</Text>
        </Pressable>
      </View>

      <Text style={styles.label}>YAML / JSON</Text>
      <ScrollView style={styles.editorWrap} contentContainerStyle={{ paddingBottom: 16 }}>
        <TextInput
          value={text}
          onChangeText={setText}
          placeholder="Paste puzzle YAML/JSON here…"
          placeholderTextColor={theme.colors.muted}
          style={styles.editor}
          multiline
          autoCapitalize="none"
          autoCorrect={false}
          textAlignVertical="top"
        />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: theme.colors.bg, padding: 16, gap: 10 },
  label: { color: theme.colors.text, fontWeight: '800' },
  input: {
    borderWidth: 1,
    borderColor: theme.colors.border,
    borderRadius: 10,
    padding: 10,
    color: theme.colors.text,
    backgroundColor: theme.colors.card,
  },
  actions: { flexDirection: 'row', gap: 12 },
  primaryBtn: { backgroundColor: theme.colors.accent, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  primaryBtnText: { color: '#08101F', fontWeight: '800' },
  secondaryBtn: { borderColor: theme.colors.border, borderWidth: 1, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  secondaryBtnText: { color: theme.colors.text, fontWeight: '700' },
  editorWrap: { flex: 1, borderWidth: 1, borderColor: theme.colors.border, borderRadius: 10, backgroundColor: theme.colors.card },
  editor: {
    minHeight: 420,
    padding: 10,
    color: theme.colors.text,
    fontFamily: undefined,
    fontSize: 13,
    lineHeight: 18,
  },
});



