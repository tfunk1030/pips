import { useFocusEffect, useNavigation } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as React from 'react';
import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import { theme } from '../theme';
import type { RootStackParamList } from '../navigation/RootNavigator';
import { listPuzzles } from '../../storage/puzzles';
import { SAMPLE_1_YAML, SAMPLE_NYT_YAML } from '../../samples/samples';

type Nav = NativeStackNavigationProp<RootStackParamList, 'Library'>;

export function LibraryScreen() {
  const navigation = useNavigation<Nav>();
  const [items, setItems] = React.useState<Array<{ id: string; name: string; updatedAt: number }>>([]);

  useFocusEffect(
    React.useCallback(() => {
      let mounted = true;
      (async () => {
        const rows = await listPuzzles();
        if (mounted) setItems(rows);
      })();
      return () => {
        mounted = false;
      };
    }, [])
  );

  return (
    <View style={styles.screen}>
      <View style={styles.headerRow}>
        <Pressable style={styles.primaryBtn} onPress={() => navigation.navigate('Editor')}>
          <Text style={styles.primaryBtnText}>Paste YAML</Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.secondaryBtnText}>Settings</Text>
        </Pressable>
      </View>

      <View style={styles.headerRow}>
        <Pressable style={styles.secondaryBtn} onPress={() => navigation.navigate('Editor', { initialText: SAMPLE_1_YAML })}>
          <Text style={styles.secondaryBtnText}>Load Sample 1</Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={() => navigation.navigate('Editor', { initialText: SAMPLE_NYT_YAML })}>
          <Text style={styles.secondaryBtnText}>Load NYT Sample</Text>
        </Pressable>
      </View>

      <Text style={styles.sectionTitle}>Saved puzzles</Text>
      <FlatList
        data={items}
        keyExtractor={(x) => x.id}
        ItemSeparatorComponent={() => <View style={styles.sep} />}
        renderItem={({ item }) => (
          <Pressable
            style={styles.row}
            onPress={() => navigation.navigate('Puzzle', { puzzleId: item.id })}
          >
            <Text style={styles.rowTitle}>{item.name}</Text>
            <Text style={styles.rowMeta}>{new Date(item.updatedAt).toLocaleString()}</Text>
          </Pressable>
        )}
        ListEmptyComponent={<Text style={styles.muted}>No saved puzzles yet.</Text>}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: theme.colors.bg, padding: 16, gap: 12 },
  headerRow: { flexDirection: 'row', gap: 12 },
  sectionTitle: { color: theme.colors.text, fontSize: 16, fontWeight: '700', marginTop: 8 },
  primaryBtn: { backgroundColor: theme.colors.accent, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  primaryBtnText: { color: '#08101F', fontWeight: '800' },
  secondaryBtn: { borderColor: theme.colors.border, borderWidth: 1, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  secondaryBtnText: { color: theme.colors.text, fontWeight: '700' },
  row: { backgroundColor: theme.colors.card, borderRadius: 12, padding: 12, borderWidth: 1, borderColor: theme.colors.border },
  rowTitle: { color: theme.colors.text, fontSize: 15, fontWeight: '800' },
  rowMeta: { color: theme.colors.muted, marginTop: 4, fontSize: 12 },
  muted: { color: theme.colors.muted, marginTop: 12 },
  sep: { height: 10 },
});



