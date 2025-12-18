import AsyncStorage from '@react-native-async-storage/async-storage';

export interface StoredPuzzle {
  id: string;
  name: string;
  text: string;
}

const KEY = 'pips-library';

export async function loadPuzzles(): Promise<StoredPuzzle[]> {
  const raw = await AsyncStorage.getItem(KEY);
  if (!raw) return [];
  return JSON.parse(raw);
}

export async function savePuzzle(entry: StoredPuzzle): Promise<void> {
  const existing = await loadPuzzles();
  const next = [entry, ...existing.filter((p) => p.id !== entry.id)];
  await AsyncStorage.setItem(KEY, JSON.stringify(next));
}
