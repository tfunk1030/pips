import AsyncStorage from '@react-native-async-storage/async-storage';

const INDEX_KEY = 'pips:puzzles:index:v1';
const PUZZLE_KEY_PREFIX = 'pips:puzzle:v1:';
const SETTINGS_KEY = 'pips:settings:v1';

export type PuzzleRow = {
  id: string;
  name: string;
  text: string;
  createdAt: number;
  updatedAt: number;
};

export type AppSettings = {
  maxPip: number;
  allowDuplicates: boolean;
  solveMode: 'first' | 'all';
  logLevel: 'off' | 'info' | 'trace';
};

export async function listPuzzles(): Promise<Array<{ id: string; name: string; updatedAt: number }>> {
  const idx = await loadIndex();
  return idx.sort((a, b) => b.updatedAt - a.updatedAt).map(({ id, name, updatedAt }) => ({ id, name, updatedAt }));
}

export async function getPuzzle(id: string): Promise<PuzzleRow | null> {
  const raw = await AsyncStorage.getItem(PUZZLE_KEY_PREFIX + id);
  if (!raw) return null;
  return JSON.parse(raw) as PuzzleRow;
}

export async function savePuzzle(input: { id?: string; name: string; text: string }): Promise<string> {
  const now = Date.now();
  const id = input.id ?? randomId();
  const existing = await getPuzzle(id);
  const row: PuzzleRow = {
    id,
    name: input.name,
    text: input.text,
    createdAt: existing?.createdAt ?? now,
    updatedAt: now,
  };

  await AsyncStorage.setItem(PUZZLE_KEY_PREFIX + id, JSON.stringify(row));

  const idx = await loadIndex();
  const next = [
    ...idx.filter((x) => x.id !== id),
    { id, name: row.name, updatedAt: row.updatedAt },
  ];
  await AsyncStorage.setItem(INDEX_KEY, JSON.stringify(next));
  return id;
}

export async function deletePuzzle(id: string): Promise<void> {
  await AsyncStorage.removeItem(PUZZLE_KEY_PREFIX + id);
  const idx = await loadIndex();
  const next = idx.filter((x) => x.id !== id);
  await AsyncStorage.setItem(INDEX_KEY, JSON.stringify(next));
}

export async function loadSettings(): Promise<AppSettings> {
  const raw = await AsyncStorage.getItem(SETTINGS_KEY);
  if (!raw) return defaultSettings();
  try {
    const s = JSON.parse(raw) as Partial<AppSettings>;
    return {
      maxPip: clampInt(s.maxPip ?? 6, 0, 12),
      allowDuplicates: !!s.allowDuplicates,
      solveMode: s.solveMode === 'all' ? 'all' : 'first',
      logLevel: s.logLevel === 'trace' ? 'trace' : s.logLevel === 'info' ? 'info' : 'off',
    };
  } catch {
    return defaultSettings();
  }
}

export async function saveSettings(s: AppSettings): Promise<void> {
  const clean: AppSettings = {
    maxPip: clampInt(s.maxPip, 0, 12),
    allowDuplicates: !!s.allowDuplicates,
    solveMode: s.solveMode,
    logLevel: s.logLevel,
  };
  await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(clean));
}

function defaultSettings(): AppSettings {
  return { maxPip: 6, allowDuplicates: false, solveMode: 'first', logLevel: 'off' };
}

async function loadIndex(): Promise<Array<{ id: string; name: string; updatedAt: number }>> {
  const raw = await AsyncStorage.getItem(INDEX_KEY);
  if (!raw) return [];
  try {
    const x = JSON.parse(raw);
    if (!Array.isArray(x)) return [];
    return x
      .filter((r) => r && typeof r.id === 'string')
      .map((r) => ({ id: r.id as string, name: String(r.name ?? 'Untitled'), updatedAt: Number(r.updatedAt ?? 0) }));
  } catch {
    return [];
  }
}

function randomId(): string {
  // Deterministic-enough, offline-friendly, no crypto dependency.
  return Math.random().toString(36).slice(2, 10) + '-' + Date.now().toString(36);
}

function clampInt(n: number, lo: number, hi: number): number {
  const x = Math.trunc(Number(n));
  if (!Number.isFinite(x)) return lo;
  return Math.max(lo, Math.min(hi, x));
}



