import * as React from 'react';
import { Pressable, StyleSheet, Switch, Text, TextInput, View } from 'react-native';
import { theme } from '../theme';
import { loadSettings, saveSettings, type AppSettings } from '../../storage/puzzles';

export function SettingsScreen() {
  const [settings, setSettings] = React.useState<AppSettings | null>(null);

  React.useEffect(() => {
    (async () => setSettings(await loadSettings()))();
  }, []);

  if (!settings) {
    return (
      <View style={styles.screen}>
        <Text style={styles.muted}>Loading…</Text>
      </View>
    );
  }

  const commit = async (next: AppSettings) => {
    setSettings(next);
    await saveSettings(next);
  };

  return (
    <View style={styles.screen}>
      <Row label="maxPip">
        <TextInput
          value={String(settings.maxPip)}
          onChangeText={(t) => {
            const n = Number.parseInt(t || '0', 10);
            if (Number.isFinite(n)) commit({ ...settings, maxPip: Math.max(0, Math.min(12, n)) });
          }}
          keyboardType="number-pad"
          style={styles.input}
        />
      </Row>

      <Row label="allow duplicates">
        <Switch
          value={settings.allowDuplicates}
          onValueChange={(v) => commit({ ...settings, allowDuplicates: v })}
        />
      </Row>

      <Row label="solve mode">
        <View style={{ flexDirection: 'row', gap: 10 }}>
          <Pressable
            style={[styles.chip, settings.solveMode === 'first' && styles.chipOn]}
            onPress={() => commit({ ...settings, solveMode: 'first' })}
          >
            <Text style={styles.chipText}>first</Text>
          </Pressable>
          <Pressable
            style={[styles.chip, settings.solveMode === 'all' && styles.chipOn]}
            onPress={() => commit({ ...settings, solveMode: 'all' })}
          >
            <Text style={styles.chipText}>all</Text>
          </Pressable>
        </View>
      </Row>

      <Row label="debug logging">
        <View style={{ flexDirection: 'row', gap: 10 }}>
          {(['off', 'info', 'trace'] as const).map((lvl) => (
            <Pressable
              key={lvl}
              style={[styles.chip, settings.logLevel === lvl && styles.chipOn]}
              onPress={() => commit({ ...settings, logLevel: lvl })}
            >
              <Text style={styles.chipText}>{lvl}</Text>
            </Pressable>
          ))}
        </View>
      </Row>
    </View>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <View style={styles.row}>
      <Text style={styles.label}>{label}</Text>
      <View style={{ flex: 1, alignItems: 'flex-end' }}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: theme.colors.bg, padding: 16, gap: 14 },
  row: {
    backgroundColor: theme.colors.card,
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: theme.colors.border,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  label: { color: theme.colors.text, fontWeight: '800' },
  muted: { color: theme.colors.muted },
  input: {
    borderWidth: 1,
    borderColor: theme.colors.border,
    borderRadius: 10,
    paddingVertical: 8,
    paddingHorizontal: 10,
    minWidth: 64,
    color: theme.colors.text,
    backgroundColor: '#0E1320',
    textAlign: 'right',
    fontWeight: '800',
  },
  chip: { borderWidth: 1, borderColor: theme.colors.border, borderRadius: 999, paddingVertical: 7, paddingHorizontal: 10 },
  chipOn: { borderColor: theme.colors.accent },
  chipText: { color: theme.colors.text, fontWeight: '800' },
});



