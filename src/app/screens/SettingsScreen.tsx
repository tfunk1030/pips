import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Switch } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'pips-settings';

interface SettingsState {
  allowDuplicates: boolean;
  findAll: boolean;
}

export default function SettingsScreen() {
  const [settings, setSettings] = useState<SettingsState>({ allowDuplicates: false, findAll: false });

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((stored) => {
      if (stored) {
        setSettings(JSON.parse(stored));
      }
    });
  }, []);

  const update = (partial: Partial<SettingsState>) => {
    const next = { ...settings, ...partial };
    setSettings(next);
    AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Solver defaults</Text>
      <View style={styles.row}>
        <Text style={styles.label}>Allow duplicate dominoes</Text>
        <Switch value={settings.allowDuplicates} onValueChange={(v) => update({ allowDuplicates: v })} />
      </View>
      <View style={styles.row}>
        <Text style={styles.label}>Find all solutions</Text>
        <Switch value={settings.findAll} onValueChange={(v) => update({ findAll: v })} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0c1021', padding: 16 },
  header: { color: '#fff', fontSize: 18, marginBottom: 12 },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  label: { color: '#e6e6e6' },
});
