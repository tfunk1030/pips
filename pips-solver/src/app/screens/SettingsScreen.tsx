/**
 * Settings Screen
 * Configure solver and app settings
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Switch,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { normalizePuzzle } from '../../model/normalize';
import { parsePuzzle } from '../../model/parser';
import { SAMPLE_PUZZLES } from '../../samples';
import { solvePuzzle } from '../../solver/solver';
import { getSettings, saveSettings } from '../../storage/puzzles';
import { validateSolution } from '../../validator/validateSolution';
import { validatePuzzleSpec } from '../../validator/validateSpec';

export default function SettingsScreen({ navigation }: any) {
  const [settings, setSettings] = useState({
    defaultMaxPip: 6,
    defaultAllowDuplicates: false,
    defaultFindAll: false,
    defaultDebugLevel: 0,
    maxIterationsPerTick: 100,
    anthropicApiKey: '',
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const loaded = await getSettings();
    setSettings({
      ...loaded,
      anthropicApiKey: loaded.anthropicApiKey || '',
    });
  };

  const handleSave = async () => {
    Keyboard.dismiss();
    await saveSettings(settings);
    Alert.alert('Success', 'Settings saved');
    navigation.goBack();
  };

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const handleSelfTest = async () => {
    try {
      const results: string[] = [];
      let failed = 0;

      for (const sample of SAMPLE_PUZZLES) {
        const parsed = parsePuzzle(sample.yaml);
        if (!parsed.success || !parsed.spec) {
          failed++;
          results.push(`${sample.id}: PARSE FAIL - ${parsed.error || 'unknown'}`);
          continue;
        }

        const specValidation = validatePuzzleSpec(parsed.spec);
        if (!specValidation.valid) {
          failed++;
          results.push(`${sample.id}: SPEC INVALID - ${specValidation.errors.join('; ')}`);
          continue;
        }

        const normalized = normalizePuzzle(parsed.spec);
        const config = {
          maxPip: parsed.spec.maxPip || 6,
          allowDuplicates: !!parsed.spec.allowDuplicates,
          findAll: false,
          maxIterationsPerTick: settings.maxIterationsPerTick,
          debugLevel: 0 as 0,
        };

        const solved = solvePuzzle(normalized, config);
        if (!solved.success || solved.solutions.length === 0) {
          failed++;
          results.push(`${sample.id}: UNSAT (${solved.explanation.message})`);
          continue;
        }

        const v = validateSolution(normalized, solved.solutions[0]);
        if (!v.valid) {
          failed++;
          results.push(`${sample.id}: SOLUTION INVALID - ${v.errors[0] || 'unknown error'}`);
          continue;
        }

        results.push(
          `${sample.id}: OK (${solved.stats.timeMs}ms, nodes=${solved.stats.nodes}, prunes=${solved.stats.prunes})`
        );
      }

      Alert.alert(failed === 0 ? 'Self-Test Passed' : 'Self-Test Issues', results.join('\n'));
    } catch (e) {
      Alert.alert('Self-Test Error', String(e));
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backButtonText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.title}>Settings</Text>
        <View style={styles.placeholder} />
      </View>

      <KeyboardAvoidingView style={styles.content} behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>
        <ScrollView
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          contentContainerStyle={styles.scrollContent}
        >
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Solver Defaults</Text>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Max Pip Value</Text>
                  <Text style={styles.settingDescription}>
                    Maximum pip value for dominoes (0-N)
                  </Text>
                </View>
                <View style={styles.settingControl}>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('defaultMaxPip', Math.max(0, settings.defaultMaxPip - 1))
                    }
                  >
                    <Text style={styles.incrementButtonText}>-</Text>
                  </TouchableOpacity>
                  <Text style={styles.settingValue}>{settings.defaultMaxPip}</Text>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() => updateSetting('defaultMaxPip', settings.defaultMaxPip + 1)}
                  >
                    <Text style={styles.incrementButtonText}>+</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Allow Duplicate Dominoes</Text>
                  <Text style={styles.settingDescription}>
                    Allow same domino to be used multiple times
                  </Text>
                </View>
                <Switch
                  value={settings.defaultAllowDuplicates}
                  onValueChange={value => updateSetting('defaultAllowDuplicates', value)}
                />
              </View>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Find All Solutions</Text>
                  <Text style={styles.settingDescription}>
                    Find all solutions vs first solution only
                  </Text>
                </View>
                <Switch
                  value={settings.defaultFindAll}
                  onValueChange={value => updateSetting('defaultFindAll', value)}
                />
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Performance</Text>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Iterations Per Tick</Text>
                  <Text style={styles.settingDescription}>Higher = faster but UI may lag</Text>
                </View>
                <View style={styles.settingControl}>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting(
                        'maxIterationsPerTick',
                        Math.max(10, settings.maxIterationsPerTick - 50)
                      )
                    }
                  >
                    <Text style={styles.incrementButtonText}>-</Text>
                  </TouchableOpacity>
                  <Text style={styles.settingValue}>{settings.maxIterationsPerTick}</Text>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('maxIterationsPerTick', settings.maxIterationsPerTick + 50)
                    }
                  >
                    <Text style={styles.incrementButtonText}>+</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Debug</Text>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Debug Level</Text>
                  <Text style={styles.settingDescription}>0=Off, 1=Basic, 2=Verbose</Text>
                </View>
                <View style={styles.settingControl}>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting(
                        'defaultDebugLevel',
                        Math.max(0, settings.defaultDebugLevel - 1)
                      )
                    }
                  >
                    <Text style={styles.incrementButtonText}>-</Text>
                  </TouchableOpacity>
                  <Text style={styles.settingValue}>{settings.defaultDebugLevel}</Text>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting(
                        'defaultDebugLevel',
                        Math.min(2, settings.defaultDebugLevel + 1)
                      )
                    }
                  >
                    <Text style={styles.incrementButtonText}>+</Text>
                  </TouchableOpacity>
                </View>
              </View>

              <TouchableOpacity style={styles.selfTestButton} onPress={handleSelfTest}>
                <Text style={styles.selfTestButtonText}>Run Solver Self-Test</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.section}>
              <Text style={styles.sectionTitle}>AI Extraction</Text>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>Anthropic API Key</Text>
                  <Text style={styles.settingDescription}>Required for AI puzzle extraction</Text>
                </View>
              </View>
              <TextInput
                style={styles.apiKeyInput}
                value={settings.anthropicApiKey}
                onChangeText={value => updateSetting('anthropicApiKey', value)}
                placeholder="sk-ant-..."
                placeholderTextColor="#999"
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                returnKeyType="done"
                blurOnSubmit
                onSubmitEditing={Keyboard.dismiss}
              />
            </View>

            <View style={styles.infoSection}>
              <Text style={styles.infoTitle}>About</Text>
              <Text style={styles.infoText}>Pips Solver v1.0.0</Text>
              <Text style={styles.infoText}>
                NYT Pips puzzle solver using constraint satisfaction
              </Text>
            </View>
          </ScrollView>

          <View style={styles.footer}>
            <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
              <Text style={styles.saveButtonText}>Save Settings</Text>
            </TouchableOpacity>
          </View>
      </KeyboardAvoidingView>
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
    padding: 16,
    paddingTop: 60,
    backgroundColor: '#fff',
    borderBottomWidth: 1,
    borderBottomColor: '#ddd',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    fontSize: 16,
    color: '#007AFF',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    flex: 1,
    textAlign: 'center',
  },
  placeholder: {
    width: 70,
  },
  content: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 140,
  },
  section: {
    backgroundColor: '#fff',
    marginTop: 16,
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#333',
  },
  setting: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  settingInfo: {
    flex: 1,
    marginRight: 16,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 12,
    color: '#666',
  },
  settingControl: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  incrementButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#007AFF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  incrementButtonText: {
    color: '#fff',
    fontSize: 20,
    fontWeight: 'bold',
  },
  settingValue: {
    fontSize: 18,
    fontWeight: 'bold',
    marginHorizontal: 16,
    minWidth: 40,
    textAlign: 'center',
  },
  infoSection: {
    backgroundColor: '#fff',
    marginTop: 16,
    marginBottom: 16,
    padding: 16,
  },
  infoTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  infoText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  footer: {
    padding: 16,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#ddd',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  apiKeyInput: {
    backgroundColor: '#f9f9f9',
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
    fontFamily: 'monospace',
    marginTop: 8,
  },
  selfTestButton: {
    marginTop: 12,
    backgroundColor: '#222',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  selfTestButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: '600',
  },
});
