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
import { ExtractionStrategy, getSettings, saveSettings } from '../../storage/puzzles';
import { validateSolution } from '../../validator/validateSolution';
import { validatePuzzleSpec } from '../../validator/validateSpec';

const STRATEGY_OPTIONS: { value: ExtractionStrategy; label: string; description: string }[] = [
  { value: 'fast', label: '⚡ Fast', description: 'Single model (~3s)' },
  { value: 'balanced', label: '⚖️ Balanced', description: 'With verification (~20s)' },
  { value: 'accurate', label: '🎯 Accurate', description: 'Multi-model (~35s)' },
  { value: 'ensemble', label: '🏆 Maximum', description: 'Ensemble consensus (~45s)' },
];

// CV Service URL placeholder - when empty, pure AI mode is used
const CV_SERVICE_PLACEHOLDER = 'http://your-cv-service:8080';

export default function SettingsScreen({ navigation }: any) {
  const [settings, setSettings] = useState({
    defaultMaxPip: 6,
    defaultAllowDuplicates: false,
    defaultFindAll: false,
    defaultDebugLevel: 0,
    maxIterationsPerTick: 100,
    anthropicApiKey: '',
    googleApiKey: '',
    openaiApiKey: '',
    extractionStrategy: 'accurate' as ExtractionStrategy,
    saveDebugResponses: false,
    cvServiceUrl: '',
  });

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    const loaded = await getSettings();
    setSettings({
      ...loaded,
      anthropicApiKey: loaded.anthropicApiKey || '',
      googleApiKey: loaded.googleApiKey || '',
      openaiApiKey: loaded.openaiApiKey || '',
      extractionStrategy: loaded.extractionStrategy || 'accurate',
      saveDebugResponses: loaded.saveDebugResponses || false,
      cvServiceUrl: loaded.cvServiceUrl || '',
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

      <KeyboardAvoidingView
        style={styles.content}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
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
                <Text style={styles.settingDescription}>Maximum pip value for dominoes (0-N)</Text>
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
                    updateSetting('defaultDebugLevel', Math.max(0, settings.defaultDebugLevel - 1))
                  }
                >
                  <Text style={styles.incrementButtonText}>-</Text>
                </TouchableOpacity>
                <Text style={styles.settingValue}>{settings.defaultDebugLevel}</Text>
                <TouchableOpacity
                  style={styles.incrementButton}
                  onPress={() =>
                    updateSetting('defaultDebugLevel', Math.min(2, settings.defaultDebugLevel + 1))
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
            <Text style={styles.sectionSubtitle}>
              Multi-model extraction for maximum accuracy. Add API keys for each provider.
            </Text>

            {/* Extraction Strategy Selector */}
            <View style={styles.setting}>
              <View style={styles.settingInfo}>
                <Text style={styles.settingLabel}>Extraction Strategy</Text>
                <Text style={styles.settingDescription}>
                  Higher accuracy = more time & API calls
                </Text>
              </View>
            </View>
            <View style={styles.strategySelector}>
              {STRATEGY_OPTIONS.map(option => (
                <TouchableOpacity
                  key={option.value}
                  style={[
                    styles.strategyOption,
                    settings.extractionStrategy === option.value && styles.strategyOptionSelected,
                  ]}
                  onPress={() => updateSetting('extractionStrategy', option.value)}
                >
                  <Text
                    style={[
                      styles.strategyLabel,
                      settings.extractionStrategy === option.value && styles.strategyLabelSelected,
                    ]}
                  >
                    {option.label}
                  </Text>
                  <Text
                    style={[
                      styles.strategyDesc,
                      settings.extractionStrategy === option.value && styles.strategyDescSelected,
                    ]}
                  >
                    {option.description}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            {/* Model Comparison Mode */}
            <View style={styles.debugModeSection}>
              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Text style={styles.settingLabel}>🔍 Model Comparison Mode</Text>
                  <Text style={styles.settingDescription}>
                    Store responses from each AI model for side-by-side comparison
                  </Text>
                </View>
                <Switch
                  value={settings.saveDebugResponses}
                  onValueChange={value => updateSetting('saveDebugResponses', value)}
                  trackColor={{ false: '#ccc', true: '#81C784' }}
                  thumbColor={settings.saveDebugResponses ? '#4CAF50' : '#f4f3f4'}
                />
              </View>
              <Text style={styles.debugModeHint}>
                {settings.saveDebugResponses
                  ? '✓ Enabled: After extraction, use "Compare Models" to see per-model results and disagreements'
                  : '○ Disabled: Better performance, but model comparison unavailable'}
              </Text>
              {settings.saveDebugResponses && (
                <View style={styles.debugModeWarning}>
                  <Text style={styles.debugModeWarningText}>
                    ⚠️ Uses additional memory to store raw responses. Recommended for debugging or verification only.
                  </Text>
                </View>
              )}
            </View>

            {/* Google API Key (Gemini) */}
            <View style={styles.apiKeySection}>
              <View style={styles.apiKeyHeader}>
                <Text style={styles.apiKeyLabel}>🔵 Google API Key (Gemini)</Text>
                <Text style={styles.apiKeyBadge}>Best for grid detection</Text>
              </View>
              <Text style={styles.apiKeyHint}>
                Best mAP (13.3) for object/bounding box detection
              </Text>
              <TextInput
                style={styles.apiKeyInput}
                value={settings.googleApiKey}
                onChangeText={value => updateSetting('googleApiKey', value)}
                placeholder="AIza..."
                placeholderTextColor="#999"
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                returnKeyType="done"
                blurOnSubmit
                onSubmitEditing={Keyboard.dismiss}
              />
            </View>

            {/* Anthropic API Key (Claude) */}
            <View style={styles.apiKeySection}>
              <View style={styles.apiKeyHeader}>
                <Text style={styles.apiKeyLabel}>🟠 Anthropic API Key (Claude)</Text>
                <Text style={styles.apiKeyBadge}>Best for JSON output</Text>
              </View>
              <Text style={styles.apiKeyHint}>85% structured JSON accuracy, best reasoning</Text>
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

            {/* OpenAI API Key (GPT-4o) */}
            <View style={styles.apiKeySection}>
              <View style={styles.apiKeyHeader}>
                <Text style={styles.apiKeyLabel}>🟢 OpenAI API Key (GPT-4o)</Text>
                <Text style={[styles.apiKeyBadge, styles.apiKeyBadgeOptional]}>
                  Optional fallback
                </Text>
              </View>
              <Text style={styles.apiKeyHint}>Good general purpose, weaker on spatial tasks</Text>
              <TextInput
                style={styles.apiKeyInput}
                value={settings.openaiApiKey}
                onChangeText={value => updateSetting('openaiApiKey', value)}
                placeholder="sk-..."
                placeholderTextColor="#999"
                secureTextEntry
                autoCapitalize="none"
                autoCorrect={false}
                returnKeyType="done"
                blurOnSubmit
                onSubmitEditing={Keyboard.dismiss}
              />
            </View>

            {/* Status indicator */}
            <View style={styles.apiKeyStatus}>
              <Text style={styles.apiKeyStatusLabel}>Configured providers:</Text>
              <View style={styles.apiKeyStatusBadges}>
                {settings.googleApiKey ? (
                  <Text style={styles.statusBadgeActive}>✓ Gemini</Text>
                ) : (
                  <Text style={styles.statusBadgeInactive}>○ Gemini</Text>
                )}
                {settings.anthropicApiKey ? (
                  <Text style={styles.statusBadgeActive}>✓ Claude</Text>
                ) : (
                  <Text style={styles.statusBadgeInactive}>○ Claude</Text>
                )}
                {settings.openaiApiKey ? (
                  <Text style={styles.statusBadgeActive}>✓ GPT-4o</Text>
                ) : (
                  <Text style={styles.statusBadgeInactive}>○ GPT-4o</Text>
                )}
              </View>
            </View>
          </View>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>CV Service (Optional)</Text>
            <Text style={styles.sectionSubtitle}>
              For improved accuracy, you can run a CV service that crops images before AI analysis.
              Leave empty to use pure AI mode (works great without CV).
            </Text>

            <View style={styles.apiKeySection}>
              <View style={styles.apiKeyHeader}>
                <Text style={styles.apiKeyLabel}>🖼️ CV Service URL</Text>
                {settings.cvServiceUrl ? (
                  <Text style={styles.statusBadgeActive}>✓ Configured</Text>
                ) : (
                  <Text style={[styles.apiKeyBadge, styles.apiKeyBadgeOptional]}>Optional</Text>
                )}
              </View>
              <Text style={styles.apiKeyHint}>
                {settings.cvServiceUrl
                  ? 'Hybrid CV+AI mode enabled for better accuracy'
                  : 'Pure AI mode (no CV service needed)'}
              </Text>
              <TextInput
                style={styles.apiKeyInput}
                value={settings.cvServiceUrl}
                onChangeText={value => updateSetting('cvServiceUrl', value)}
                placeholder={CV_SERVICE_PLACEHOLDER}
                placeholderTextColor="#666"
                autoCapitalize="none"
                autoCorrect={false}
                keyboardType="url"
                returnKeyType="done"
                blurOnSubmit
                onSubmitEditing={Keyboard.dismiss}
              />
              <Text style={styles.cvServiceNote}>
                Deploy cv-service/ to Railway, Render, or Cloud Run for remote access.
              </Text>
            </View>
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
  sectionSubtitle: {
    fontSize: 13,
    color: '#666',
    marginBottom: 16,
    lineHeight: 18,
  },
  strategySelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 20,
  },
  strategyOption: {
    flex: 1,
    minWidth: '45%',
    padding: 12,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#e0e0e0',
    backgroundColor: '#f9f9f9',
  },
  strategyOptionSelected: {
    borderColor: '#4CAF50',
    backgroundColor: '#E8F5E9',
  },
  strategyLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 2,
  },
  strategyLabelSelected: {
    color: '#2E7D32',
  },
  strategyDesc: {
    fontSize: 11,
    color: '#888',
  },
  strategyDescSelected: {
    color: '#4CAF50',
  },
  apiKeySection: {
    marginBottom: 16,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  apiKeyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  apiKeyLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
  },
  apiKeyBadge: {
    fontSize: 10,
    color: '#fff',
    backgroundColor: '#4CAF50',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    overflow: 'hidden',
  },
  apiKeyBadgeOptional: {
    backgroundColor: '#9E9E9E',
  },
  apiKeyHint: {
    fontSize: 11,
    color: '#888',
    marginBottom: 8,
  },
  apiKeyStatus: {
    marginTop: 12,
    padding: 12,
    backgroundColor: '#f5f5f5',
    borderRadius: 8,
  },
  apiKeyStatusLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#666',
    marginBottom: 8,
  },
  apiKeyStatusBadges: {
    flexDirection: 'row',
    gap: 8,
  },
  statusBadgeActive: {
    fontSize: 12,
    color: '#2E7D32',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    overflow: 'hidden',
  },
  statusBadgeInactive: {
    fontSize: 12,
    color: '#999',
    backgroundColor: '#f0f0f0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    overflow: 'hidden',
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
  cvServiceNote: {
    fontSize: 11,
    color: '#666',
    marginTop: 8,
    fontStyle: 'italic',
  },
  debugModeSection: {
    marginBottom: 16,
    paddingTop: 12,
    paddingBottom: 8,
    borderTopWidth: 1,
    borderTopColor: '#eee',
  },
  debugModeHint: {
    fontSize: 12,
    color: '#666',
    marginTop: 4,
    paddingHorizontal: 4,
  },
  debugModeWarning: {
    marginTop: 8,
    padding: 10,
    backgroundColor: '#FFF8E1',
    borderRadius: 6,
    borderLeftWidth: 3,
    borderLeftColor: '#FFA000',
  },
  debugModeWarningText: {
    fontSize: 11,
    color: '#795548',
    lineHeight: 16,
  },
});
