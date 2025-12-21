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
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import { normalizePuzzle } from '../../model/normalize';
import { parsePuzzle } from '../../model/parser';
import { SAMPLE_PUZZLES } from '../../samples';
import { solvePuzzle } from '../../solver/solver';
import { ApiKeyMode, ExtractionStrategy, getSettings, saveSettings } from '../../storage/puzzles';
import { validateSolution } from '../../validator/validateSolution';
import { validatePuzzleSpec } from '../../validator/validateSpec';
import { colors, spacing, radii } from '../../theme';
import { fontFamilies } from '../../theme/fonts';
import { Button, Card, Badge, Heading, Body, Label, Mono, Screen } from '../components/ui';

const STRATEGY_OPTIONS: { value: ExtractionStrategy; label: string; description: string }[] = [
  { value: 'fast', label: 'Fast', description: 'Single model (~5s)' },
  { value: 'balanced', label: 'Balanced', description: 'With verification (~15s)' },
  { value: 'accurate', label: 'Accurate', description: 'Multi-stage (~25s)' },
  { value: 'ensemble', label: 'Maximum', description: '3-model consensus (~45s)' },
];

const API_MODE_OPTIONS: { value: ApiKeyMode; label: string; description: string }[] = [
  { value: 'openrouter', label: 'OpenRouter', description: 'One key for all models' },
  { value: 'individual', label: 'Individual', description: 'Separate provider keys' },
];

const CV_SERVICE_PLACEHOLDER = 'http://your-cv-service:8080';

export default function SettingsScreen({ navigation }: any) {
  const [settings, setSettings] = useState({
    defaultMaxPip: 6,
    defaultAllowDuplicates: false,
    defaultFindAll: false,
    defaultDebugLevel: 0,
    maxIterationsPerTick: 100,
    apiKeyMode: 'openrouter' as ApiKeyMode,
    openrouterApiKey: '',
    anthropicApiKey: '',
    googleApiKey: '',
    openaiApiKey: '',
    extractionStrategy: 'ensemble' as ExtractionStrategy,
    useMultiStagePipeline: true,
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
      apiKeyMode: loaded.apiKeyMode || 'openrouter',
      openrouterApiKey: loaded.openrouterApiKey || '',
      anthropicApiKey: loaded.anthropicApiKey || '',
      googleApiKey: loaded.googleApiKey || '',
      openaiApiKey: loaded.openaiApiKey || '',
      extractionStrategy: loaded.extractionStrategy || 'ensemble',
      useMultiStagePipeline: loaded.useMultiStagePipeline !== false,
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
    <Screen>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(400)} style={styles.header}>
        <Button
          variant="ghost"
          size="small"
          title="Back"
          onPress={() => navigation.goBack()}
        />
        <Heading size="medium" style={styles.title}>Settings</Heading>
        <View style={styles.placeholder} />
      </Animated.View>

      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Solver Defaults Section */}
          <Animated.View entering={FadeInUp.delay(100).duration(400)}>
            <Card variant="elevated" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>Solver Defaults</Heading>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Max Pip Value</Label>
                  <Body size="small" color="tertiary">Maximum pip value for dominoes (0-N)</Body>
                </View>
                <View style={styles.settingControl}>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('defaultMaxPip', Math.max(0, settings.defaultMaxPip - 1))
                    }
                  >
                    <Mono style={styles.incrementText}>-</Mono>
                  </TouchableOpacity>
                  <Mono style={styles.settingValue}>{settings.defaultMaxPip}</Mono>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() => updateSetting('defaultMaxPip', settings.defaultMaxPip + 1)}
                  >
                    <Mono style={styles.incrementText}>+</Mono>
                  </TouchableOpacity>
                </View>
              </View>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Allow Duplicate Dominoes</Label>
                  <Body size="small" color="tertiary">Allow same domino to be used multiple times</Body>
                </View>
                <Switch
                  value={settings.defaultAllowDuplicates}
                  onValueChange={value => updateSetting('defaultAllowDuplicates', value)}
                  trackColor={{ false: colors.surface.graphite, true: colors.accent.brass }}
                  thumbColor={settings.defaultAllowDuplicates ? colors.accent.brassLight : colors.surface.ash}
                />
              </View>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Find All Solutions</Label>
                  <Body size="small" color="tertiary">Find all solutions vs first solution only</Body>
                </View>
                <Switch
                  value={settings.defaultFindAll}
                  onValueChange={value => updateSetting('defaultFindAll', value)}
                  trackColor={{ false: colors.surface.graphite, true: colors.accent.brass }}
                  thumbColor={settings.defaultFindAll ? colors.accent.brassLight : colors.surface.ash}
                />
              </View>
            </Card>
          </Animated.View>

          {/* Performance Section */}
          <Animated.View entering={FadeInUp.delay(200).duration(400)}>
            <Card variant="elevated" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>Performance</Heading>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Iterations Per Tick</Label>
                  <Body size="small" color="tertiary">Higher = faster but UI may lag</Body>
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
                    <Mono style={styles.incrementText}>-</Mono>
                  </TouchableOpacity>
                  <Mono style={styles.settingValue}>{settings.maxIterationsPerTick}</Mono>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('maxIterationsPerTick', settings.maxIterationsPerTick + 50)
                    }
                  >
                    <Mono style={styles.incrementText}>+</Mono>
                  </TouchableOpacity>
                </View>
              </View>
            </Card>
          </Animated.View>

          {/* Debug Section */}
          <Animated.View entering={FadeInUp.delay(300).duration(400)}>
            <Card variant="elevated" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>Debug</Heading>

              <View style={styles.setting}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Debug Level</Label>
                  <Body size="small" color="tertiary">0=Off, 1=Basic, 2=Verbose</Body>
                </View>
                <View style={styles.settingControl}>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('defaultDebugLevel', Math.max(0, settings.defaultDebugLevel - 1))
                    }
                  >
                    <Mono style={styles.incrementText}>-</Mono>
                  </TouchableOpacity>
                  <Mono style={styles.settingValue}>{settings.defaultDebugLevel}</Mono>
                  <TouchableOpacity
                    style={styles.incrementButton}
                    onPress={() =>
                      updateSetting('defaultDebugLevel', Math.min(2, settings.defaultDebugLevel + 1))
                    }
                  >
                    <Mono style={styles.incrementText}>+</Mono>
                  </TouchableOpacity>
                </View>
              </View>

              <Button
                variant="secondary"
                size="medium"
                title="Run Solver Self-Test"
                onPress={handleSelfTest}
                style={styles.selfTestButton}
              />
            </Card>
          </Animated.View>

          {/* AI Extraction Section */}
          <Animated.View entering={FadeInUp.delay(400).duration(400)}>
            <Card variant="elevated" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>AI Extraction</Heading>
              <Body size="small" color="secondary" style={styles.sectionSubtitle}>
                5-stage multi-model extraction with Gemini 3 Pro, GPT-5.2, and Claude Opus 4.5.
              </Body>

              {/* Strategy Selector */}
              <Label size="medium" style={styles.strategyLabel}>Extraction Strategy</Label>
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
                    <Label
                      size="small"
                      style={[
                        styles.strategyOptionLabel,
                        settings.extractionStrategy === option.value && styles.strategyOptionLabelSelected,
                      ]}
                    >
                      {option.label}
                    </Label>
                    <Body
                      size="small"
                      style={[
                        styles.strategyOptionDesc,
                        settings.extractionStrategy === option.value && styles.strategyOptionDescSelected,
                      ]}
                    >
                      {option.description}
                    </Body>
                  </TouchableOpacity>
                ))}
              </View>

              {/* API Key Mode Selector */}
              <Label size="medium" style={styles.strategyLabel}>API Key Mode</Label>
              <View style={styles.strategySelector}>
                {API_MODE_OPTIONS.map(option => (
                  <TouchableOpacity
                    key={option.value}
                    style={[
                      styles.strategyOption,
                      settings.apiKeyMode === option.value && styles.strategyOptionSelected,
                    ]}
                    onPress={() => updateSetting('apiKeyMode', option.value)}
                  >
                    <Label
                      size="small"
                      style={[
                        styles.strategyOptionLabel,
                        settings.apiKeyMode === option.value && styles.strategyOptionLabelSelected,
                      ]}
                    >
                      {option.label}
                    </Label>
                    <Body
                      size="small"
                      style={[
                        styles.strategyOptionDesc,
                        settings.apiKeyMode === option.value && styles.strategyOptionDescSelected,
                      ]}
                    >
                      {option.description}
                    </Body>
                  </TouchableOpacity>
                ))}
              </View>

              {/* OpenRouter API Key */}
              {settings.apiKeyMode === 'openrouter' && (
                <View style={styles.apiKeySection}>
                  <View style={styles.apiKeyHeader}>
                    <Label size="medium">OpenRouter API Key</Label>
                    <Badge variant="accent" label="Recommended" size="small" />
                  </View>
                  <Body size="small" color="tertiary" style={styles.apiKeyHint}>
                    Single key for Gemini 3 Pro, GPT-5.2, and Claude Opus 4.5
                  </Body>
                  <TextInput
                    style={styles.apiKeyInput}
                    value={settings.openrouterApiKey}
                    onChangeText={value => updateSetting('openrouterApiKey', value)}
                    placeholder="sk-or-v1-..."
                    placeholderTextColor={colors.text.tertiary}
                    secureTextEntry
                    autoCapitalize="none"
                    autoCorrect={false}
                    returnKeyType="done"
                    blurOnSubmit
                    onSubmitEditing={Keyboard.dismiss}
                  />
                </View>
              )}

              {/* Individual API Keys */}
              {settings.apiKeyMode === 'individual' && (
                <>
                  <View style={styles.apiKeySection}>
                    <View style={styles.apiKeyHeader}>
                      <Label size="medium">Google API Key (Gemini 3 Pro)</Label>
                      <Badge variant="info" label="Spatial" size="small" />
                    </View>
                    <Body size="small" color="tertiary" style={styles.apiKeyHint}>
                      Best for grid geometry and spatial understanding
                    </Body>
                    <TextInput
                      style={styles.apiKeyInput}
                      value={settings.googleApiKey}
                      onChangeText={value => updateSetting('googleApiKey', value)}
                      placeholder="AIza..."
                      placeholderTextColor={colors.text.tertiary}
                      secureTextEntry
                      autoCapitalize="none"
                      autoCorrect={false}
                      returnKeyType="done"
                      blurOnSubmit
                      onSubmitEditing={Keyboard.dismiss}
                    />
                  </View>

                  <View style={styles.apiKeySection}>
                    <View style={styles.apiKeyHeader}>
                      <Label size="medium">OpenAI API Key (GPT-5.2)</Label>
                      <Badge variant="info" label="Detail" size="small" />
                    </View>
                    <Body size="small" color="tertiary" style={styles.apiKeyHint}>
                      Best for OCR, pip counting, and fine visual detail
                    </Body>
                    <TextInput
                      style={styles.apiKeyInput}
                      value={settings.openaiApiKey}
                      onChangeText={value => updateSetting('openaiApiKey', value)}
                      placeholder="sk-..."
                      placeholderTextColor={colors.text.tertiary}
                      secureTextEntry
                      autoCapitalize="none"
                      autoCorrect={false}
                      returnKeyType="done"
                      blurOnSubmit
                      onSubmitEditing={Keyboard.dismiss}
                    />
                  </View>

                  <View style={styles.apiKeySection}>
                    <View style={styles.apiKeyHeader}>
                      <Label size="medium">Anthropic API Key (Claude Opus 4.5)</Label>
                      <Badge variant="accent" label="Validation" size="small" />
                    </View>
                    <Body size="small" color="tertiary" style={styles.apiKeyHint}>
                      Best for instruction following and structured output
                    </Body>
                    <TextInput
                      style={styles.apiKeyInput}
                      value={settings.anthropicApiKey}
                      onChangeText={value => updateSetting('anthropicApiKey', value)}
                      placeholder="sk-ant-..."
                      placeholderTextColor={colors.text.tertiary}
                      secureTextEntry
                      autoCapitalize="none"
                      autoCorrect={false}
                      returnKeyType="done"
                      blurOnSubmit
                      onSubmitEditing={Keyboard.dismiss}
                    />
                  </View>
                </>
              )}

              {/* Status */}
              <View style={styles.apiStatus}>
                <Label size="small" color="secondary">API Status:</Label>
                <View style={styles.statusBadges}>
                  {settings.apiKeyMode === 'openrouter' ? (
                    settings.openrouterApiKey ? (
                      <Badge variant="success" label="OpenRouter Ready" size="small" />
                    ) : (
                      <Badge variant="warning" label="Add OpenRouter Key" size="small" />
                    )
                  ) : (
                    <>
                      {settings.googleApiKey ? (
                        <Badge variant="success" label="Gemini" size="small" />
                      ) : (
                        <Badge variant="info" label="Gemini" size="small" />
                      )}
                      {settings.openaiApiKey ? (
                        <Badge variant="success" label="GPT-5.2" size="small" />
                      ) : (
                        <Badge variant="info" label="GPT-5.2" size="small" />
                      )}
                      {settings.anthropicApiKey ? (
                        <Badge variant="success" label="Claude" size="small" />
                      ) : (
                        <Badge variant="info" label="Claude" size="small" />
                      )}
                    </>
                  )}
                </View>
              </View>

              {/* Advanced Options */}
              <View style={[styles.setting, { marginTop: spacing[4] }]}>
                <View style={styles.settingInfo}>
                  <Label size="medium">Save Debug Responses</Label>
                  <Body size="small" color="tertiary">Store raw AI responses for troubleshooting</Body>
                </View>
                <Switch
                  value={settings.saveDebugResponses}
                  onValueChange={value => updateSetting('saveDebugResponses', value)}
                  trackColor={{ false: colors.surface.graphite, true: colors.accent.brass }}
                  thumbColor={settings.saveDebugResponses ? colors.accent.brassLight : colors.surface.ash}
                />
              </View>
            </Card>
          </Animated.View>

          {/* CV Service Section */}
          <Animated.View entering={FadeInUp.delay(500).duration(400)}>
            <Card variant="elevated" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>CV Service (Optional)</Heading>
              <Body size="small" color="secondary" style={styles.sectionSubtitle}>
                For improved accuracy, you can run a CV service that crops images before AI analysis.
                Leave empty to use pure AI mode.
              </Body>

              <View style={styles.apiKeySection}>
                <View style={styles.apiKeyHeader}>
                  <Label size="medium">CV Service URL</Label>
                  {settings.cvServiceUrl ? (
                    <Badge variant="success" label="Configured" size="small" />
                  ) : (
                    <Badge variant="warning" label="Optional" size="small" />
                  )}
                </View>
                <Body size="small" color="tertiary" style={styles.apiKeyHint}>
                  {settings.cvServiceUrl
                    ? 'Hybrid CV+AI mode enabled for better accuracy'
                    : 'Pure AI mode (no CV service needed)'}
                </Body>
                <TextInput
                  style={styles.apiKeyInput}
                  value={settings.cvServiceUrl}
                  onChangeText={value => updateSetting('cvServiceUrl', value)}
                  placeholder={CV_SERVICE_PLACEHOLDER}
                  placeholderTextColor={colors.text.tertiary}
                  autoCapitalize="none"
                  autoCorrect={false}
                  keyboardType="url"
                  returnKeyType="done"
                  blurOnSubmit
                  onSubmitEditing={Keyboard.dismiss}
                />
              </View>
            </Card>
          </Animated.View>

          {/* About Section */}
          <Animated.View entering={FadeInUp.delay(600).duration(400)}>
            <Card variant="default" style={styles.section}>
              <Heading size="small" style={styles.sectionTitle}>About</Heading>
              <Body size="small" color="secondary">Pips Solver v1.0.0</Body>
              <Body size="small" color="tertiary">
                NYT Pips puzzle solver using constraint satisfaction
              </Body>
            </Card>
          </Animated.View>

          {/* Bottom spacing */}
          <View style={{ height: spacing[6] }} />
        </ScrollView>

        {/* Footer */}
        <Animated.View entering={FadeInUp.delay(700).duration(400)} style={styles.footer}>
          <Button
            variant="success"
            size="large"
            title="Save Settings"
            onPress={handleSave}
            style={styles.saveButton}
          />
        </Animated.View>
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[3],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.slate,
  },
  title: {
    flex: 1,
    textAlign: 'center',
  },
  placeholder: {
    width: 70,
  },
  container: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing[4],
    paddingBottom: spacing[20],
  },
  section: {
    marginTop: spacing[4],
  },
  sectionTitle: {
    marginBottom: spacing[4],
  },
  sectionSubtitle: {
    marginBottom: spacing[4],
    marginTop: -spacing[2],
  },
  setting: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing[3],
    borderBottomWidth: 1,
    borderBottomColor: colors.surface.slate,
  },
  settingInfo: {
    flex: 1,
    marginRight: spacing[4],
  },
  settingControl: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  incrementButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.accent.brass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  incrementText: {
    fontSize: 20,
    color: colors.text.inverse,
  },
  settingValue: {
    fontSize: 18,
    color: colors.text.primary,
    marginHorizontal: spacing[4],
    minWidth: 40,
    textAlign: 'center',
  },
  selfTestButton: {
    marginTop: spacing[4],
  },
  strategyLabel: {
    marginBottom: spacing[3],
  },
  strategySelector: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing[2],
    marginBottom: spacing[4],
  },
  strategyOption: {
    flex: 1,
    minWidth: '45%',
    padding: spacing[3],
    borderRadius: radii.md,
    borderWidth: 2,
    borderColor: colors.surface.slate,
    backgroundColor: colors.surface.charcoal,
  },
  strategyOptionSelected: {
    borderColor: colors.accent.brass,
    backgroundColor: colors.surface.slate,
  },
  strategyOptionLabel: {
    color: colors.text.secondary,
  },
  strategyOptionLabelSelected: {
    color: colors.accent.brass,
  },
  strategyOptionDesc: {
    color: colors.text.tertiary,
    marginTop: spacing[1],
  },
  strategyOptionDescSelected: {
    color: colors.text.secondary,
  },
  apiKeySection: {
    marginBottom: spacing[4],
    paddingTop: spacing[3],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
  },
  apiKeyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[1],
  },
  apiKeyHint: {
    marginBottom: spacing[2],
  },
  apiKeyInput: {
    backgroundColor: colors.surface.charcoal,
    borderWidth: 1,
    borderColor: colors.surface.slate,
    borderRadius: radii.md,
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[3],
    fontSize: 14,
    fontFamily: fontFamilies.monoRegular,
    color: colors.text.primary,
  },
  apiStatus: {
    marginTop: spacing[4],
    padding: spacing[3],
    backgroundColor: colors.surface.charcoal,
    borderRadius: radii.md,
  },
  statusBadges: {
    flexDirection: 'row',
    gap: spacing[2],
    marginTop: spacing[2],
  },
  footer: {
    padding: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.surface.slate,
    backgroundColor: colors.surface.charcoal,
  },
  saveButton: {
    width: '100%',
  },
});
