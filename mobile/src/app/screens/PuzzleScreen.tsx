import type { NativeStackScreenProps } from '@react-navigation/native-stack';
import * as React from 'react';
import { ActivityIndicator, Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import type { RootStackParamList } from '../navigation/RootNavigator';
import { theme } from '../theme';
import { getPuzzle } from '../../storage/puzzles';
import { parsePuzzleText } from '../../model/parser';
import { normalizePuzzle } from '../../model/normalize';
import { validateSpec } from '../../validator/validateSpec';
import { validateSolution } from '../../validator/validateSolution';
import { solvePuzzleAsync } from '../../solver/solver';
import type { SolveProgress, SolveResult } from '../../solver/solver';
import { GridSvg } from '../components/GridSvg';
import { ValidationReportView } from '../components/ValidationReportView';
import { SolverStatsView } from '../components/SolverStatsView';
import { loadSettings } from '../../storage/puzzles';

type Props = NativeStackScreenProps<RootStackParamList, 'Puzzle'>;

export function PuzzleScreen({ route }: Props) {
  const [loading, setLoading] = React.useState(true);
  const [puzzleText, setPuzzleText] = React.useState<string>('');
  const [error, setError] = React.useState<string | null>(null);
  const [puzzle, setPuzzle] = React.useState<ReturnType<typeof normalizePuzzle> | null>(null);
  const [specOk, setSpecOk] = React.useState<boolean>(false);

  const [progress, setProgress] = React.useState<SolveProgress | null>(null);
  const [result, setResult] = React.useState<SolveResult | null>(null);
  const [validReport, setValidReport] = React.useState<ReturnType<typeof validateSolution> | null>(null);

  const abortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        if (route.params?.puzzleText != null) {
          if (mounted) setPuzzleText(route.params.puzzleText);
        } else if (route.params?.puzzleId) {
          const row = await getPuzzle(route.params.puzzleId);
          if (!row) throw new Error('Puzzle not found');
          if (mounted) setPuzzleText(row.text);
        } else {
          throw new Error('No puzzle specified');
        }
        if (mounted) setError(null);
      } catch (e: any) {
        if (mounted) setError(String(e?.message ?? e));
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
      abortRef.current?.abort();
    };
  }, [route.params?.puzzleId, route.params?.puzzleText]);

  React.useEffect(() => {
    try {
      const parsed = parsePuzzleText(puzzleText);
      const norm = normalizePuzzle(parsed);
      const rep = validateSpec(norm);
      setPuzzle(norm);
      setSpecOk(rep.ok);
      setError(rep.ok ? null : rep.errors[0]?.message ?? 'Invalid spec');
    } catch (e: any) {
      setPuzzle(null);
      setSpecOk(false);
      setError(String(e?.message ?? e));
    }
  }, [puzzleText]);

  const onSolve = async () => {
    try {
      setResult(null);
      setValidReport(null);
      setProgress(null);

      if (!puzzle) {
        Alert.alert('Invalid puzzle', error ?? 'Could not parse puzzle.');
        return;
      }

      const specReport = validateSpec(puzzle);
      if (!specReport.ok) {
        Alert.alert('Invalid puzzle spec', specReport.errors[0]?.message ?? 'Invalid spec');
        setResult({ kind: 'invalid_spec', report: specReport });
        return;
      }

      abortRef.current?.abort();
      abortRef.current = new AbortController();
      const settings = await loadSettings();

      const res = await solvePuzzleAsync(puzzle, {
        mode: settings.solveMode,
        allowDuplicates: settings.allowDuplicates,
        maxPipOverride: settings.maxPip,
        yieldEvery: 250,
        onProgress: (p) => setProgress(p),
        signal: abortRef.current.signal,
        logLevel: settings.logLevel,
      });

      setResult(res);

      if (res.kind === 'solved') {
        const rep = validateSolution(puzzle, res.solution);
        setValidReport(rep);
        if (!rep.ok) {
          Alert.alert('Internal error', 'Solver returned a solution that failed validation. The solution will not be shown as solved.');
        }
      }
    } catch (e: any) {
      Alert.alert('Solve error', String(e?.message ?? e));
    }
  };

  const onCancel = () => abortRef.current?.abort();

  if (loading) {
    return (
      <View style={styles.screen}>
        <ActivityIndicator />
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.screen}>
        <Text style={styles.errorTitle}>Error</Text>
        <Text style={styles.errorBody}>{error}</Text>
      </View>
    );
  }

  return (
    <View style={styles.screen}>
      <View style={styles.actions}>
        <Pressable style={styles.primaryBtn} onPress={onSolve}>
          <Text style={styles.primaryBtnText}>Solve</Text>
        </Pressable>
        <Pressable style={styles.secondaryBtn} onPress={onCancel}>
          <Text style={styles.secondaryBtnText}>Cancel</Text>
        </Pressable>
      </View>

      {progress ? (
        <Text style={styles.muted}>
          Searching… nodes={progress.nodes.toLocaleString()} backtracks={progress.backtracks.toLocaleString()}
        </Text>
      ) : null}

      <View style={styles.canvasCard}>
        {puzzle ? (
          <GridSvg
            puzzle={puzzle}
            solution={result?.kind === 'solved' && validReport?.ok ? result.solution : undefined}
          />
        ) : null}
        {!specOk && error ? (
          <View style={{ position: 'absolute', inset: 0, justifyContent: 'center', alignItems: 'center', padding: 16 }}>
            <Text style={styles.errorBody}>{error}</Text>
          </View>
        ) : null}
      </View>

      <ScrollView style={styles.bottom} contentContainerStyle={{ paddingBottom: 24 }}>
        {result ? <SolverStatsView result={result} /> : null}
        {validReport ? <ValidationReportView report={validReport} /> : null}
        {result?.kind === 'unsat' ? (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>Unsatisfiable</Text>
            <Text style={styles.cardBody}>{result.explanation}</Text>
          </View>
        ) : null}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: theme.colors.bg, padding: 16, gap: 10 },
  actions: { flexDirection: 'row', gap: 12 },
  primaryBtn: { backgroundColor: theme.colors.accent, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  primaryBtnText: { color: '#08101F', fontWeight: '900' },
  secondaryBtn: { borderColor: theme.colors.border, borderWidth: 1, paddingVertical: 10, paddingHorizontal: 12, borderRadius: 10 },
  secondaryBtnText: { color: theme.colors.text, fontWeight: '800' },
  canvasCard: { flex: 1, minHeight: 260, borderRadius: 12, overflow: 'hidden', borderWidth: 1, borderColor: theme.colors.border, backgroundColor: theme.colors.card },
  bottom: { flex: 1 },
  muted: { color: theme.colors.muted },
  errorTitle: { color: theme.colors.danger, fontWeight: '900', fontSize: 18 },
  errorBody: { color: theme.colors.muted, marginTop: 8 },
  card: { marginTop: 12, backgroundColor: theme.colors.card, borderRadius: 12, padding: 12, borderWidth: 1, borderColor: theme.colors.border },
  cardTitle: { color: theme.colors.text, fontWeight: '900', fontSize: 16 },
  cardBody: { color: theme.colors.muted, marginTop: 8, lineHeight: 18 },
});


