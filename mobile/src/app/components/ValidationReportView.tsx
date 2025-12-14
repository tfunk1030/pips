import * as React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import type { SolutionReport } from '../../validator/validateSolution';
import { theme } from '../theme';

export function ValidationReportView({ report }: { report: SolutionReport }) {
  return (
    <View style={styles.card}>
      <Text style={styles.title}>
        Validation {report.ok ? <Text style={{ color: theme.colors.ok }}>OK</Text> : <Text style={{ color: theme.colors.danger }}>FAIL</Text>}
      </Text>

      {!report.ok ? (
        <View style={{ marginTop: 8, gap: 6 }}>
          {report.errors.slice(0, 8).map((e, i) => (
            <Text key={i} style={styles.err}>
              {e.path}: {e.message}
            </Text>
          ))}
        </View>
      ) : null}

      <Text style={styles.subTitle}>Regions</Text>
      <View style={{ gap: 6 }}>
        {report.regionReports.map((r) => (
          <Text key={r.regionId} style={[styles.row, !r.ok && { color: theme.colors.danger }]}>
            {r.regionId}: sum={r.sum} ({r.constraint})
          </Text>
        ))}
      </View>

      <Text style={styles.subTitle}>Domino usage</Text>
      <View style={{ gap: 6 }}>
        {report.dominoUsage
          .filter((d) => d.used > 0 || d.allowed === 0)
          .slice(0, 24)
          .map((d) => (
            <Text key={d.dominoKey} style={[styles.row, !d.ok && { color: theme.colors.danger }]}>
              {d.dominoKey}: used {d.used} / {d.allowed}
            </Text>
          ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  card: { marginTop: 12, backgroundColor: theme.colors.card, borderRadius: 12, padding: 12, borderWidth: 1, borderColor: theme.colors.border },
  title: { color: theme.colors.text, fontWeight: '900', fontSize: 16 },
  subTitle: { color: theme.colors.text, fontWeight: '900', marginTop: 12, marginBottom: 6 },
  row: { color: theme.colors.muted, fontSize: 12 },
  err: { color: theme.colors.danger, fontSize: 12, lineHeight: 16 },
});



