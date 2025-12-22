/**
 * Shared Zod Schemas for AI Extraction
 * Used by both aiExtraction.ts and ensembleExtraction.ts
 */

import { z } from 'zod';

// ════════════════════════════════════════════════════════════════════════════
// Constraint Schema
// ════════════════════════════════════════════════════════════════════════════

/**
 * Lenient constraint schema - normalizes variations in model output
 * Handles: sum, total, all_equal, all_different, etc.
 */
export const ConstraintSchema = z.object({
  // Accept various type formats that models might return
  type: z.string().transform(t => {
    const normalized = t.toLowerCase().replace(/[_\s]/g, '');
    if (normalized === 'sum' || normalized === 'total') return 'sum';
    if (normalized.includes('equal') && !normalized.includes('diff')) return 'all_equal';
    if (normalized.includes('diff') || normalized.includes('unique')) return 'all_different';
    return t; // Keep original if unknown - will be handled downstream
  }),
  op: z
    .enum(['==', '<', '>', '!='])
    .optional()
    .nullable()
    .transform(v => v ?? undefined),
  value: z
    .number()
    .optional()
    .nullable()
    .transform(v => v ?? undefined),
});

export type ConstraintSchemaType = z.infer<typeof ConstraintSchema>;

// ════════════════════════════════════════════════════════════════════════════
// Confidence Schema
// ════════════════════════════════════════════════════════════════════════════

export const ConfidenceScoresSchema = z
  .object({
    grid: z.number().min(0).max(1),
    regions: z.number().min(0).max(1),
    constraints: z.number().min(0).max(1),
  })
  .optional();

export type ConfidenceScoresType = z.infer<typeof ConfidenceScoresSchema>;

// ════════════════════════════════════════════════════════════════════════════
// Board Extraction Schema
// ════════════════════════════════════════════════════════════════════════════

export const BoardExtractionSchema = z.object({
  rows: z.number().min(2).max(12),
  cols: z.number().min(2).max(12),
  gridLocation: z
    .object({
      left: z.number(),
      top: z.number(),
      right: z.number(),
      bottom: z.number(),
      imageWidth: z.number(),
      imageHeight: z.number(),
    })
    .optional(),
  shape: z.string().min(1),
  regions: z.string().min(1),
  constraints: z.record(z.string(), ConstraintSchema).optional(),
  confidence: ConfidenceScoresSchema,
  reasoning: z.string().optional(),
});

export type BoardExtractionSchemaType = z.infer<typeof BoardExtractionSchema>;

// ════════════════════════════════════════════════════════════════════════════
// Domino Extraction Schema
// ════════════════════════════════════════════════════════════════════════════

export const DominoExtractionSchema = z.object({
  dominoes: z.array(z.tuple([z.number().min(0).max(6), z.number().min(0).max(6)])),
  confidence: z.number().min(0).max(1).optional(),
  reasoning: z.string().optional(),
});

export type DominoExtractionSchemaType = z.infer<typeof DominoExtractionSchema>;

// ════════════════════════════════════════════════════════════════════════════
// Verification Schema (used by ensemble extraction)
// ════════════════════════════════════════════════════════════════════════════

export const VerificationSchema = z.object({
  verified: z.boolean(),
  issues: z.array(z.string()).optional(),
  corrections: z
    .object({
      rows: z.number().optional(),
      cols: z.number().optional(),
      shape: z.string().optional(),
      regions: z.string().optional(),
      constraints: z.record(z.string(), ConstraintSchema).optional(),
      dominoes: z.array(z.tuple([z.number(), z.number()])).optional(),
    })
    .optional()
    .nullable(),
});

export type VerificationSchemaType = z.infer<typeof VerificationSchema>;
