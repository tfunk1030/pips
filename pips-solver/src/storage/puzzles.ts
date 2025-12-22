/**
 * Puzzle storage using AsyncStorage
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { specToYAML } from '../model/parser';
import { PuzzleSpec, Solution, StoredPuzzle } from '../model/types';

const PUZZLES_KEY = '@pips_puzzles';
const SETTINGS_KEY = '@pips_settings';

/**
 * Extraction strategy for AI puzzle extraction
 * - 'fast': Single model (Gemini Flash), ~3s
 * - 'balanced': Gemini Pro with verification, ~20s
 * - 'accurate': Gemini Pro + Claude verification, ~35s
 * - 'ensemble': Multi-model consensus, ~45s (MAXIMUM ACCURACY)
 */
export type ExtractionStrategy = 'fast' | 'balanced' | 'accurate' | 'ensemble';

/**
 * App settings interface
 */
export interface AppSettings {
  defaultMaxPip: number;
  defaultAllowDuplicates: boolean;
  defaultFindAll: boolean;
  defaultDebugLevel: number;
  maxIterationsPerTick: number;

  // API Keys for multi-model extraction
  anthropicApiKey?: string; // Claude (best for structured JSON)
  googleApiKey?: string; // Gemini (best for grid/object detection)
  openaiApiKey?: string; // GPT-4o (fallback)

  // Extraction configuration
  extractionStrategy?: ExtractionStrategy;

  // Debug mode for extraction comparison
  // When enabled, stores raw responses from each AI model for side-by-side comparison
  // Default: false (off for performance - storing responses uses more memory)
  saveDebugResponses?: boolean;

  // CV Service URL for hybrid extraction (default: http://localhost:8080)
  cvServiceUrl?: string;
}

/**
 * Get all stored puzzles
 */
export async function getAllPuzzles(): Promise<StoredPuzzle[]> {
  try {
    const data = await AsyncStorage.getItem(PUZZLES_KEY);
    if (!data) {
      return [];
    }
    return JSON.parse(data);
  } catch (error) {
    console.error('Error loading puzzles:', error);
    return [];
  }
}

/**
 * Get a specific puzzle by ID
 */
export async function getPuzzle(id: string): Promise<StoredPuzzle | null> {
  const puzzles = await getAllPuzzles();
  return puzzles.find(p => p.id === id) || null;
}

/**
 * Save a puzzle
 */
export async function savePuzzle(
  spec: PuzzleSpec,
  yaml: string,
  solution?: Solution
): Promise<StoredPuzzle> {
  const puzzles = await getAllPuzzles();

  const now = Date.now();
  const existing = puzzles.find(p => p.id === spec.id);

  const puzzle: StoredPuzzle = {
    id: spec.id!,
    name: spec.name || 'Untitled',
    yaml,
    spec,
    createdAt: existing?.createdAt || now,
    updatedAt: now,
    solved: solution !== undefined,
    solution,
  };

  if (existing) {
    // Update existing
    const index = puzzles.indexOf(existing);
    puzzles[index] = puzzle;
  } else {
    // Add new
    puzzles.push(puzzle);
  }

  await AsyncStorage.setItem(PUZZLES_KEY, JSON.stringify(puzzles));
  return puzzle;
}

/**
 * Delete a puzzle
 */
export async function deletePuzzle(id: string): Promise<void> {
  const puzzles = await getAllPuzzles();
  const filtered = puzzles.filter(p => p.id !== id);
  await AsyncStorage.setItem(PUZZLES_KEY, JSON.stringify(filtered));
}

/**
 * Update puzzle solution
 */
export async function updatePuzzleSolution(id: string, solution: Solution): Promise<void> {
  const puzzles = await getAllPuzzles();
  const puzzle = puzzles.find(p => p.id === id);

  if (puzzle) {
    puzzle.solution = solution;
    puzzle.solved = true;
    puzzle.updatedAt = Date.now();
    await AsyncStorage.setItem(PUZZLES_KEY, JSON.stringify(puzzles));
  }
}

/**
 * Import a puzzle from YAML string
 */
export async function importPuzzle(yaml: string, spec: PuzzleSpec): Promise<StoredPuzzle> {
  return savePuzzle(spec, yaml);
}

/**
 * Export a puzzle to YAML string
 */
export function exportPuzzle(puzzle: StoredPuzzle): string {
  return puzzle.yaml || specToYAML(puzzle.spec);
}

/**
 * Clear all puzzles
 */
export async function clearAllPuzzles(): Promise<void> {
  await AsyncStorage.removeItem(PUZZLES_KEY);
}

/**
 * Get app settings
 */
export async function getSettings(): Promise<AppSettings> {
  try {
    const data = await AsyncStorage.getItem(SETTINGS_KEY);
    if (!data) {
      return getDefaultSettings();
    }
    return { ...getDefaultSettings(), ...JSON.parse(data) };
  } catch (error) {
    console.error('Error loading settings:', error);
    return getDefaultSettings();
  }
}

/**
 * Save app settings
 */
export async function saveSettings(settings: AppSettings): Promise<void> {
  await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

/**
 * Get default settings
 */
function getDefaultSettings(): AppSettings {
  return {
    defaultMaxPip: 6,
    defaultAllowDuplicates: false,
    defaultFindAll: false,
    defaultDebugLevel: 0,
    maxIterationsPerTick: 100,
    extractionStrategy: 'accurate', // Default to accurate mode
  };
}
