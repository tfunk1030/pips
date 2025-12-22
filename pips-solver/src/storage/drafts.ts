/**
 * Draft storage for OverlayBuilder workflow
 * Saves work-in-progress puzzles for recovery
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { OverlayBuilderState, DraftMeta } from '../model/overlayTypes';

const DRAFTS_KEY = '@pips_drafts';
const DRAFT_PREFIX = '@pips_draft_';
const MAX_DRAFT_AGE_HOURS = 24;

/**
 * Save a draft to storage
 */
export async function saveDraft(state: OverlayBuilderState): Promise<void> {
  try {
    const key = `${DRAFT_PREFIX}${state.draftId}`;
    const updatedState: OverlayBuilderState = {
      ...state,
      draftUpdatedAt: Date.now(),
    };
    await AsyncStorage.setItem(key, JSON.stringify(updatedState));
    await updateDraftIndex(state.draftId, updatedState);
  } catch (error) {
    console.error('Failed to save draft:', error);
  }
}

/**
 * Load a draft by ID
 */
export async function loadDraft(draftId: string): Promise<OverlayBuilderState | null> {
  try {
    const key = `${DRAFT_PREFIX}${draftId}`;
    const data = await AsyncStorage.getItem(key);
    if (data) {
      return JSON.parse(data) as OverlayBuilderState;
    }
    return null;
  } catch (error) {
    console.error('Failed to load draft:', error);
    return null;
  }
}

/**
 * List all draft metadata (for recovery UI)
 */
export async function listDrafts(): Promise<DraftMeta[]> {
  try {
    const indexData = await AsyncStorage.getItem(DRAFTS_KEY);
    if (!indexData) {
      return [];
    }
    const index: DraftMeta[] = JSON.parse(indexData);

    // Filter out expired drafts
    const now = Date.now();
    const maxAge = MAX_DRAFT_AGE_HOURS * 60 * 60 * 1000;
    const validDrafts = index.filter(d => now - d.updatedAt < maxAge);

    // Sort by most recent first
    validDrafts.sort((a, b) => b.updatedAt - a.updatedAt);

    return validDrafts;
  } catch (error) {
    console.error('Failed to list drafts:', error);
    return [];
  }
}

/**
 * Delete a draft
 */
export async function deleteDraft(draftId: string): Promise<void> {
  try {
    const key = `${DRAFT_PREFIX}${draftId}`;
    await AsyncStorage.removeItem(key);
    await removeDraftFromIndex(draftId);
  } catch (error) {
    console.error('Failed to delete draft:', error);
  }
}

/**
 * Clean up expired drafts (older than MAX_DRAFT_AGE_HOURS)
 */
export async function cleanExpiredDrafts(): Promise<number> {
  try {
    const drafts = await listDrafts();
    const now = Date.now();
    const maxAge = MAX_DRAFT_AGE_HOURS * 60 * 60 * 1000;

    // Get all draft keys
    const indexData = await AsyncStorage.getItem(DRAFTS_KEY);
    if (!indexData) return 0;

    const index: DraftMeta[] = JSON.parse(indexData);
    const expiredDrafts = index.filter(d => now - d.updatedAt >= maxAge);

    // Delete expired drafts
    for (const draft of expiredDrafts) {
      await AsyncStorage.removeItem(`${DRAFT_PREFIX}${draft.draftId}`);
    }

    // Update index to only keep valid drafts
    const validDrafts = index.filter(d => now - d.updatedAt < maxAge);
    await AsyncStorage.setItem(DRAFTS_KEY, JSON.stringify(validDrafts));

    return expiredDrafts.length;
  } catch (error) {
    console.error('Failed to clean expired drafts:', error);
    return 0;
  }
}

/**
 * Check if there are any recent drafts
 */
export async function hasRecentDraft(): Promise<boolean> {
  const drafts = await listDrafts();
  return drafts.length > 0;
}

/**
 * Get the most recent draft
 */
export async function getMostRecentDraft(): Promise<OverlayBuilderState | null> {
  const drafts = await listDrafts();
  if (drafts.length === 0) {
    return null;
  }
  return loadDraft(drafts[0].draftId);
}

// ════════════════════════════════════════════════════════════════════════════
// Internal helpers
// ════════════════════════════════════════════════════════════════════════════

async function updateDraftIndex(draftId: string, state: OverlayBuilderState): Promise<void> {
  try {
    const indexData = await AsyncStorage.getItem(DRAFTS_KEY);
    let index: DraftMeta[] = indexData ? JSON.parse(indexData) : [];

    // Remove existing entry for this draft
    index = index.filter(d => d.draftId !== draftId);

    // Add/update the draft meta
    const meta: DraftMeta = {
      draftId,
      imageUri: state.image?.uri || '',
      step: state.step,
      updatedAt: state.draftUpdatedAt,
      rows: state.grid.rows,
      cols: state.grid.cols,
    };
    index.push(meta);

    await AsyncStorage.setItem(DRAFTS_KEY, JSON.stringify(index));
  } catch (error) {
    console.error('Failed to update draft index:', error);
  }
}

async function removeDraftFromIndex(draftId: string): Promise<void> {
  try {
    const indexData = await AsyncStorage.getItem(DRAFTS_KEY);
    if (!indexData) return;

    let index: DraftMeta[] = JSON.parse(indexData);
    index = index.filter(d => d.draftId !== draftId);
    await AsyncStorage.setItem(DRAFTS_KEY, JSON.stringify(index));
  } catch (error) {
    console.error('Failed to remove draft from index:', error);
  }
}
