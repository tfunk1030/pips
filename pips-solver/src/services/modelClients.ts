/**
 * Multi-Model API Clients
 * Unified interface for calling Claude, Gemini, and GPT vision APIs
 */

import { Platform } from 'react-native';
import { MODELS, ModelConfig, ModelProvider } from '../config/models';

// ════════════════════════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════════════════════════

export interface VisionMessage {
  role: 'user' | 'assistant' | 'system';
  content: VisionContent[];
}

export type VisionContent =
  | { type: 'text'; text: string }
  | { type: 'image'; base64: string; mediaType: string };

export interface VisionResponse {
  text: string;
  model: string;
  provider: ModelProvider;
  latencyMs: number;
  tokensUsed?: {
    input: number;
    output: number;
  };
}

export interface ModelClientOptions {
  temperature?: number;
  maxTokens?: number;
  jsonMode?: boolean;
}

// ════════════════════════════════════════════════════════════════════════════
// Claude (Anthropic) Client
// ════════════════════════════════════════════════════════════════════════════

async function callClaude(
  apiKey: string,
  model: ModelConfig,
  messages: VisionMessage[],
  options: ModelClientOptions = {}
): Promise<VisionResponse> {
  const startTime = Date.now();

  // Convert to Claude message format
  const claudeMessages = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role as 'user' | 'assistant',
      content: m.content.map(c => {
        if (c.type === 'text') {
          return { type: 'text' as const, text: c.text };
        } else {
          return {
            type: 'image' as const,
            source: {
              type: 'base64' as const,
              media_type: c.mediaType,
              data: c.base64,
            },
          };
        }
      }),
    }));

  // Extract system message if present
  const systemMessage = messages.find(m => m.role === 'system');
  const systemText = systemMessage?.content
    .filter(c => c.type === 'text')
    .map(c => (c as { type: 'text'; text: string }).text)
    .join('\n');

  // For Expo web, Anthropic requires this header
  const extraHeaders: Record<string, string> =
    Platform.OS === 'web' ? { 'anthropic-dangerous-direct-browser-access': 'true' } : {};

  const requestBody: Record<string, unknown> = {
    model: model.id,
    max_tokens: options.maxTokens ?? 4096,
    messages: claudeMessages,
  };

  if (systemText) {
    requestBody.system = systemText;
  }

  if (options.temperature !== undefined) {
    requestBody.temperature = options.temperature;
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      ...extraHeaders,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Claude API error (${response.status}): ${error}`);
  }

  const data = await response.json();
  const latencyMs = Date.now() - startTime;

  return {
    text: data.content.map((b: { text: string }) => b.text).join('\n'),
    model: model.id,
    provider: 'anthropic',
    latencyMs,
    tokensUsed: {
      input: data.usage?.input_tokens ?? 0,
      output: data.usage?.output_tokens ?? 0,
    },
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Gemini (Google) Client
// ════════════════════════════════════════════════════════════════════════════

async function callGemini(
  apiKey: string,
  model: ModelConfig,
  messages: VisionMessage[],
  options: ModelClientOptions = {}
): Promise<VisionResponse> {
  const startTime = Date.now();

  // Convert to Gemini format
  // Gemini uses "parts" within "contents"
  const contents = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: m.content.map(c => {
        if (c.type === 'text') {
          return { text: c.text };
        } else {
          return {
            inline_data: {
              mime_type: c.mediaType,
              data: c.base64,
            },
          };
        }
      }),
    }));

  // Extract system instruction
  const systemMessage = messages.find(m => m.role === 'system');
  const systemInstruction = systemMessage
    ? {
        parts: systemMessage.content
          .filter(c => c.type === 'text')
          .map(c => ({ text: (c as { type: 'text'; text: string }).text })),
      }
    : undefined;

  const requestBody: Record<string, unknown> = {
    contents,
    generationConfig: {
      maxOutputTokens: options.maxTokens ?? 4096,
      temperature: options.temperature ?? 0.1,
    },
  };

  // Note: thinkingConfig/thinkingBudget are not currently supported in the
  // generateContent REST API. They may be available in future SDK versions.
  // For now, we rely on the model's default behavior.

  if (systemInstruction) {
    requestBody.systemInstruction = systemInstruction;
  }

  // Enable JSON mode if requested
  if (options.jsonMode) {
    (requestBody.generationConfig as Record<string, unknown>).responseMimeType = 'application/json';
  }

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model.id}:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Gemini API error (${response.status}): ${error}`);
  }

  const data = await response.json();
  const latencyMs = Date.now() - startTime;

  // Check for truncation due to max tokens
  const finishReason = data.candidates?.[0]?.finishReason;
  if (finishReason === 'MAX_TOKENS') {
    console.warn(`[Gemini] Response truncated due to MAX_TOKENS limit`);
  }

  // Extract text from response
  const text =
    data.candidates?.[0]?.content?.parts?.map((p: { text?: string }) => p.text || '').join('') ??
    '';

  return {
    text,
    model: model.id,
    provider: 'google',
    latencyMs,
    tokensUsed: {
      input: data.usageMetadata?.promptTokenCount ?? 0,
      output: data.usageMetadata?.candidatesTokenCount ?? 0,
    },
  };
}

// ════════════════════════════════════════════════════════════════════════════
// GPT-4o (OpenAI) Client
// ════════════════════════════════════════════════════════════════════════════

async function callOpenAI(
  apiKey: string,
  model: ModelConfig,
  messages: VisionMessage[],
  options: ModelClientOptions = {}
): Promise<VisionResponse> {
  const startTime = Date.now();

  // Convert to OpenAI format
  const openaiMessages = messages.map(m => ({
    role: m.role,
    content: m.content.map(c => {
      if (c.type === 'text') {
        return { type: 'text' as const, text: c.text };
      } else {
        return {
          type: 'image_url' as const,
          image_url: {
            url: `data:${c.mediaType};base64,${c.base64}`,
            detail: 'high' as const,
          },
        };
      }
    }),
  }));

  const requestBody: Record<string, unknown> = {
    model: model.id,
    messages: openaiMessages,
    max_tokens: options.maxTokens ?? 4096,
  };

  if (options.temperature !== undefined) {
    requestBody.temperature = options.temperature;
  }

  if (options.jsonMode) {
    requestBody.response_format = { type: 'json_object' };
  }

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error (${response.status}): ${error}`);
  }

  const data = await response.json();
  const latencyMs = Date.now() - startTime;

  return {
    text: data.choices?.[0]?.message?.content ?? '',
    model: model.id,
    provider: 'openai',
    latencyMs,
    tokensUsed: {
      input: data.usage?.prompt_tokens ?? 0,
      output: data.usage?.completion_tokens ?? 0,
    },
  };
}

// ════════════════════════════════════════════════════════════════════════════
// Unified Client Interface
// ════════════════════════════════════════════════════════════════════════════

export interface APIKeys {
  anthropic?: string;
  google?: string;
  openai?: string;
}

/**
 * Call a vision model with the unified interface
 */
export async function callVisionModel(
  keys: APIKeys,
  modelKey: keyof typeof MODELS,
  messages: VisionMessage[],
  options: ModelClientOptions = {}
): Promise<VisionResponse> {
  const model = MODELS[modelKey];
  if (!model) {
    throw new Error(`Unknown model: ${modelKey}`);
  }

  switch (model.provider) {
    case 'anthropic': {
      const apiKey = keys.anthropic?.trim();
      if (!apiKey) throw new Error('Anthropic API key not provided');
      return callClaude(apiKey, model, messages, options);
    }
    case 'google': {
      const apiKey = keys.google?.trim();
      if (!apiKey) throw new Error('Google API key not provided');
      return callGemini(apiKey, model, messages, options);
    }
    case 'openai': {
      const apiKey = keys.openai?.trim();
      if (!apiKey) throw new Error('OpenAI API key not provided');
      return callOpenAI(apiKey, model, messages, options);
    }
    default:
      throw new Error(`Unsupported provider: ${model.provider}`);
  }
}

/**
 * Call multiple models in parallel and return all results
 */
export async function callMultipleModels(
  keys: APIKeys,
  modelKeys: (keyof typeof MODELS)[],
  messages: VisionMessage[],
  options: ModelClientOptions = {}
): Promise<{ modelKey: string; result: VisionResponse | null; error: string | null }[]> {
  const promises = modelKeys.map(async modelKey => {
    try {
      const result = await callVisionModel(keys, modelKey, messages, options);
      return { modelKey, result, error: null };
    } catch (e) {
      return {
        modelKey,
        result: null,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  });

  return Promise.all(promises);
}

// ════════════════════════════════════════════════════════════════════════════
// Image Utilities
// ════════════════════════════════════════════════════════════════════════════

/**
 * Normalize base64 image data (strip data URL prefix, remove whitespace)
 */
export function normalizeBase64(input: string): string {
  let s = input.trim();

  // Handle "data:image/...;base64,xxxx" strings
  if (s.startsWith('data:')) {
    const idx = s.indexOf('base64,');
    if (idx >= 0) {
      s = s.slice(idx + 'base64,'.length);
    }
  }

  // Remove whitespace
  s = s.replace(/\s+/g, '');

  return s;
}

/**
 * Infer media type from base64 image data
 */
export function inferMediaType(base64: string): string {
  const b64 = normalizeBase64(base64);

  // Check magic bytes
  if (typeof atob !== 'undefined') {
    try {
      let sample = b64.substring(0, 16);
      while (sample.length % 4 !== 0) sample += '=';
      const firstBytes = atob(sample);
      const bytes = new Uint8Array(firstBytes.length);
      for (let i = 0; i < firstBytes.length; i++) {
        bytes[i] = firstBytes.charCodeAt(i);
      }

      // PNG
      if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) {
        return 'image/png';
      }
      // JPEG
      if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
        return 'image/jpeg';
      }
      // WebP
      if (
        bytes[0] === 0x52 &&
        bytes[1] === 0x49 &&
        bytes[2] === 0x46 &&
        bytes[3] === 0x46 &&
        bytes.length >= 12 &&
        bytes[8] === 0x57 &&
        bytes[9] === 0x45 &&
        bytes[10] === 0x42 &&
        bytes[11] === 0x50
      ) {
        return 'image/webp';
      }
    } catch {
      // Fall through to string detection
    }
  }

  // String-based fallback
  if (b64.startsWith('iVBORw0KGgo')) return 'image/png';
  if (b64.startsWith('/9j/')) return 'image/jpeg';
  if (b64.startsWith('UklGR')) return 'image/webp';

  return 'image/jpeg'; // Default
}

/**
 * Create a vision message with image
 */
export function createImageMessage(
  base64Image: string,
  prompt: string,
  systemPrompt?: string
): VisionMessage[] {
  const normalized = normalizeBase64(base64Image);
  const mediaType = inferMediaType(base64Image);

  const messages: VisionMessage[] = [];

  if (systemPrompt) {
    messages.push({
      role: 'system',
      content: [{ type: 'text', text: systemPrompt }],
    });
  }

  messages.push({
    role: 'user',
    content: [
      { type: 'image', base64: normalized, mediaType },
      { type: 'text', text: prompt },
    ],
  });

  return messages;
}
