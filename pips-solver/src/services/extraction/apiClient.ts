/**
 * Multi-Model API Client
 *
 * Unified API client supporting OpenRouter and direct provider APIs
 * for Gemini 3 Pro, GPT-5.2, and Claude Opus 4.5.
 */

import {
  ApiEndpoint,
  ApiProvider,
  ExtractionConfig,
  VisionApiRequest,
  VisionApiResponse,
} from './types';

// =============================================================================
// API Endpoints
// =============================================================================

const API_ENDPOINTS: Record<ApiProvider, string> = {
  openrouter: 'https://openrouter.ai/api/v1/chat/completions',
  google: 'https://generativelanguage.googleapis.com/v1beta/models',
  openai: 'https://api.openai.com/v1/chat/completions',
  anthropic: 'https://api.anthropic.com/v1/messages',
};

// =============================================================================
// Route Resolution
// =============================================================================

/**
 * Determine which API endpoint to use for a given model
 */
export function resolveApiEndpoint(model: string, config: ExtractionConfig): ApiEndpoint {
  const { apiKeys } = config;

  // Check for model-specific direct key first
  if (model.startsWith('google/') && apiKeys.google) {
    return {
      provider: 'google',
      endpoint: API_ENDPOINTS.google,
      key: apiKeys.google,
      model: model.replace('google/', ''),
    };
  }

  if (model.startsWith('openai/') && apiKeys.openai) {
    return {
      provider: 'openai',
      endpoint: API_ENDPOINTS.openai,
      key: apiKeys.openai,
      model: model.replace('openai/', ''),
    };
  }

  if (model.startsWith('anthropic/') && apiKeys.anthropic) {
    return {
      provider: 'anthropic',
      endpoint: API_ENDPOINTS.anthropic,
      key: apiKeys.anthropic,
      model: model.replace('anthropic/', ''),
    };
  }

  // Fall back to OpenRouter
  if (apiKeys.openrouter) {
    return {
      provider: 'openrouter',
      endpoint: API_ENDPOINTS.openrouter,
      key: apiKeys.openrouter,
      model: model,
    };
  }

  throw new Error(`No API key configured for model: ${model}`);
}

// =============================================================================
// Provider-Specific Request Builders
// =============================================================================

function buildOpenRouterRequest(request: VisionApiRequest): object {
  return {
    model: request.model,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: {
              url: `data:image/jpeg;base64,${request.imageBase64}`,
            },
          },
          {
            type: 'text',
            text: request.prompt,
          },
        ],
      },
    ],
    max_tokens: request.maxTokens || 1024,
  };
}

function buildOpenAIRequest(request: VisionApiRequest): object {
  return {
    model: request.model,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image_url',
            image_url: {
              url: `data:image/jpeg;base64,${request.imageBase64}`,
            },
          },
          {
            type: 'text',
            text: request.prompt,
          },
        ],
      },
    ],
    max_tokens: request.maxTokens || 1024,
  };
}

function buildAnthropicRequest(request: VisionApiRequest): object {
  return {
    model: request.model,
    max_tokens: request.maxTokens || 1024,
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'image',
            source: {
              type: 'base64',
              media_type: 'image/jpeg',
              data: request.imageBase64,
            },
          },
          {
            type: 'text',
            text: request.prompt,
          },
        ],
      },
    ],
  };
}

function buildGoogleRequest(request: VisionApiRequest): object {
  return {
    contents: [
      {
        parts: [
          {
            inline_data: {
              mime_type: 'image/jpeg',
              data: request.imageBase64,
            },
          },
          {
            text: request.prompt,
          },
        ],
      },
    ],
    generationConfig: {
      maxOutputTokens: request.maxTokens || 1024,
    },
  };
}

// =============================================================================
// Response Parsers
// =============================================================================

function parseOpenRouterResponse(data: any): string {
  return data?.choices?.[0]?.message?.content || '';
}

function parseOpenAIResponse(data: any): string {
  return data?.choices?.[0]?.message?.content || '';
}

function parseAnthropicResponse(data: any): string {
  const content = data?.content?.[0];
  return content?.type === 'text' ? content.text : '';
}

function parseGoogleResponse(data: any): string {
  return data?.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

// =============================================================================
// Main API Call Function
// =============================================================================

/**
 * Make a vision API call to the specified model
 */
export async function callVisionApi(
  request: VisionApiRequest,
  config: ExtractionConfig
): Promise<VisionApiResponse> {
  const startTime = Date.now();
  const endpoint = resolveApiEndpoint(request.model, config);

  try {
    let url: string;
    let body: object;
    let headers: Record<string, string>;

    switch (endpoint.provider) {
      case 'openrouter':
        url = endpoint.endpoint;
        body = buildOpenRouterRequest({ ...request, model: endpoint.model });
        headers = {
          'Authorization': `Bearer ${endpoint.key}`,
          'Content-Type': 'application/json',
          'HTTP-Referer': 'https://pips-solver.app',
          'X-Title': 'Pips Solver',
        };
        break;

      case 'openai':
        url = endpoint.endpoint;
        body = buildOpenAIRequest({ ...request, model: endpoint.model });
        headers = {
          'Authorization': `Bearer ${endpoint.key}`,
          'Content-Type': 'application/json',
        };
        break;

      case 'anthropic':
        url = endpoint.endpoint;
        body = buildAnthropicRequest({ ...request, model: endpoint.model });
        headers = {
          'x-api-key': endpoint.key,
          'Content-Type': 'application/json',
          'anthropic-version': '2024-10-22',
        };
        break;

      case 'google':
        url = `${endpoint.endpoint}/${endpoint.model}:generateContent?key=${endpoint.key}`;
        body = buildGoogleRequest(request);
        headers = {
          'Content-Type': 'application/json',
        };
        break;

      default:
        throw new Error(`Unknown provider: ${endpoint.provider}`);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), config.timeoutMs);

    const response = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API error ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    const latencyMs = Date.now() - startTime;

    let content: string;
    switch (endpoint.provider) {
      case 'openrouter':
        content = parseOpenRouterResponse(data);
        break;
      case 'openai':
        content = parseOpenAIResponse(data);
        break;
      case 'anthropic':
        content = parseAnthropicResponse(data);
        break;
      case 'google':
        content = parseGoogleResponse(data);
        break;
      default:
        content = '';
    }

    return { content, latencyMs };
  } catch (error) {
    const latencyMs = Date.now() - startTime;
    const errorMessage = error instanceof Error ? error.message : String(error);

    // Handle timeout specifically
    if (errorMessage.includes('aborted')) {
      return {
        content: '',
        latencyMs,
        error: `Request timeout after ${config.timeoutMs}ms`,
      };
    }

    return {
      content: '',
      latencyMs,
      error: errorMessage,
    };
  }
}

// =============================================================================
// Parallel Calls
// =============================================================================

/**
 * Call all configured models in parallel for a stage
 */
export async function callAllModels(
  imageBase64: string,
  prompt: string,
  config: ExtractionConfig
): Promise<Map<string, VisionApiResponse>> {
  const { models, apiKeys } = config;
  const results = new Map<string, VisionApiResponse>();

  // Determine which models to call based on available keys
  const modelsToCall: string[] = [];

  if (apiKeys.openrouter) {
    // OpenRouter can access all models
    modelsToCall.push(models.gemini, models.gpt, models.claude);
  } else {
    if (apiKeys.google) modelsToCall.push(models.gemini);
    if (apiKeys.openai) modelsToCall.push(models.gpt);
    if (apiKeys.anthropic) modelsToCall.push(models.claude);
  }

  if (modelsToCall.length === 0) {
    throw new Error('No API keys configured. Please add an OpenRouter key or individual provider keys in Settings.');
  }

  // Call all models in parallel
  const promises = modelsToCall.map(async (model) => {
    const response = await callVisionApi(
      { model, imageBase64, prompt },
      config
    );
    return { model, response };
  });

  const responses = await Promise.all(promises);

  for (const { model, response } of responses) {
    results.set(model, response);
  }

  return results;
}

// =============================================================================
// Retry with Exponential Backoff
// =============================================================================

/**
 * Call API with retry on rate limit errors
 */
export async function callWithRetry(
  request: VisionApiRequest,
  config: ExtractionConfig,
  maxRetries: number = 3
): Promise<VisionApiResponse> {
  let lastError: string | undefined;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await callVisionApi(request, config);

    // Success
    if (!response.error) {
      return response;
    }

    // Check if retryable (rate limit, server error)
    const isRetryable =
      response.error.includes('429') ||
      response.error.includes('rate limit') ||
      response.error.includes('500') ||
      response.error.includes('502') ||
      response.error.includes('503');

    if (!isRetryable) {
      return response;
    }

    lastError = response.error;

    // Exponential backoff: 1s, 2s, 4s
    const backoffMs = Math.pow(2, attempt) * 1000;
    await new Promise((resolve) => setTimeout(resolve, backoffMs));
  }

  return {
    content: '',
    latencyMs: 0,
    error: `Max retries exceeded. Last error: ${lastError}`,
  };
}
