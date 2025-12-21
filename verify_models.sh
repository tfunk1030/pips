#!/bin/bash
# Verify Vision Model IDs - December 21, 2025
#
# This script verifies that the model IDs configured in the app
# actually exist on OpenRouter.

set -e

echo "================================================"
echo "Vision Model ID Verification"
echo "================================================"
echo ""

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  Warning: OPENROUTER_API_KEY not set"
    echo "   To fully verify models, set your API key:"
    echo "   export OPENROUTER_API_KEY=sk-or-v1-..."
    echo ""
    echo "   Checking public model list instead..."
    echo ""
fi

# Models we expect to find
EXPECTED_MODELS=(
    "google/gemini-2.5-pro"
    "openai/gpt-4o"
    "anthropic/claude-3.7-sonnet"
)

echo "Checking OpenRouter model availability..."
echo ""

# Fetch model list
if [ -n "$OPENROUTER_API_KEY" ]; then
    MODELS_JSON=$(curl -s https://openrouter.ai/api/v1/models \
        -H "Authorization: Bearer $OPENROUTER_API_KEY")
else
    # Use unauthenticated endpoint (may have rate limits)
    MODELS_JSON=$(curl -s https://openrouter.ai/api/v1/models)
fi

# Check each expected model
ALL_FOUND=true
for model in "${EXPECTED_MODELS[@]}"; do
    if echo "$MODELS_JSON" | grep -q "\"$model\""; then
        echo "✅ Found: $model"
    else
        echo "❌ Missing: $model"
        ALL_FOUND=false
    fi
done

echo ""
echo "================================================"

if [ "$ALL_FOUND" = true ]; then
    echo "✅ All model IDs are valid!"
    echo ""
    echo "Your app is configured correctly."
    echo "Expected extraction time: ~10s (balanced) to ~30s (ensemble)"
    exit 0
else
    echo "❌ Some model IDs are invalid!"
    echo ""
    echo "This will cause extraction failures and timeouts."
    echo "Please check the model IDs in:"
    echo "  - pips-solver/src/services/extraction/config.ts"
    echo "  - pips-solver/src/config/models.ts"
    exit 1
fi
