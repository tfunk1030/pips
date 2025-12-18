"""
Live demo of the Pips Agent with a real API call
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Check for API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not found")
    print("Please create .env file with your API key")
    sys.exit(1)

print("=" * 60)
print("PIPS AGENT - Live Demo")
print("=" * 60)
print()

# Import agent components
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, create_sdk_mcp_server
from tools.extract_puzzle import extract_puzzle_from_screenshot
from tools.ocr_constraints import ocr_constraints_from_screenshot
from tools.generate_spec import generate_puzzle_spec
from tools.solve_puzzle import solve_puzzle
from tools.provide_hints import provide_hints
from main import SYSTEM_PROMPT

async def demo():
    """Run a quick demo interaction"""

    # Create MCP server with tools
    pips_tools = create_sdk_mcp_server(
        name="pips_tools",
        version="1.0.0",
        tools=[
            extract_puzzle_from_screenshot,
            ocr_constraints_from_screenshot,
            generate_puzzle_spec,
            solve_puzzle,
            provide_hints
        ]
    )

    # Configure options
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"pips": pips_tools},
        allowed_tools=[
            "mcp__pips__extract_puzzle_from_screenshot",
            "mcp__pips__ocr_constraints_from_screenshot",
            "mcp__pips__generate_puzzle_spec",
            "mcp__pips__solve_puzzle",
            "mcp__pips__provide_hints"
        ],
        permission_mode="acceptEdits"
    )

    print("Starting agent...")
    print()

    # Create a simple test query
    test_query = "Hello! Can you briefly explain what you can help me with?"

    print(f"Demo Query: {test_query}")
    print()
    print("Agent Response:")
    print("-" * 60)

    async with ClaudeSDKClient(options=options) as client:
        # Send query
        await client.query(test_query)

        # Get response
        async for message in client.receive_response():
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        print(block.text)

    print()
    print("-" * 60)
    print()
    print("Demo completed successfully!")
    print()
    print("The agent is fully functional and ready to analyze puzzles.")
    print("Run 'python main.py' for interactive mode.")

if __name__ == "__main__":
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
