"""
Pips Puzzle Agent

A Claude Agent SDK application for analyzing and solving NYT Pips puzzles from screenshots.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock
)

# Import custom tools
from tools.extract_puzzle import extract_puzzle_from_screenshot
from tools.ocr_constraints import ocr_constraints_from_screenshot
from tools.generate_spec import generate_puzzle_spec
from tools.solve_puzzle import solve_puzzle
from tools.provide_hints import provide_hints


# Load environment variables
load_dotenv()

# System prompt for the agent
SYSTEM_PROMPT = """You are an expert Pips puzzle assistant. You help users solve NYT Pips puzzles from screenshots.

Pips Puzzle Rules:
- Grid filled with dominoes (two adjacent cells)
- Each domino has pip values (0-6 typically)
- Regions have constraints (sum, all equal, comparison)
- Each domino used at most once

Your Tools:
1. extract_puzzle_from_screenshot: Analyze screenshot to detect grid/regions
2. ocr_constraints_from_screenshot: Try to read constraint text from image
3. generate_puzzle_spec: Create puzzle specification from extracted data
4. solve_puzzle: Solve puzzle completely using CSP solver
5. provide_hints: Give strategic hints without full solution

Workflow:
1. User provides screenshot → extract puzzle structure using Tool 1
2. Try OCR to detect constraints using Tool 2:
   - If OCR finds constraints with high confidence → use them
   - If OCR is uncertain or finds nothing → ask user to provide constraints
3. Ask user for domino tiles if not visible in tray
4. Generate puzzle specification using Tool 3
5. ALWAYS ASK USER: "Would you like me to solve it completely or provide hints?"
6. Based on response:
   - Solve: Run Tool 4, show solution with validation
   - Hints: Run Tool 5, suggest starting regions and explain strategies

IMPORTANT: Never solve the puzzle automatically - always ask first!

Be encouraging and educational when providing hints!
"""


async def main():
    """Main application entry point."""
    print("🎯 Pips Puzzle Agent")
    print("=" * 50)
    print()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please create a .env file with your API key:")
        print("ANTHROPIC_API_KEY=your_api_key_here")
        return

    # Create SDK MCP server with custom tools
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

    print("Starting Claude agent with Pips puzzle solving capabilities...")
    print("Type 'exit' to quit, 'help' for usage instructions")
    print()

    # Start interactive session
    async with ClaudeSDKClient(options=options) as client:
        # Show help message
        print("💡 Quick Start:")
        print("1. Provide a path to a Pips puzzle screenshot")
        print("2. I'll extract the puzzle structure")
        print("3. I'll try to read constraints via OCR")
        print("4. You can choose to solve it or get hints")
        print()

        turn = 0
        while True:
            # Get user input
            user_input = input(f"[Turn {turn + 1}] You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break

            if user_input.lower() == 'help':
                print("\n" + "=" * 50)
                print("📖 Help")
                print("=" * 50)
                print("\nTo analyze a puzzle screenshot:")
                print('  "Analyze IMG_2050.png"')
                print('  "Help me solve this puzzle: C:\\path\\to\\screenshot.png"')
                print("\nTo get hints:")
                print('  "Give me hints for this puzzle"')
                print("\nTo solve completely:")
                print('  "Solve the puzzle"')
                print("\nCommands:")
                print("  exit - Quit the application")
                print("  help - Show this help message")
                print("=" * 50 + "\n")
                continue

            # Send message to Claude
            await client.query(user_input)
            turn += 1

            # Process response
            print(f"[Turn {turn}] Claude: ", end="", flush=True)

            response_text = []
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text
                            print(text, end="", flush=True)
                            response_text.append(text)

            print()  # Newline after response


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
