#!/usr/bin/env python3
"""
Test client for the Surgical Command Agent API

Usage:
    python test_agent.py                    # Interactive mode
    python test_agent.py "more light"       # Single command
    python test_agent.py --url http://host:8080
"""

import argparse
import json
import sys
import requests

DEFAULT_URL = "http://localhost:8080"


def send_command(url: str, text: str, session_id: str = "test") -> dict:
    """Send a command to the agent API"""
    response = requests.post(
        f"{url}/command",
        json={
            "text": text,
            "session_id": session_id
        }
    )
    response.raise_for_status()
    return response.json()


def print_response(response: dict):
    """Pretty print the response"""
    print("\n" + "=" * 50)

    if response.get("tool_calls"):
        print("TOOL CALLS:")
        for tc in response["tool_calls"]:
            print(f"\n  {tc['name']}:")
            for k, v in tc["arguments"].items():
                print(f"    {k}: {v}")

    if response.get("message"):
        print(f"\nMESSAGE: {response['message']}")

    if response.get("needs_confirmation"):
        print("\n⚠️  CONFIRMATION REQUIRED")

    print(f"\n[{response.get('processing_time_ms', 0):.0f}ms]")
    print("=" * 50)


def interactive_mode(url: str):
    """Interactive command loop"""
    print("\n" + "=" * 50)
    print("Surgical Command Agent - Interactive Test")
    print("=" * 50)
    print(f"Connected to: {url}")
    print("Type commands or 'quit' to exit")
    print("=" * 50 + "\n")

    session_id = "interactive"

    # Check health
    try:
        health = requests.get(f"{url}/health").json()
        print(f"Status: {health['status']}")
        print(f"Model: {health['model']}")
        print(f"Tools: {health['tools_available']}")
        print()
    except Exception as e:
        print(f"⚠️  Could not connect to agent: {e}")
        print("Make sure vLLM and agent services are running:")
        print("  ./run_llm.sh  (in one terminal)")
        print("  ./run_agent.sh  (in another terminal)")
        return

    while True:
        try:
            text = input("Command> ").strip()

            if not text:
                continue

            if text.lower() in ("quit", "exit", "q"):
                break

            if text.lower() == "tools":
                tools = requests.get(f"{url}/tools").json()
                print("\nAvailable tools:")
                for t in tools["tools"]:
                    print(f"  - {t['function']['name']}: {t['function']['description'][:60]}...")
                print()
                continue

            if text.lower() == "clear":
                requests.delete(f"{url}/session/{session_id}")
                print("Session cleared.\n")
                continue

            response = send_command(url, text, session_id)
            print_response(response)

        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Test the Surgical Command Agent")
    parser.add_argument("command", nargs="?", help="Command to send (interactive mode if omitted)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Agent API URL (default: {DEFAULT_URL})")
    parser.add_argument("--session", default="test", help="Session ID")
    args = parser.parse_args()

    if args.command:
        # Single command mode
        try:
            response = send_command(args.url, args.command, args.session)
            print_response(response)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_mode(args.url)


if __name__ == "__main__":
    main()
