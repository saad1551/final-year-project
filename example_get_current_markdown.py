#!/usr/bin/env python3
"""
Example script showing how to get the markdown of the currently open page
in the Playwright server.

This demonstrates both methods:
1. Using an existing session ID
2. Creating a new session and navigating to a URL
"""

from utils import get_current_page_markdown, convert_url_to_markdown


def example_with_existing_session():
    """Example using an existing session ID."""
    print("=== Example with Existing Session ===")
    
    # You would replace this with an actual session ID from your running server
    session_id = "your-session-id-here"
    
    try:
        markdown = get_current_page_markdown(session_id)
        print("Current page markdown:")
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a valid session ID and the server is running.")


def example_with_new_session():
    """Example creating a new session and navigating to a URL."""
    print("\n=== Example with New Session ===")
    
    url = "https://example.com"
    
    try:
        markdown = convert_url_to_markdown(url)
        print(f"Markdown for {url}:")
        print(markdown[:500] + "..." if len(markdown) > 500 else markdown)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the Playwright server is running on localhost:3000")


def main():
    """Run both examples."""
    print("Getting markdown from Playwright server...")
    print("Make sure the server is running: node javascript/server/src/index.js 3000")
    print()
    
    # Example 1: With existing session (commented out since we don't have a real session ID)
    # example_with_existing_session()
    
    # Example 2: With new session
    example_with_new_session()


if __name__ == "__main__":
    main()
