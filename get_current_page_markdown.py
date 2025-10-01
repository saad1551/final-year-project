#!/usr/bin/env python3
"""
Script to get the markdown of the currently open page in the Playwright server.

This script connects to an existing Playwright server session and converts
the currently open page to markdown format.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from server_client import ServerClient
from server_to_markdown import ServerToMarkdown


def get_current_page_markdown(session_id: str, server_url: str = "http://localhost:3000") -> str:
    """
    Get the markdown of the currently open page in the Playwright server.
    
    Args:
        session_id: The session ID of the active browser session
        server_url: The URL of the JavaScript server
        
    Returns:
        The markdown content of the current page
        
    Raises:
        Exception: If unable to get observation or convert to markdown
    """
    # Create a server client
    client = ServerClient(server_url)
    client.session_id = session_id  # Use existing session
    
    try:
        # Get the current page observation
        print(f"Getting observation for session: {session_id}")
        observation = client.get_observation()
        
        # Create markdown converter
        converter = ServerToMarkdown(server_url)
        
        # Convert to markdown
        markdown_content = converter.convert_to_markdown(
            observation['raw_html'], 
            observation['metadata']
        )
        
        print(f"Successfully converted current page to markdown")
        print(f"Current URL: {observation.get('current_url', 'Unknown')}")
        
        return markdown_content
        
    except Exception as e:
        raise Exception(f"Failed to get current page markdown: {e}")


def get_current_page_markdown_with_new_session(url: str, server_url: str = "http://localhost:3000") -> str:
    """
    Get the markdown of a page by creating a new session and navigating to it.
    
    Args:
        url: The URL to navigate to
        server_url: The URL of the JavaScript server
        
    Returns:
        The markdown content of the page
    """
    # Use the existing ServerToMarkdown class which handles session management
    converter = ServerToMarkdown(server_url)
    return converter.url_to_markdown(url)


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Get markdown of currently open page in Playwright server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get markdown of currently open page (requires session_id)
  python get_current_page_markdown.py --session-id abc123
  
  # Get markdown by navigating to a URL (creates new session)
  python get_current_page_markdown.py --url https://example.com
  
  # Save output to file
  python get_current_page_markdown.py --session-id abc123 -o output.md
  
  # Use different server
  python get_current_page_markdown.py --session-id abc123 --server http://localhost:3001
        """
    )
    
    # Mutually exclusive group for session-id vs url
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-id",
        help="Session ID of the active browser session"
    )
    group.add_argument(
        "--url",
        help="URL to navigate to (creates new session)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: print to stdout)"
    )
    
    parser.add_argument(
        "--server",
        default="http://localhost:3000",
        help="JavaScript server URL (default: http://localhost:3000)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.session_id:
            # Get markdown of currently open page
            markdown_content = get_current_page_markdown(args.session_id, args.server)
        else:
            # Navigate to URL and get markdown
            if not args.url.startswith(('http://', 'https://')):
                print("Error: URL must start with http:// or https://", file=sys.stderr)
                sys.exit(1)
            markdown_content = get_current_page_markdown_with_new_session(args.url, args.server)
        
        # Output the result
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"Markdown content saved to: {output_path}")
        else:
            print("\n" + "="*50)
            print("MARKDOWN OUTPUT")
            print("="*50)
            print(markdown_content)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
