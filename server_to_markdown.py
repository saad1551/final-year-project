#!/usr/bin/env python3
"""
Script to convert HTML from the JavaScript server to markdown.

This script fetches HTML content from the JavaScript server's observation endpoint
and converts it to markdown using the existing markdown conversion system.
"""

import argparse
import requests
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from markdown import get_markdown_tree, render_markdown_tree
from utils import safe_call, BrowserStatus


class ServerToMarkdown:
    """Class for converting HTML from JavaScript server to markdown."""
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        """
        Initialize the converter.
        
        Args:
            server_url: The URL of the JavaScript server
        """
        self.server_url = server_url.rstrip('/')
        self.session_id = None
    
    def start_session(self, width: int = 1920, height: int = 1080) -> str:
        """
        Start a new browser session.
        
        Args:
            width: Viewport width
            height: Viewport height
            
        Returns:
            Session ID
        """
        url = f"{self.server_url}/start?width={width}&height={height}"
        
        try:
            response = requests.post(url, json={})
            response.raise_for_status()
            self.session_id = response.text.strip()
            print(f"Started session: {self.session_id}")
            return self.session_id
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to start session: {e}")
    
    def navigate_to_url(self, url: str) -> None:
        """
        Navigate to a URL in the current session.
        
        Args:
            url: The URL to navigate to
        """
        if not self.session_id:
            raise Exception("No active session. Call start_session() first.")
        
        request_url = f"{self.server_url}/goto?url={url}&session_id={self.session_id}"
        
        try:
            response = requests.post(request_url)
            response.raise_for_status()
            print(f"Navigated to: {url}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to navigate to {url}: {e}")
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get the current page observation (HTML + metadata).
        
        Returns:
            Dictionary containing raw_html, metadata, screenshot, and current_url
        """
        if not self.session_id:
            raise Exception("No active session. Call start_session() first.")
        
        url = f"{self.server_url}/observation?session_id={self.session_id}"
        
        try:
            response = requests.post(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get observation: {e}")
    
    def close_session(self) -> None:
        """Close the current browser session."""
        if not self.session_id:
            return
        
        url = f"{self.server_url}/close?session_id={self.session_id}"
        
        try:
            response = requests.post(url)
            response.raise_for_status()
            print(f"Closed session: {self.session_id}")
            self.session_id = None
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to close session: {e}")
    
    def convert_to_markdown(self, html_content: str, metadata: Dict[str, Any]) -> str:
        """
        Convert HTML content to markdown using the existing conversion system.
        
        Args:
            html_content: The HTML content to convert
            metadata: The metadata from the observation
            
        Returns:
            The converted markdown content
        """
        print("Converting HTML to markdown...")

        # Convert HTML to markdown tree
        markdown_nodes = safe_call(
            get_markdown_tree,
            html_content,  # positional arg 1: raw_html
            metadata,      # positional arg 2: metadata
            catch_errors=True,  # safe_call parameter
            log_errors=True,    # safe_call parameter
            max_errors=3        # safe_call parameter
        )

        if markdown_nodes is BrowserStatus.ERROR:
            raise Exception("Failed to parse HTML into markdown tree")

        # Render markdown tree to text
        markdown_outputs = safe_call(
            render_markdown_tree,
            markdown_nodes,  # positional arg 1
            catch_errors=True,  # safe_call parameter
            log_errors=True,    # safe_call parameter
            max_errors=3        # safe_call parameter
        )

        if markdown_outputs is BrowserStatus.ERROR:
            raise Exception("Failed to render markdown tree")

        # Join all markdown outputs (use space, not newline)
        markdown_content = " ".join(markdown_outputs)

        print("Successfully converted to markdown")
        return markdown_content
    
    def url_to_markdown(self, url: str, width: int = 1920, height: int = 1080) -> str:
        """
        Convert a URL to markdown content using the server.
        
        Args:
            url: The URL to convert
            width: Viewport width
            height: Viewport height
            
        Returns:
            The markdown content
        """
        try:
            # Start session
            self.start_session(width, height)
            
            # Navigate to URL
            self.navigate_to_url(url)
            
            # Get observation
            observation = self.get_observation()
            
            # Convert to markdown
            markdown_content = self.convert_to_markdown(
                observation['raw_html'], 
                observation['metadata']
            )
            
            return markdown_content
            
        finally:
            # Always close the session
            self.close_session()


def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a webpage to markdown using the JavaScript server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server_to_markdown.py https://example.com
  python server_to_markdown.py https://example.com -o output.md
  python server_to_markdown.py https://example.com --server http://localhost:3000
  python server_to_markdown.py https://example.com --width 1280 --height 720
        """
    )
    
    parser.add_argument(
        "url",
        help="The URL of the webpage to convert to markdown"
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
    
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Viewport width in pixels (default: 1920)"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Viewport height in pixels (default: 1080)"
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create converter
        converter = ServerToMarkdown(server_url=args.server)
        
        # Convert URL to markdown
        markdown_content = converter.url_to_markdown(
            args.url, 
            width=args.width, 
            height=args.height
        )
        
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
