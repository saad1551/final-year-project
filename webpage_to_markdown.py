#!/usr/bin/env python3
"""
Script to fetch webpage content using Playwright and convert it to markdown.

This script uses the existing markdown conversion functionality in the codebase
to convert HTML content from webpages into clean markdown format.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright
from markdown import get_markdown_tree, render_markdown_tree
from configs.browser_config import BrowserObservation, NodeToMetadata
from utils import safe_call, BrowserStatus


class WebpageToMarkdown:
    """Main class for converting webpages to markdown."""
    
    def __init__(self, headless: bool = True, timeout: int = 60000):
        """
        Initialize the converter.
        
        Args:
            headless: Whether to run browser in headless mode
            timeout: Page load timeout in milliseconds
        """
        self.headless = headless
        self.timeout = timeout
    
    async def fetch_page_content(self, url: str) -> tuple[str, str]:
        """
        Fetch the HTML content of a webpage using Playwright.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Tuple of (html_content, current_url)
        """
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Navigate to the page
                print(f"Fetching content from: {url}")
                response = await page.goto(url, timeout=self.timeout)
                
                if not response or not response.ok:
                    raise Exception(f"Failed to load page: {response.status if response else 'No response'}")
                
                # Wait for page to be fully loaded
                await page.wait_for_load_state('networkidle')
                
                # Get the HTML content
                html_content = await page.content()
                current_url = page.url
                
                print(f"Successfully fetched content from: {current_url}")
                return html_content, current_url
                
            finally:
                await browser.close()
    
    def convert_to_markdown(self, html_content: str, current_url: str) -> str:
        """
        Convert HTML content to markdown using the existing conversion system.
        
        Args:
            html_content: The HTML content to convert
            current_url: The current URL of the page
            
        Returns:
            The converted markdown content
        """
        print("Converting HTML to markdown...")

        # Convert HTML to markdown tree
        markdown_nodes = safe_call(
            get_markdown_tree,
            html_content,  # positional arg 1: raw_html
            None,  # positional arg 2: metadata
            catch_errors=True,  # safe_call parameter
            log_errors=True,  # safe_call parameter
            max_errors=3  # safe_call parameter
        )

        if markdown_nodes is BrowserStatus.ERROR:
            raise Exception("Failed to parse HTML into markdown tree")

        # Render markdown tree to text
        markdown_outputs = safe_call(
            render_markdown_tree,
            markdown_nodes,  # positional arg 1
            catch_errors=True,  # safe_call parameter
            log_errors=True,  # safe_call parameter
            max_errors=3  # safe_call parameter
        )

        if markdown_outputs is BrowserStatus.ERROR:
            raise Exception("Failed to render markdown tree")

        # Join all markdown outputs (use space, not newline)
        markdown_content = " ".join(markdown_outputs)

        print("Successfully converted to markdown")
        return markdown_content
    
    async def convert_url_to_markdown(self, url: str) -> str:
        """
        Convert a URL to markdown content.
        
        Args:
            url: The URL to convert
            
        Returns:
            The markdown content
        """
        # Fetch the webpage content
        html_content, current_url = await self.fetch_page_content(url)
        
        # Convert to markdown
        markdown_content = self.convert_to_markdown(html_content, current_url)
        
        return markdown_content


async def main():
    """Main function to handle command line arguments and execute conversion."""
    parser = argparse.ArgumentParser(
        description="Convert a webpage to markdown using Playwright",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webpage_to_markdown.py https://example.com
  python webpage_to_markdown.py https://example.com -o output.md
  python webpage_to_markdown.py https://example.com --no-headless
  python webpage_to_markdown.py https://example.com --timeout 60000
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
        "--no-headless",
        action="store_true",
        help="Run browser in non-headless mode (visible browser window)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=30000,
        help="Page load timeout in milliseconds (default: 30000)"
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        print("Error: URL must start with http:// or https://", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Create converter
        converter = WebpageToMarkdown(
            headless=not args.no_headless,
            timeout=args.timeout
        )
        
        # Convert URL to markdown
        markdown_content = await converter.convert_url_to_markdown(args.url)
        
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
    asyncio.run(main())
