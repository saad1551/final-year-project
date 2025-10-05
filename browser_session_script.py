#!/usr/bin/env python3
"""
Browser Session Script

This script starts a browser session, navigates to a specified URL,
captures the observation, converts it to markdown, and saves it to files.

Usage:
    python browser_session_script.py <URL> [output_file]
    
Examples:
    # Save full observation JSON and markdown
    python browser_session_script.py https://example.com observation.json
    
    # Save only markdown
    python browser_session_script.py https://example.com page.md --markdown-only
    
    # Save with custom markdown file
    python browser_session_script.py https://example.com data.json --markdown-file page.md
    
    # Verbose output
    python browser_session_script.py https://example.com --verbose
"""

import sys
import json
import argparse
import base64
import io
from datetime import datetime
from client import BrowserClient
from markdown import get_markdown_tree, render_markdown_tree
from configs.browser_config import BrowserObservation, NodeMetadata, BrowserConfig
from utils import safe_call, BrowserStatus


def image_to_base64(image):
    """Convert PIL Image to base64 string for JSON serialization."""
    if image is None:
        return None
    
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str


def convert_metadata_dict_to_objects(metadata_dict):
    """Convert metadata dict to NodeMetadata objects."""
    if not metadata_dict:
        return {}
    
    converted = {}
    for node_id, metadata in metadata_dict.items():
        if isinstance(metadata, dict):
            # Convert dict to NodeMetadata object
            converted[node_id] = NodeMetadata(**metadata)
        else:
            # Already a NodeMetadata object
            converted[node_id] = metadata
    
    return converted


def convert_to_markdown(observation_data):
    """
    Convert HTML observation to markdown format.
    
    Args:
        observation_data: Dictionary containing raw_html, metadata, etc.
        
    Returns:
        str: Markdown representation of the page
    """
    try:
        # Extract the raw HTML and metadata
        raw_html = observation_data.get('raw_html', '')
        metadata = observation_data.get('metadata', {})
        
        if not raw_html:
            return "No HTML content available"
        
        # Convert metadata dict to NodeMetadata objects if needed
        node_metadata = convert_metadata_dict_to_objects(metadata)
        
        # Get markdown tree
        markdown_nodes = safe_call(
            get_markdown_tree,
            raw_html,
            node_metadata,
            catch_errors=True,
            log_errors=True,
            max_errors=1
        )
        
        if markdown_nodes is BrowserStatus.ERROR:
            return "Failed to parse HTML into markdown structure"
        
        # Render markdown tree to text
        markdown_outputs = safe_call(
            render_markdown_tree,
            markdown_nodes,
            catch_errors=True,
            log_errors=True,
            max_errors=1
        )
        
        if markdown_outputs is BrowserStatus.ERROR:
            return "Failed to render markdown tree"
        
        # Join all markdown outputs
        markdown_text = " ".join(markdown_outputs)
        
        return markdown_text
        
    except Exception as e:
        return f"Error converting to markdown: {str(e)}"


def main():
    """Main function to run the browser session script."""
    parser = argparse.ArgumentParser(
        description="Start a browser session, navigate to a URL, and save observation"
    )
    parser.add_argument("url", help="URL to navigate to")
    parser.add_argument(
        "output_file", 
        nargs="?", 
        default=None,
        help="Output file to save observation (default: auto-generated)"
    )
    parser.add_argument(
        "--server-url", 
        default="http://localhost:3000",
        help="Server URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1920,
        help="Browser viewport width (default: 1920)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=1080,
        help="Browser viewport height (default: 1080)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--markdown-only", 
        action="store_true",
        help="Save only the markdown text (not the full observation JSON)"
    )
    parser.add_argument(
        "--markdown-file",
        help="Separate file to save markdown output (default: same as output file with .md extension)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        domain = args.url.replace("https://", "").replace("http://", "").split("/")[0]
        domain = "".join(c for c in domain if c.isalnum() or c in ".-_")[:20]
        args.output_file = f"observation_{domain}_{timestamp}.json"
    
    if args.verbose:
        print(f"Starting browser session...")
        print(f"URL: {args.url}")
        print(f"Output file: {args.output_file}")
        print(f"Server: {args.server_url}")
        print(f"Viewport: {args.width}x{args.height}")
    
    # Initialize the server client with proper config
    config = BrowserConfig(
        playwright_url=args.server_url,
        screen_width=args.width,
        screen_height=args.height
    )
    client = BrowserClient(config)
    
    try:
        # Start a new session
        if args.verbose:
            print("Starting new browser session...")
        
        start_result = client.start()
        
        if start_result != BrowserStatus.SUCCESS:
            print(f"❌ Failed to start session: {start_result}")
            sys.exit(1)
        
        session_id = client.session_id
        
        if args.verbose:
            print(f"Session started with ID: {session_id}")
        
        # Navigate to the URL
        if args.verbose:
            print(f"Navigating to: {args.url}")
        
        goto_result = client.goto(args.url)
        
        if goto_result != BrowserStatus.SUCCESS:
            print(f"❌ Failed to navigate to URL: {goto_result}")
            sys.exit(1)
        
        if args.verbose:
            print("Page loaded successfully")
        
        # Get the observation
        if args.verbose:
            print("Capturing observation...")
        
        observation_result = client.observation()
        
        if isinstance(observation_result, BrowserStatus) and observation_result != BrowserStatus.SUCCESS:
            print(f"❌ Failed to capture observation: {observation_result}")
            sys.exit(1)
        
        observation = observation_result
        
        if args.verbose:
            print("Observation captured successfully")
            print(f"Current URL: {observation.current_url}")
            print(f"HTML size: {len(observation.raw_html)} characters")
            print(f"Screenshot size: {len(observation.screenshot.tobytes())} bytes")
            print(f"Metadata nodes: {len(observation.metadata)}")
        
        # Convert to markdown
        if args.verbose:
            print("Converting to markdown...")
        
        # Convert BrowserObservation to dict format for convert_to_markdown
        observation_dict = {
            'raw_html': observation.raw_html,
            'metadata': observation.metadata,
            'current_url': observation.current_url,
            'screenshot': image_to_base64(observation.screenshot),
            'processed_image': image_to_base64(observation.processed_image)
        }
        
        markdown_text = convert_to_markdown(observation_dict)
        
        if args.verbose:
            print(f"Markdown conversion completed ({len(markdown_text)} characters)")
        
        # Add metadata to the observation
        observation_dict['session_info'] = {
            'session_id': session_id,
            'target_url': args.url,
            'timestamp': datetime.now().isoformat(),
            'viewport': {
                'width': args.width,
                'height': args.height
            },
            'server_url': args.server_url
        }
        
        # Add markdown to observation
        observation_dict['markdown'] = markdown_text
        
        # Determine output files
        if args.markdown_only:
            # Save only markdown
            output_file = args.output_file
            if not output_file.endswith('.md'):
                output_file += '.md'
            
            if args.verbose:
                print(f"Saving markdown to: {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            if args.verbose:
                print("Markdown saved successfully!")
            
            print(f"✅ Success! Markdown saved to: {output_file}")
            
        else:
            # Save full observation JSON
            if args.verbose:
                print(f"Saving observation to: {args.output_file}")
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(observation_dict, f, indent=2, ensure_ascii=False)
            
            if args.verbose:
                print("Observation saved successfully!")
            
            print(f"✅ Success! Observation saved to: {args.output_file}")
            
            # Save markdown separately if requested
            if args.markdown_file:
                markdown_file = args.markdown_file
            else:
                # Generate markdown filename from output file
                if args.output_file.endswith('.json'):
                    markdown_file = args.output_file[:-5] + '.md'
                else:
                    markdown_file = args.output_file + '.md'
            
            if args.verbose:
                print(f"Saving markdown to: {markdown_file}")
            
            with open(markdown_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            
            if args.verbose:
                print("Markdown saved successfully!")
            
            print(f"✅ Markdown also saved to: {markdown_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Always close the session
        try:
            client.close()
            if args.verbose:
                print("Session closed successfully")
        except Exception as e:
            if args.verbose:
                print(f"Warning: Failed to close session: {e}")


if __name__ == "__main__":
    main()
