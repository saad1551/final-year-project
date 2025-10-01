#!/usr/bin/env python3
"""
Test script for the webpage to markdown converter.
"""

import asyncio
import tempfile
from pathlib import Path
from webpage_to_markdown import WebpageToMarkdown


async def test_converter():
    """Test the webpage converter with a simple webpage."""
    print("Testing webpage to markdown converter...")
    
    # Create converter
    converter = WebpageToMarkdown(headless=True, timeout=30000)
    
    # Test with a simple webpage
    test_url = "https://httpbin.org/html"
    
    try:
        # Convert URL to markdown
        markdown_content = await converter.convert_url_to_markdown(test_url)
        
        print("✅ Conversion successful!")
        print(f"Markdown content length: {len(markdown_content)} characters")
        print("\nFirst 500 characters of markdown:")
        print("-" * 50)
        print(markdown_content[:500])
        print("-" * 50)
        
        # Test saving to file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            temp_file = f.name
        
        print(f"✅ Test file saved to: {temp_file}")
        
        # Clean up
        Path(temp_file).unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_converter())
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        exit(1)
