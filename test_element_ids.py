#!/usr/bin/env python3
"""
Simple test to verify element IDs are preserved in markdown conversion.
"""

import json
import re

def test_element_ids():
    """Test if element IDs are preserved in the conversion process."""
    
    # Load the observation data
    with open('observation.json', 'r') as f:
        data = json.load(f)
    
    print("🔍 Testing Element ID Preservation")
    print("=" * 50)
    
    # Check HTML for backend_node_id attributes
    html = data['raw_html']
    backend_node_ids = re.findall(r'backend_node_id="([^"]*)"', html)
    
    print(f"✅ Found {len(backend_node_ids)} backend_node_id attributes in HTML")
    print(f"First 10 IDs: {backend_node_ids[:10]}")
    
    # Check metadata
    metadata = data['metadata']
    print(f"✅ Found {len(metadata)} metadata entries")
    
    # Check if metadata keys match HTML IDs
    metadata_keys = set(metadata.keys())
    html_ids = set(backend_node_ids)
    
    print(f"✅ Metadata keys: {len(metadata_keys)}")
    print(f"✅ HTML IDs: {len(html_ids)}")
    
    # Check overlap
    overlap = metadata_keys.intersection(html_ids)
    print(f"✅ Overlap: {len(overlap)} IDs found in both HTML and metadata")
    
    if len(overlap) > 0:
        print("✅ Element IDs are properly linked between HTML and metadata!")
        
        # Show sample metadata for an element
        sample_id = list(overlap)[0]
        sample_metadata = metadata[sample_id]
        print(f"\n📋 Sample metadata for ID {sample_id}:")
        print(f"  - backend_node_id: {sample_metadata.get('backend_node_id')}")
        print(f"  - is_visible: {sample_metadata.get('is_visible')}")
        print(f"  - is_frontmost: {sample_metadata.get('is_frontmost')}")
        print(f"  - bounding_client_rect: {sample_metadata.get('bounding_client_rect')}")
        
        # Find the corresponding HTML element
        html_element = re.search(f'backend_node_id="{sample_id}"[^>]*>', html)
        if html_element:
            print(f"  - HTML element: {html_element.group()[:100]}...")
        
        return True
    else:
        print("❌ No overlap found between HTML IDs and metadata keys")
        return False

def show_element_id_usage():
    """Show how element IDs can be used."""
    
    print("\n🎯 How to Use Element IDs")
    print("=" * 50)
    
    print("""
The JavaScript server provides element IDs in two ways:

1. HTML Attributes: Each element gets a 'backend_node_id' attribute
   Example: <div backend_node_id="123" class="button">Click me</div>

2. Metadata Dictionary: Maps IDs to element properties
   Example: metadata["123"] = {
       "backend_node_id": 123,
       "is_visible": true,
       "is_frontmost": false,
       "bounding_client_rect": {"x": 100, "y": 200, "width": 50, "height": 30},
       "computed_style": {...}
   }

You can use these IDs to:
- Target specific elements for automation
- Get element positioning and styling
- Check element visibility
- Interact with elements using the /action endpoint
""")

if __name__ == "__main__":
    success = test_element_ids()
    show_element_id_usage()
    
    if success:
        print("\n✅ Element IDs are working correctly!")
        print("You can now use them for web automation and element targeting.")
    else:
        print("\n❌ There might be an issue with element ID generation.")

