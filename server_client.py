#!/usr/bin/env python3
"""
Simple Python client for the JavaScript server.

This provides a convenient interface to interact with the JavaScript server
and convert HTML observations to markdown.
"""

import requests
import json
from typing import Dict, Any, Optional


class ServerClient:
    """Client for interacting with the JavaScript server."""
    
    def __init__(self, server_url: str = "http://localhost:3000"):
        """
        Initialize the client.
        
        Args:
            server_url: The URL of the JavaScript server
        """
        self.server_url = server_url.rstrip('/')
        self.session_id = None
    
    def start_session(self, width: int = 1920, height: int = 1080) -> str:
        """Start a new browser session."""
        url = f"{self.server_url}/start?width={width}&height={height}"
        response = requests.post(url, json={})
        response.raise_for_status()
        self.session_id = response.text.strip()
        return self.session_id
    
    def navigate(self, url: str) -> None:
        """Navigate to a URL."""
        if not self.session_id:
            raise Exception("No active session")
        
        request_url = f"{self.server_url}/goto?url={url}&session_id={self.session_id}"
        response = requests.post(request_url)
        response.raise_for_status()
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current page observation."""
        if not self.session_id:
            raise Exception("No active session")
        
        url = f"{self.server_url}/observation?session_id={self.session_id}"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()
    
    def execute_action(self, action: list) -> None:
        """Execute an action on the page."""
        if not self.session_id:
            raise Exception("No active session")
        
        url = f"{self.server_url}/action?session_id={self.session_id}"
        response = requests.post(url, json=action)
        response.raise_for_status()
    
    def close_session(self) -> None:
        """Close the current session."""
        if not self.session_id:
            return
        
        url = f"{self.server_url}/close?session_id={self.session_id}"
        try:
            response = requests.post(url)
            response.raise_for_status()
        finally:
            self.session_id = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - always close session."""
        self.close_session()


# Example usage
if __name__ == "__main__":
    # Example: Convert Google to markdown
    from server_to_markdown import ServerToMarkdown
    
    converter = ServerToMarkdown()
    markdown = converter.url_to_markdown("https://google.com")
    print("Google homepage as markdown:")
    print(markdown)

