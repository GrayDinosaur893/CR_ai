"""
Clash Royale Official Supercell Developer API Connector.
Docs: https://developer.clashroyale.com
Provides direct access to live battle logs, player decks, card levels, and match history.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

CR_API_BASE_URL = "https://api.clashroyale.com/v1"

class ClashRoyaleAPIConnector:
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token or os.getenv("CLASH_ROYALE_API_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.api_token)

    def _get(self, endpoint: str) -> Dict[str, Any]:
        if not self.api_token:
            raise ValueError(
                "CLASH_ROYALE_API_TOKEN is not set. Get an official API key from "
                "https://developer.clashroyale.com"
            )

        url = f"{CR_API_BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_player(self, player_tag: str) -> Dict[str, Any]:
        """Fetch player profile by #TAG (e.g. #9PJ0L2PR)."""
        formatted_tag = player_tag.replace("#", "%23")
        return self._get(f"players/{formatted_tag}")

    def get_battle_log(self, player_tag: str) -> Dict[str, Any]:
        """Fetch recent 25 live matches, decks, tower damage, and crowns."""
        formatted_tag = player_tag.replace("#", "%23")
        return self._get(f"players/{formatted_tag}/battlelog")

    def get_cards(self) -> Dict[str, Any]:
        """Fetch all official cards metadata, elixir costs, and icon URLs."""
        return self._get("cards")
