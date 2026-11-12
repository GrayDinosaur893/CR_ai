"""
OBSERVER MODULE 3: Qwen VLM Client.
Responsibilities:
1. Dispatch client-side PII-redacted image frames to Qwen VLM via OpenRouter API.
2. System & User prompts strictly enforce deterministic JSON schema.
3. Unknown items must go to "unknown" array - zero guessing allowed.
4. "Skip-on-busy" concurrency guard: drops incoming frames if previous call is in-flight.
5. Measures exact roundtrip inference latency (ms).
"""

import os
import time
import json
import base64
import threading
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a pure vision perception engine for Clash Royale.
Analyze the provided game screenshot and output strict JSON ONLY.
Do not include any conversational markdown fences, headers, or explanations.

Rules:
1. Identify the screen state (e.g., "lobby", "battle_arena", "clan", "shop", "deck_builder", "matchmaking", "loading", "battle_end").
2. Detect visible interactive or informational components. For each component provide:
   - "name": clear identifier (e.g. "battle_button", "pass_royale_banner", "gold_display", "gem_display", "elixir_bar", "card_slot_1", "card_slot_2", "card_slot_3", "card_slot_4", "trophy_count", "crown_chest").
   - "box_pct": bounding box in relative percentage coordinates [x1, y1, x2, y2] where 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0.
   - "seen": true
3. IF AN ELEMENT IS UNCLEAR, BLURRED, OR YOU ARE NOT 100% CERTAIN, PUT ITS NAME OR DESCRIPTION IN THE "unknown" ARRAY. DO NOT GUESS.
4. Provide "confidence": "high", "medium", or "low".

Required JSON schema:
{
  "screen": "<screen_name>",
  "confidence": "high|medium|low",
  "components": [
    {
      "name": "<name>",
      "box_pct": [0.0, 0.0, 0.0, 0.0],
      "seen": true
    }
  ],
  "unknown": []
}"""

USER_PROMPT = "Observe this game frame. Output valid JSON matching the schema precisely. No guessing."


TACTICAL_BRAIN_PROMPT = """You are the real-time Qwen Clash Royale Tactical Brain (Ryley CR Positive-Trade & Overspend Decision Engine).
You receive the raw arena frame PLUS live CV telemetry: OUR elixir, OUR 4-card hand, OPPONENT estimated elixir, OPPONENT estimated hand/recent plays, and active threats.

Decision Rules (NEVER just imitate blindly — compute hand-vs-hand & elixir math):
1. OVERSPEND vs CONSERVE DECISION:
   - Compute elixir_advantage = player_elixir - opp_elixir_est.
   - Set allow_overspend=true ONLY when:
     a) elixir_advantage >= +2.0 AND opponent just spent heavy elixir or their Hog/wincon counter is out of opp_estimated_hand, OR
     b) urgent tower defense against a 3+ unit heavy push where cheap cards alone will lose the tower.
   - Otherwise set allow_overspend=false and enforce STRICT POSITIVE ELIXIR TRADES (conserve elixir, max_defense_cost <= 3).
2. STOP WASTING CANNON:
   - Do NOT recommend Cannon on lone 1-3 elixir ground troops (Knight, Musketeer, Skeletons, Wizard, Bomber) — counter cheap with Skeletons (1), Electro Spirit (1), Knight (3), or Arrows (3)!
   - Reserve Cannon specifically for building-targeters (Hog Rider, Giant, Balloon, Royal Giant, Ram Rider, Golem) or heavy multi-unit ground pushes.
3. FAST POSITIVE COUNTERS:
   - React promptly as enemy crosses bridge (row >= 18) with a 1-3 elixir counter rather than letting pushes pile up late at the tower.
   - Spot flying threats (Balloon, Minions, Baby Dragon, Bats) -> anti_air_needed=true -> prioritize Musketeer / Arrows / Electro Spirit.
4. OPPONENT CARD TRACKING:
   - List real enemy card names visible on screen in opp_cards_seen so our elixir & 4-card cycle tracker stays exact.

Output STRICT JSON ONLY (no markdown):
{
  "strategy_mode": "conserve_elixir|cheap_trade_defend|cannon_pull_defend|overspend_punish|build_counterpush",
  "allow_overspend": false,
  "max_defense_cost": 3,
  "preferred_lane": "left|right|auto",
  "elixir_floor": 3.5,
  "anti_air_needed": false,
  "recommended_card": "skeletons|electro_spirit|knight|arrows|musketeer|cannon|fireball|hog_rider|none",
  "opp_cards_seen": [],
  "hand_cards": {"1": "", "2": "", "3": "", "4": ""},
  "threats_summary": "concise enemy units seen",
  "reasoning": "elixir diff + hand-vs-hand reason why to conserve or overspend now"
}"""


class QwenVLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: float = 15.0
    ):
        env_vals: Dict[str, str] = {}
        if os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#") and "=" in s:
                        k, v = s.split("=", 1)
                        env_vals[k.strip()] = v.strip()

        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY") or env_vals.get("OPENROUTER_API_KEY")

        self.api_key = api_key
        # Configurable model - prefer env VLM_MODEL / OPENROUTER_MODEL (e.g. qwen/qwen3-vl-30b-a3b-instruct)
        self.model = (
            model
            or os.getenv("VLM_MODEL")
            or env_vals.get("VLM_MODEL")
            or os.getenv("OPENROUTER_MODEL")
            or env_vals.get("OPENROUTER_MODEL")
            or "qwen/qwen3-vl-30b-a3b-instruct"
        )
        self.timeout_sec = timeout_sec
        self._is_busy = False
        self._lock = threading.Lock()

    def is_busy(self) -> bool:
        with self._lock:
            return self._is_busy

    def encode_frame_to_base64(self, frame_bgr: np.ndarray, quality: int = 85) -> str:
        """Encodes frame numpy array to JPEG base64 string."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', frame_bgr, encode_params)
        if not success:
            raise ValueError("Failed to encode frame to JPEG")
        return base64.b64encode(buffer).decode('utf-8')

    def _post_vlm_messages(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Shared skip-on-busy OpenRouter VLM request helper."""
        if not self.api_key:
            return False, "API_KEY_NOT_CONFIGURED", {"error": "Missing OPENROUTER_API_KEY"}

        with self._lock:
            if self._is_busy:
                return False, "BUSY_SKIPPED", {"skipped": True}
            self._is_busy = True

        start_time = time.time()
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/clash-ai/cr-observer",
                "X-Title": "CR Bot Observer",
            }
            req = urllib.request.Request(
                OPENROUTER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                latency_ms = (time.time() - start_time) * 1000.0
                resp_json = json.loads(response.read().decode("utf-8"))
                choices = resp_json.get("choices", [])
                if not choices:
                    return False, "EMPTY_CHOICES", {"latency_ms": latency_ms, "raw": resp_json}
                content = choices[0].get("message", {}).get("content", "").strip()
                metadata = {
                    "model": self.model,
                    "latency_ms": latency_ms,
                    "usage": resp_json.get("usage", {}),
                    "full_response": resp_json,
                }
                return True, content, metadata
        except urllib.error.HTTPError as e:
            latency_ms = (time.time() - start_time) * 1000.0
            error_body = e.read().decode("utf-8", errors="ignore")
            return False, f"HTTP_{e.code}: {error_body}", {"latency_ms": latency_ms, "http_code": e.code}
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000.0
            return False, f"NETWORK_ERROR: {str(e)}", {"latency_ms": latency_ms, "error": str(e)}
        finally:
            with self._lock:
                self._is_busy = False

    def infer(self, frame_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sends frame to Qwen VLM for strict component perception audit.
        Implements strict skip-on-busy: drops call immediately if in flight.
        """
        b64_image = self.encode_frame_to_base64(frame_bgr)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            },
        ]
        return self._post_vlm_messages(messages, max_tokens=1024, temperature=0.1)

    def analyze_tactical_state(
        self, frame_bgr: np.ndarray, telemetry: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Real-time async Qwen tactical brain query using compressed frame + exact CV telemetry.
        """
        h, w = frame_bgr.shape[:2]
        if w > 500:
            scale = 480.0 / float(w)
            resized = cv2.resize(frame_bgr, (480, max(1, int(round(h * scale)))))
        else:
            resized = frame_bgr
        b64_image = self.encode_frame_to_base64(resized, quality=72)
        user_text = f"LIVE TELEMETRY:\n{json.dumps(telemetry, ensure_ascii=False)}\nReturn tactical JSON."
        messages = [
            {"role": "system", "content": TACTICAL_BRAIN_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            },
        ]
        return self._post_vlm_messages(messages, max_tokens=260, temperature=0.1)
