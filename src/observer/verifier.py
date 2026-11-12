"""
OBSERVER MODULE 4: Schema Verifier & Sanitizer.
Responsibilities:
1. Strict schema validation against the Qwen VLM response contract.
2. Invariants:
   - "screen": str
   - "confidence": "high|medium|low" (stored for logging only, never used as an execution gate)
   - "components": list of {name: str, box_pct: [x1,y1,x2,y2], seen: bool}
   - Bounding box invariants: 0.0 <= x1 <= x2 <= 1.0 and 0.0 <= y1 <= y2 <= 1.0
     (Automatically normalizes if model outputs per-mille 0-1000 or pixel coordinates).
   - "unknown": list of str
3. If JSON syntax or schema verification fails -> mark UNKNOWN immediately. ZERO retry loops.
4. Account Name PII Sanitizer: Masks any account or player tags appearing in string values.
"""

import re
import json
from typing import Dict, Any, Tuple, Optional

class SchemaVerifier:
    def __init__(self, masked_terms: Optional[list] = None):
        self.masked_terms = masked_terms or []

    def clean_json_string(self, raw_str: str) -> str:
        """Strips potential surrounding markdown backticks if present."""
        cleaned = raw_str.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def sanitize_pii_in_text(self, text: str) -> str:
        """Sanitizes sensitive terms from strings."""
        sanitized = text
        for term in self.masked_terms:
            if term and len(term) > 3:
                sanitized = re.sub(re.escape(term), "[MASKED_PII]", sanitized, flags=re.IGNORECASE)
        return sanitized

    def normalize_box(self, box: list) -> Optional[list]:
        """
        Normalizes box to [x1, y1, x2, y2] in 0.0 to 1.0 float range.
        Handles both 0.0-1.0 floats and Qwen's native 0-1000 integer coordinates.
        """
        try:
            coords = [float(v) for v in box]
        except (ValueError, TypeError):
            return None

        # Check if coordinates are in 0-1000 space (Qwen standard vision tokens)
        if any(c > 1.0 for c in coords):
            # If max coordinate is <= 1000, scale down by 1000
            scale = 1000.0 if max(coords) <= 1000.0 else max(coords)
            coords = [c / scale for c in coords]

        c1, c2, c3, c4 = coords
        x1 = max(0.0, min(1.0, min(c1, c3)))
        x2 = max(0.0, min(1.0, max(c1, c3)))
        y1 = max(0.0, min(1.0, min(c2, c4)))
        y2 = max(0.0, min(1.0, max(c2, c4)))

        return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]

    def verify(self, raw_content: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validates raw model output against the strict observer schema.
        Returns (is_valid: bool, status: str, parsed_data: dict).
        Status is either 'VALID' or 'UNKNOWN'.
        """
        cleaned_str = self.clean_json_string(raw_content)

        try:
            data = json.loads(cleaned_str)
        except Exception as e:
            return False, "UNKNOWN", {
                "error": f"JSON_DECODE_ERROR: {str(e)}",
                "raw_text": raw_content
            }

        if not isinstance(data, dict):
            return False, "UNKNOWN", {
                "error": "ROOT_NOT_AN_OBJECT",
                "raw_text": raw_content
            }

        # Check required fields
        required_keys = {"screen", "confidence", "components", "unknown"}
        missing_keys = required_keys - set(data.keys())
        if missing_keys:
            return False, "UNKNOWN", {
                "error": f"MISSING_REQUIRED_KEYS: {list(missing_keys)}",
                "raw_text": raw_content
            }

        screen = str(data.get("screen", "")).strip()
        if not screen:
            return False, "UNKNOWN", {"error": "EMPTY_SCREEN_NAME", "raw_text": raw_content}

        confidence = str(data.get("confidence", "")).lower().strip()
        if confidence not in {"high", "medium", "low"}:
            confidence = "unspecified"

        components = data.get("components")
        if not isinstance(components, list):
            return False, "UNKNOWN", {"error": "COMPONENTS_NOT_A_LIST", "raw_text": raw_content}

        unknown_list = data.get("unknown")
        if not isinstance(unknown_list, list):
            return False, "UNKNOWN", {"error": "UNKNOWN_NOT_A_LIST", "raw_text": raw_content}

        validated_components = []
        for idx, comp in enumerate(components):
            if not isinstance(comp, dict):
                return False, "UNKNOWN", {
                    "error": f"COMPONENT_{idx}_NOT_AN_OBJECT",
                    "raw_text": raw_content
                }

            name = str(comp.get("name", "")).strip()
            if not name:
                return False, "UNKNOWN", {
                    "error": f"COMPONENT_{idx}_MISSING_NAME",
                    "raw_text": raw_content
                }

            box = comp.get("box_pct")
            if not isinstance(box, list) or len(box) != 4:
                return False, "UNKNOWN", {
                    "error": f"COMPONENT_{idx}_INVALID_BOX_LENGTH",
                    "raw_text": raw_content
                }

            norm_box = self.normalize_box(box)
            if norm_box is None:
                return False, "UNKNOWN", {
                    "error": f"COMPONENT_{idx}_INVALID_COORDINATES",
                    "raw_text": raw_content
                }

            seen = bool(comp.get("seen", True))

            validated_components.append({
                "name": self.sanitize_pii_in_text(name),
                "box_pct": norm_box,
                "seen": seen
            })

        sanitized_unknown = [self.sanitize_pii_in_text(str(item)) for item in unknown_list]

        verified_data = {
            "screen": self.sanitize_pii_in_text(screen),
            "confidence": confidence,
            "components": validated_components,
            "unknown": sanitized_unknown
        }

        return True, "VALID", verified_data
