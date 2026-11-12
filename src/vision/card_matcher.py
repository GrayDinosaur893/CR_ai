"""
Fast local OpenCV Card Template Matcher (< 4 ms per 4-slot hand).
Matches cropped hand slots against deck templates in `data/card_templates/*.png`
using a sliding inner-artwork patch with blended grayscale + Sobel gradient
normalized cross-correlation (invariant to low-elixir dimming & cooldown curtains).
"""

import glob
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


CARD_METADATA: Dict[str, Dict[str, Any]] = {
    "skeletons": {"cost": 1, "role": "cycle", "spell_radius": 0.0},
    "electro_spirit": {"cost": 1, "role": "cycle", "spell_radius": 0.0},
    "arrows": {"cost": 3, "role": "spell", "spell_radius": 4.0},
    "cannon": {"cost": 3, "role": "building", "spell_radius": 0.0},
    "knight": {"cost": 3, "role": "minitank", "spell_radius": 0.0},
    "fireball": {"cost": 4, "role": "spell", "spell_radius": 2.5},
    "musketeer": {"cost": 4, "role": "ranged", "spell_radius": 0.0},
    "hog_rider": {"cost": 4, "role": "wincon", "spell_radius": 0.0},
    "minions": {"cost": 3, "role": "air_dps", "spell_radius": 0.0},
    "archers": {"cost": 3, "role": "ranged", "spell_radius": 0.0},
    "goblin_barrel": {"cost": 3, "role": "wincon", "spell_radius": 0.0},
}


class CardTemplateMatcher:
    """Identifies cards in Clash Royale hand slots using sliding gray + gradient templates."""

    # Reference template canvas (matches 100x120 templates scaled 0.5x -> 50x60)
    REF_W = 50
    REF_H = 60
    PATCH_W = 32
    PATCH_H = 31

    def __init__(self, template_dir: Optional[str] = None, min_confidence: float = 0.40, min_margin: float = 0.012):
        if template_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            template_dir = os.path.join(root_dir, "data", "card_templates")
        self.template_dir = template_dir
        self.min_confidence = float(min_confidence)
        self.min_margin = float(min_margin)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self.templates: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._slot_cache: Dict[int, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_templates()

    @staticmethod
    def _grad_mag(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def _prep_patch(self, bgr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        return self._clahe.apply(gray)

    def _load_templates(self) -> None:
        if not os.path.isdir(self.template_dir):
            return
        for path in sorted(glob.glob(os.path.join(self.template_dir, "*.png"))):
            card_name = os.path.splitext(os.path.basename(path))[0].lower()
            img = cv2.imread(path)
            if img is None or img.size == 0:
                continue
            resized = cv2.resize(img, (100, 120), interpolation=cv2.INTER_AREA)
            gray = self._clahe.apply(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
            # Central card artwork (avoids border gutters & bottom elixir cost droplet)
            patch = cv2.resize(gray[24:86, 22:86], (self.PATCH_W, self.PATCH_H), interpolation=cv2.INTER_AREA)
            self.templates[card_name] = (patch, self._grad_mag(patch))

    @staticmethod
    def _slot_sig(crop: np.ndarray) -> np.ndarray:
        """16x16 zero-mean L2-normalized grayscale artwork signature for cache hit check."""
        sub = crop[::4, ::4] if (crop.shape[0] >= 64 and crop.shape[1] >= 64) else (crop[::2, ::2] if (crop.shape[0] >= 32 and crop.shape[1] >= 32) else crop)
        g = cv2.resize(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
        g -= float(g.mean())
        return g / (float(np.linalg.norm(g)) + 1e-6)

    def invalidate_slot(self, slot_num: Optional[int] = None) -> None:
        """Clears cached card ID for one slot (e.g., on card play/cooldown) or all slots."""
        if slot_num is None:
            self._slot_cache.clear()
        else:
            self._slot_cache.pop(int(slot_num), None)

    def identify(self, slot_crop: np.ndarray, slot_num: int, use_cache: bool = True) -> Dict[str, Any]:
        """
        Matches a single hand slot BGR crop against loaded card templates.
        Uses CLAHE + top1/top2 margin + 16x16 structural slot cache (invalidated on slot play/cooldown).
        """
        fallback = {
            "card_name": f"slot_{slot_num}",
            "card_cost": None,
            "card_role": "unknown",
            "spell_radius": 0.0,
            "card_confidence": 0.0,
            "card_margin": 0.0,
            "cache_hit": False,
        }
        if not self.templates or slot_crop is None or slot_crop.size == 0:
            return fallback

        sig = None
        if use_cache:
            sig = self._slot_sig(slot_crop)
            cached = self._slot_cache.get(int(slot_num))
            if cached is not None:
                sim = float((cached["sig"] * sig).sum())
                thresh = 0.86 if int(slot_num) == 0 else 0.94
                if sim > thresh:
                    self._cache_hits += 1
                    res = dict(cached["res"])
                    res["cache_hit"] = True
                    return res

        self._cache_misses += 1
        gray = self._prep_patch(slot_crop, (self.REF_W, self.REF_H))
        search = gray[4:52, 2:48]
        if search.shape[0] < self.PATCH_H or search.shape[1] < self.PATCH_W:
            return fallback

        coarse = []
        for name, (tg, _tgm) in self.templates.items():
            res_g = cv2.matchTemplate(search, tg, cv2.TM_CCOEFF_NORMED)
            coarse.append((float(res_g.max()), name, res_g))
        coarse.sort(reverse=True, key=lambda item: item[0])

        search_gm = self._grad_mag(search)
        scores = []
        for _, name, res_g in coarse[:4]:
            res_m = cv2.matchTemplate(search_gm, self.templates[name][1], cv2.TM_CCOEFF_NORMED)
            scores.append((float((0.62 * res_g + 0.38 * res_m).max()), name))
        scores.sort(reverse=True)
        best_score, best_name = scores[0]
        second_score = scores[1][0] if len(scores) > 1 else 0.0
        margin = best_score - second_score

        if best_score < self.min_confidence or margin < self.min_margin:
            fallback["card_confidence"] = round(max(0.0, best_score), 3)
            fallback["card_margin"] = round(margin, 3)
            if use_cache and sig is not None and int(slot_num) == 0:
                self._slot_cache[0] = {"sig": sig, "res": dict(fallback)}
            else:
                self._slot_cache.pop(int(slot_num), None)
            return fallback

        meta = CARD_METADATA.get(best_name, {})
        result = {
            "card_name": best_name,
            "card_cost": meta.get("cost"),
            "card_role": meta.get("role", "unknown"),
            "spell_radius": float(meta.get("spell_radius", 0.0)),
            "card_confidence": round(best_score, 3),
            "card_margin": round(margin, 3),
            "cache_hit": False,
        }
        if use_cache and sig is not None:
            self._slot_cache[int(slot_num)] = {"sig": sig, "res": dict(result)}
        return result

    @staticmethod
    def get_card_meta(card_name: str) -> Dict[str, Any]:
        return CARD_METADATA.get(str(card_name).lower().replace(" ", "_"), {})
