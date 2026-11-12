"""
CLASH IQ — Full Perception -> 90%+ Accuracy Gate -> Tactical Planner -> Dyadic Execution Engine.

Pipeline:
  1. Perception (ComponentDetector + ArenaThreat + optional async Qwen VLM card tagger)
  2. Strict AccuracyGate (>= 90.0% benchmark + >= 0.90 live frame confidence)
  3. TacticalActionPlanner + ElixirTracker (playable slots, enemy threat tiles, tower states)
  4. DyadicOpportunityScheduler (2^x gates + emergency threat interrupt)
  5. ActionExecutor (SHADOW DRY-RUN by default; live clicks ONLY when Gate UNLOCKED + --live)
"""

import os
import sys
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.vision.component_detector import ComponentDetector
from src.game_state.elixir_tracker import ElixirTracker
from src.game_state.opponent_tracker import OpponentTracker
from src.strategy.action_planner import TacticalActionPlanner
from src.strategy.opportunity_scheduler import DyadicOpportunityScheduler
from src.engine.accuracy_gate import AccuracyGate
from src.engine.action_executor import ActionExecutor
from src.engine.recorder import GameplayRecorder


class AutonomousClashPlayer:
    def __init__(
        self,
        live_armed: bool = False,
        min_accuracy_pct: float = 90.0,
        use_vlm_card_id: bool = False,
        auto_battle: bool = False,
        record: bool = False,
        record_dir: str = "recordings",
        save_preview_path: str = os.path.join("frames", "autonomous_preview.png"),
        state_export_path: str = os.path.join("data", "live_game_state.json"),
        human_observe: bool = False,
    ):
        self.detector = ComponentDetector()
        self.gate = AccuracyGate(min_accuracy_pct=min_accuracy_pct)
        self.tracker = ElixirTracker(initial_elixir=5.0)
        self.opp_tracker = OpponentTracker()
        self.planner = TacticalActionPlanner()
        self.scheduler = DyadicOpportunityScheduler()
        self.human_observe = bool(human_observe)
        effective_live = False if self.human_observe else bool(live_armed)
        self.executor = ActionExecutor(accuracy_gate=self.gate, live_armed=effective_live)

        self.live_armed = effective_live
        self.use_vlm_card_id = bool(use_vlm_card_id)
        self.auto_battle = False if self.human_observe else bool(auto_battle)
        self.record = bool(record)
        self.record_dir = record_dir
        self.recorder: Optional[GameplayRecorder] = (
            GameplayRecorder(base_dir=record_dir, prefix="human" if self.human_observe else "session")
            if self.record
            else None
        )
        self.save_preview_path = save_preview_path
        self.state_export_path = state_export_path

        self._vlm_client = None
        self._vlm_card_map: Dict[int, str] = {}
        self._vlm_tactical_advice: Dict[str, Any] = {}
        self._vlm_tactical_ts: float = -100.0
        self._vlm_lock = threading.Lock()
        self._last_vlm_ts = 0.0
        self._last_enemy_count = 0
        self._last_opp_spawn_count = 0
        self._last_opp_spend_elapsed = -10.0
        self._in_battle_prev = False
        self._battle_streak = 0
        self._frame_counter = 0
        self._prev_hand_slots: Dict[int, Dict[str, Any]] = {}
        self._prev_player_elixir: float = 0.0
        self._last_human_play_ts: float = -10.0
        self._battle_start_sec: Optional[float] = None
        self._elixir_leak_sec: float = 0.0
        self._prev_leak_ts: Optional[float] = None

        if self.use_vlm_card_id:
            try:
                from src.observer.vlm_client import QwenVLMClient
                self._vlm_client = QwenVLMClient()
            except Exception as e:
                print(f"[!] VLM init warning (continuing CV-only): {e}")

    def _maybe_trigger_vlm_card_id(
        self,
        frame: np.ndarray,
        detection: Dict[str, Any],
        player_elixir: float = 0.0,
        opp_elixir: float = 5.0,
        hand_slots: Optional[List[Dict[str, Any]]] = None,
        enemies: Optional[List[Dict[str, Any]]] = None,
        opp_intel: Optional[Dict[str, Any]] = None,
        elapsed_sec: float = 0.0,
    ):
        """
        Asynchronously runs the configured Qwen3-VL Brain (`qwen/qwen3-vl-30b-a3b-instruct`)
        every ~2.2s (skip-on-busy) comparing OUR hand + elixir vs OPPONENT hand + elixir
        to decide positive-trade cheap counters vs overspend punish windows.
        """
        if not self._vlm_client or self._vlm_client.is_busy():
            return
        now = time.time()
        if (now - self._last_vlm_ts) < 2.2:
            return
        self._last_vlm_ts = now

        oi = opp_intel or {}
        telemetry = {
            "player_elixir": round(float(player_elixir), 1),
            "opp_elixir_est": round(float(opp_elixir), 1),
            "opp_elixir_range": oi.get("opp_elixir_range", [0.0, 10.0]),
            "elixir_advantage": round(float(player_elixir) - float(opp_elixir), 1),
            "opp_estimated_hand": oi.get("opp_hand_archetypes", []),
            "opp_known_cards": oi.get("opp_known_cards", []),
            "opp_has_rusher_in_hand": bool(oi.get("opp_has_rusher", False)),
            "opp_has_building_counter": bool(oi.get("opp_has_building_counter", False)),
            "next_card": (detection.get("next_card") or {}).get("card_name"),
            "hand": [
                {
                    "slot": s.get("slot"),
                    "card": s.get("card_name"),
                    "cost": s.get("card_cost"),
                    "playable": s.get("playable"),
                }
                for s in (hand_slots or [])
            ],
            "enemies_detected": [
                {"col": e.get("col"), "row": e.get("row"), "our_half": e.get("in_our_half")}
                for e in (enemies or [])[:6]
            ],
            "towers": detection.get("tower_states", {}),
        }

        def worker(img_copy: np.ndarray, telem_copy: Dict[str, Any], elap: float):
            ok, raw_resp, meta = self._vlm_client.analyze_tactical_state(img_copy, telem_copy)
            if not ok:
                return
            try:
                cleaned = raw_resp.strip()
                if "```" in cleaned:
                    parts = cleaned.split("```")
                    cleaned = max(parts, key=len)
                if cleaned.lstrip().startswith("json"):
                    cleaned = cleaned.lstrip()[4:].strip()
                start_i = cleaned.find("{")
                end_i = cleaned.rfind("}")
                if start_i >= 0 and end_i > start_i:
                    cleaned = cleaned[start_i : end_i + 1]
                parsed = json.loads(cleaned)

                card_updates = {}
                hc = parsed.get("hand_cards") or {}
                if isinstance(hc, dict):
                    for k, v in hc.items():
                        if str(k).isdigit() and v:
                            card_updates[int(k)] = str(v).lower().replace(" ", "_")

                opp_seen = parsed.get("opp_cards_seen") or []
                if isinstance(opp_seen, list) and opp_seen:
                    self.opp_tracker.ingest_vlm_cards(opp_seen, elap, elixir_tracker=self.tracker)

                advice = {
                    "strategy_mode": parsed.get("strategy_mode"),
                    "allow_overspend": bool(parsed.get("allow_overspend", False)),
                    "max_defense_cost": parsed.get("max_defense_cost"),
                    "preferred_lane": parsed.get("preferred_lane"),
                    "elixir_floor": parsed.get("elixir_floor"),
                    "anti_air_needed": bool(parsed.get("anti_air_needed", False)),
                    "recommended_card": parsed.get("recommended_card"),
                    "opp_cards_seen": opp_seen if isinstance(opp_seen, list) else [],
                    "threats_summary": parsed.get("threats_summary"),
                    "reasoning": parsed.get("reasoning"),
                    "vlm_latency_ms": round(float(meta.get("latency_ms", 0.0)), 1),
                    "model": meta.get("model"),
                }
                with self._vlm_lock:
                    if card_updates:
                        self._vlm_card_map.update(card_updates)
                    self._vlm_tactical_advice = advice
                    self._vlm_tactical_ts = time.time()
            except Exception:
                pass

        threading.Thread(target=worker, args=(frame.copy(), telemetry, elapsed_sec), daemon=True).start()

    def _detect_menu_button(
        self, frame: np.ndarray, detection: Dict[str, Any]
    ) -> Optional[Tuple[str, Tuple[float, float]]]:
        """
        Finds safe menu CTA buttons inside the active Clash Royale viewport when auto_battle=True:
          1) Lobby yellow/gold 'Battle' button
          2) Post-match / popup bright-blue 'OK' button near bottom center
        """
        vx1, vy1, vx2, vy2 = detection.get("viewport_px", [0, 0, frame.shape[1], frame.shape[0]])
        vp = frame[vy1:vy2, vx1:vx2]
        if vp.size == 0:
            return None
        vh, vw = vp.shape[:2]
        screen = detection.get("screen")

        if screen == "lobby":
            y1, y2 = int(0.56 * vh), int(0.76 * vh)
            x1, x2 = int(0.24 * vw), int(0.76 * vw)
            crop = vp[y1:y2, x1:x2]
            if crop.size > 0:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                gold = cv2.inRange(hsv, np.array([15, 120, 140]), np.array([35, 255, 255]))
                cnts, _ = cv2.findContours(gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    best = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(best) >= 0.04 * crop.shape[0] * crop.shape[1]:
                        bx, by, bw, bh = cv2.boundingRect(best)
                        cx = (x1 + bx + bw * 0.5) / float(vw)
                        cy = (y1 + by + bh * 0.5) / float(vh)
                        return ("lobby_battle", (cx, cy))
            return ("lobby_battle", (0.50, 0.67))

        if screen == "non_game":
            # Look specifically for the Clash Royale bottom-center bright-blue OK button
            y1, y2 = int(0.89 * vh), int(0.99 * vh)
            x1, x2 = int(0.32 * vw), int(0.68 * vw)
            crop = vp[y1:y2, x1:x2]
            if crop.size > 0:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                blue_btn = cv2.inRange(hsv, np.array([96, 130, 150]), np.array([118, 255, 255]))
                cnts, _ = cv2.findContours(blue_btn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    best = max(cnts, key=cv2.contourArea)
                    bx, by, bw, bh = cv2.boundingRect(best)
                    area_ratio = cv2.contourArea(best) / float(max(1, crop.shape[0] * crop.shape[1]))
                    if area_ratio >= 0.18 and bw >= 0.25 * crop.shape[1]:
                        cx = (x1 + bx + bw * 0.5) / float(vw)
                        cy = (y1 + by + bh * 0.5) / float(vh)
                        return ("post_match_ok", (cx, cy))

        return None

    def process_frame(
        self,
        frame: np.ndarray,
        elapsed_sec: float = 0.0,
        client_origin: Tuple[int, int] = (0, 0),
        tick_steps: int = 1,
        client_size: Optional[Tuple[int, int]] = None,
        hwnd: Optional[int] = None,
        preview_every_n: int = 1,
        t_capture: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs one full cycle: Perception -> Gate -> Plan -> Dyadic Schedule -> Safe Execute.
        Tracks 3-stage latency timestamps: t_capture -> t_decision -> t_tap.
        """
        t0 = time.perf_counter()
        cap_ts = t_capture if t_capture is not None else t0
        self._frame_counter += 1
        t_perc0 = time.perf_counter()
        detection = self.detector.detect_components(frame)
        perception_ms = (time.perf_counter() - t_perc0) * 1000.0

        # Debounce battle entry across >=3 consecutive battle_arena frames (eliminates 1-frame lobby loading clicks like t=6.8s)
        raw_battle_now = detection["screen"] == "battle_arena" and detection.get("screen_confidence", 0.0) >= 0.88
        if raw_battle_now:
            self._battle_streak += 1
        else:
            self._battle_streak = 0
        in_battle_now = self._battle_streak >= 3
        if in_battle_now and not self._in_battle_prev:
            self._battle_start_sec = elapsed_sec
            self._elixir_leak_sec = 0.0
            self._prev_leak_ts = elapsed_sec
            self.tracker = ElixirTracker(initial_elixir=5.0)
            self.tracker.last_timestamp = 0.0
            self.opp_tracker.reset()
        elif not in_battle_now:
            self._battle_start_sec = None
            self._prev_leak_ts = None
        self._in_battle_prev = in_battle_now

        match_sec = max(0.0, elapsed_sec - self._battle_start_sec) if self._battle_start_sec is not None else elapsed_sec

        # Advance opponent elixir estimate heuristically alongside real bar read
        if in_battle_now:
            self.tracker.advance_time(match_sec)
        player_elixir = float(detection.get("elixir_estimate", 0.0))

        # Accumulate elixir leak time when capped at >= 9.8 in battle
        if in_battle_now and self._prev_leak_ts is not None:
            dt_leak = max(0.0, elapsed_sec - self._prev_leak_ts)
            if player_elixir >= 9.8:
                self._elixir_leak_sec += dt_leak
        self._prev_leak_ts = elapsed_sec if in_battle_now else None

        # Prefer fast CV template card names; fall back to async VLM map if CV returned placeholder
        with self._vlm_lock:
            card_map = dict(self._vlm_card_map)
        hand_slots = []
        for slot_info in detection.get("hand_slots", []):
            s_num = slot_info["slot"]
            enriched = dict(slot_info)
            cv_name = str(enriched.get("card_name") or f"slot_{s_num}")
            if cv_name.startswith("slot_") and s_num in card_map:
                enriched["card_name"] = card_map[s_num]
            else:
                enriched["card_name"] = cv_name
            hand_slots.append(enriched)

        enemies = detection.get("arena_entities", {}).get("enemies", [])
        if in_battle_now:
            opp_intel = self.opp_tracker.update_from_frame(enemies, match_sec, self.tracker)
        else:
            opp_intel = self.opp_tracker.get_intel(self.tracker, active_enemies=[])
        opp_intel["elixir_leak_warning"] = bool(in_battle_now and player_elixir >= 9.2)
        opp_intel["elixir_leak_sec"] = round(self._elixir_leak_sec, 2)
        opp_elixir = float(opp_intel["opp_elixir"])

        urgent_threats = [e for e in enemies if e.get("in_our_half") and e.get("row", 0) >= 18]
        if len(urgent_threats) > self._last_enemy_count:
            self.scheduler.trigger_emergency_interrupt(reason="enemy_crossed_bridge")
        self._last_enemy_count = len(urgent_threats)

        hand_names = [s["card_name"] for s in hand_slots]
        decision = {"action": "wait", "slot": None, "target_tile": None, "reasoning": f"Screen is {detection['screen']}"}
        active_qwen_advice = None
        if in_battle_now:
            if self.use_vlm_card_id:
                self._maybe_trigger_vlm_card_id(
                    frame,
                    detection,
                    player_elixir=player_elixir,
                    opp_elixir=opp_elixir,
                    hand_slots=hand_slots,
                    enemies=enemies,
                    opp_intel=opp_intel,
                    elapsed_sec=elapsed_sec,
                )
                with self._vlm_lock:
                    if self._vlm_tactical_advice and (time.time() - self._vlm_tactical_ts) <= 7.0:
                        active_qwen_advice = dict(self._vlm_tactical_advice)

            decision = self.planner.decide_next_action(
                current_elixir=player_elixir,
                opponent_estimated_elixir=opp_elixir,
                hand_cards=hand_names,
                arena_threats=enemies,
                hand_slots=hand_slots,
                tower_states=detection.get("tower_states"),
                opp_intel=opp_intel,
                next_card=detection.get("next_card"),
                qwen_advice=active_qwen_advice,
            )
            self.scheduler.submit_decision(decision)
        t_decision = time.perf_counter()

        # Step Dyadic 2^x Opportunity Scheduler (snap emergency plays immediately via >= 4 ticks)
        effective_steps = max(4, tick_steps) if decision.get("emergency") else max(1, tick_steps)
        fired_action = None
        telem = {}
        for _ in range(effective_steps):
            fired, act, telem = self.scheduler.tick()
            if fired and act is not None:
                fired_action = act
                break

        is_modal_false_battle = (
            detection["screen"] == "battle_arena"
            and detection["screen_confidence"] < 0.88
            and not detection.get("droplet_present")
            and player_elixir == 0.0
        )

        exec_result = {"status": "SCHEDULE_WAIT", "executed": False, "reason": "Waiting for next 2^x opportunity gate"}
        if (detection["screen"] != "battle_arena" or is_modal_false_battle) and self.auto_battle:
            menu_btn = self._detect_menu_button(frame, detection)
            if menu_btn is not None:
                btn_label, btn_vp_xy = menu_btn
                decision = {
                    "action": f"click_{btn_label}",
                    "slot": None,
                    "target_tile": None,
                    "reasoning": f"Auto-battle clicking {btn_label}",
                }
                exec_result = self.executor.execute_menu_click(
                    btn_label,
                    btn_vp_xy,
                    detection,
                    client_origin=client_origin,
                    client_size=client_size,
                    hwnd=hwnd,
                )
        elif in_battle_now and fired_action is not None:
            exec_result = self.executor.execute(
                fired_action, detection, client_origin=client_origin, client_size=client_size, hwnd=hwnd
            )
            if exec_result.get("executed") and fired_action.get("slot"):
                self.planner.confirm_executed(fired_action)
                self.detector.card_matcher.invalidate_slot(int(fired_action["slot"]))
                played_slot = next((s for s in hand_slots if s["slot"] == fired_action["slot"]), None)
                if played_slot and played_slot.get("card_cost"):
                    self.tracker.register_card_played("player", int(played_slot["card_cost"]), match_sec)
        elif decision.get("action") != "wait" and tick_steps == 1:
            # Expose shadow preview coordinates even between gates for UI/logging
            coords = (
                self.executor.compute_screen_targets(
                    detection,
                    int(decision["slot"]),
                    decision["target_tile"],
                    client_origin=client_origin,
                    client_size=client_size,
                )
                if decision.get("slot") and decision.get("target_tile")
                else None
            )
            exec_result["planned_preview"] = coords

        human_play = None
        if in_battle_now and self._prev_hand_slots:
            elixir_drop = self._prev_player_elixir - player_elixir
            if (elapsed_sec - self._last_human_play_ts) >= 0.45:
                for hs in hand_slots:
                    s_num = hs["slot"]
                    prev = self._prev_hand_slots.get(s_num, {})
                    became_cooldown = (
                        prev.get("slot_state") != "cooldown_empty"
                        and hs.get("slot_state") == "cooldown_empty"
                    )
                    card_swapped_with_drop = (
                        prev.get("card_name")
                        and hs.get("card_name")
                        and not str(prev["card_name"]).startswith("slot_")
                        and not str(hs["card_name"]).startswith("slot_")
                        and prev["card_name"] != hs["card_name"]
                        and elixir_drop >= 0.8
                    )
                    if became_cooldown or card_swapped_with_drop:
                        self._last_human_play_ts = elapsed_sec
                        human_play = {
                            "slot": s_num,
                            "card": prev.get("card_name") or hs.get("card_name"),
                            "cost": prev.get("card_cost"),
                            "elixir_at_play": round(self._prev_player_elixir, 1),
                            "threats_count": len(enemies),
                            "our_half_threats": sum(1 for e in enemies if e.get("in_our_half")),
                        }
                        break
        if in_battle_now:
            self._prev_hand_slots = {hs["slot"]: dict(hs) for hs in hand_slots}
            self._prev_player_elixir = player_elixir
        else:
            self._prev_hand_slots = {}
            self._prev_player_elixir = 0.0

        gate_unlocked, gate_reason, _ = self.gate.check()
        mode_tag = (
            "HUMAN_BASELINE"
            if self.human_observe
            else ("ARMED_LIVE" if (gate_unlocked and self.live_armed) else ("SHADOW_READY" if gate_unlocked else "GATE_LOCKED"))
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        t_tap_done = float(exec_result.get("t_tap_done") or exec_result.get("t_tap_sent") or time.perf_counter())
        c2d_ms = round((t_decision - cap_ts) * 1000.0, 2)
        c2t_ms = round((t_tap_done - cap_ts) * 1000.0, 2)

        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": mode_tag,
            "gate_unlocked": gate_unlocked,
            "gate_reason": gate_reason,
            "screen": detection["screen"],
            "confidence": detection["screen_confidence"],
            "elixir": player_elixir,
            "elixir_leak_sec": round(self._elixir_leak_sec, 2),
            "elixir_leak_warning": bool(opp_intel.get("elixir_leak_warning", False)),
            "opponent_elixir_est": opp_elixir,
            "opponent_intel": opp_intel,
            "next_card": detection.get("next_card"),
            "layout": detection["layout"],
            "viewport_px": detection["viewport_px"],
            "hand_slots": hand_slots,
            "enemy_threats_count": len(enemies),
            "enemy_threats": enemies[:5],
            "decision": decision,
            "qwen_brain": active_qwen_advice,
            "human_play": human_play,
            "scheduler_telemetry": telem,
            "execution": exec_result,
            "pipeline_latency_ms": latency_ms,
            "latency_breakdown": {
                "perception_ms": round(perception_ms, 2),
                "capture_to_decision_ms": c2d_ms,
                "capture_to_tap_ms": c2t_ms,
            },
        }

        if self.state_export_path and (
            preview_every_n <= 1
            or exec_result.get("executed")
            or human_play is not None
            or (self._frame_counter % max(1, preview_every_n) == 0)
        ):
            os.makedirs(os.path.dirname(self.state_export_path) or ".", exist_ok=True)
            with open(self.state_export_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

        should_save_preview = (
            self.save_preview_path
            and (
                preview_every_n <= 1
                or exec_result.get("executed")
                or (self._frame_counter % preview_every_n == 0)
            )
        )
        annotated = None
        if should_save_preview or self.recorder is not None:
            annotated = self.detector.annotate_frame(
                frame,
                detection,
                planned_action=decision if decision.get("action") != "wait" else None,
                gate_status=mode_tag,
            )
        if should_save_preview and annotated is not None:
            os.makedirs(os.path.dirname(self.save_preview_path) or ".", exist_ok=True)
            cv2.imwrite(self.save_preview_path, annotated)

        if self.recorder is not None and annotated is not None:
            self.recorder.record_step(frame, annotated, payload, elapsed_sec=elapsed_sec)

        return payload

    def run_replay(self, replay_dir: str, delay_sec: float = 0.05) -> List[Dict[str, Any]]:
        """Runs the full pipeline over a sorted directory of frames (always dry-run safe)."""
        exts = (".png", ".jpg", ".jpeg")
        files = sorted(f for f in os.listdir(replay_dir) if f.lower().endswith(exts))
        if not files:
            print(f"[-] No frames found in {replay_dir}")
            return []

        gate_ok, gate_reason, _ = self.gate.check()
        print("=" * 75)
        print(f"CLASH IQ AUTONOMOUS REPLAY | Frames: {len(files)} | Gate: {'UNLOCKED' if gate_ok else 'LOCKED'}")
        print(f"Gate Reason: {gate_reason}")
        print("=" * 75)

        outputs = []
        elapsed = 0.0
        for idx, fname in enumerate(files):
            img = cv2.imread(os.path.join(replay_dir, fname))
            if img is None:
                continue
            # Advance up to 2 frames per replay step so Frame-2 Reflex gate can fire on battle frames
            res = self.process_frame(img, elapsed_sec=elapsed, client_origin=(0, 0), tick_steps=2)
            outputs.append(res)
            dec = res["decision"]
            ex = res["execution"]
            print(
                f"[{idx+1:02d}/{len(files):02d}] {fname:<14s} | {res['screen']:<12s} ({res['confidence']:.2f}) "
                f"| Elx:{res['elixir']:4.1f} | Threats:{res['enemy_threats_count']} "
                f"| Plan:{dec['action']:<20s} Tile:{str(dec.get('target_tile')):<9s} "
                f"| Exec:{ex['status']} ({res['pipeline_latency_ms']}ms)"
            )
            elapsed += 0.5
            if delay_sec > 0:
                time.sleep(delay_sec)

        if self.recorder is not None:
            rec_summary = self.recorder.close()
            print(f"\n[RECORDING SAVED] {rec_summary}")
        print(f"\n[DONE] Replay Summary -> Executor Stats: {self.executor.stats}")
        return outputs

    def run_live(self, tick_interval_sec: float = 0.066, max_duration_sec: float = 0.0):
        """
        Attaches to live Clash Royale window via background LatestFrameCapture thread
        (drops stale queued frames) and runs the reflex decision loop at 10-15 Hz.
        """
        import ctypes
        from src.observer.window_guard import WindowGuard, RECT
        from src.observer.capture_engine import LatestFrameCapture

        gate_ok, gate_reason, _ = self.gate.check()
        target_period = max(0.033, min(0.10, float(tick_interval_sec) if tick_interval_sec > 0 else (1.0 / 15.0)))
        target_hz = round(1.0 / target_period, 1)
        print("=" * 75)
        print("CLASH IQ AUTONOMOUS LIVE ENGINE" + (" [HUMAN BASELINE OBSERVER]" if self.human_observe else ""))
        print(f"  Accuracy Gate (>= {self.gate.min_accuracy_pct:.0f}%): {'UNLOCKED' if gate_ok else 'LOCKED'} -> {gate_reason}")
        print(f"  Live Input Injection:  {'DISABLED (Observing Human)' if self.human_observe else ('ARMED (--live)' if (gate_ok and self.live_armed) else 'SHADOW DRY-RUN (Zero Clicks)')}")
        print(f"  Decision Cadence:      {target_hz:.1f} Hz (Capture thread @ 30 Hz latest-slot)")
        print(f"  Qwen Tactical Brain:   {self._vlm_client.model if self._vlm_client else 'DISABLED (CV-Only Default)'}")
        print(f"  Auto-Battle Menu Nav:  {'ENABLED (--auto-battle)' if self.auto_battle else 'DISABLED (Battle-only)'}")
        print("  Safety Killswitch:     Press F12 or ESC anytime to HALT immediately")
        print("=" * 75)

        guard = WindowGuard(target_keywords=("clash royale",))

        if not guard.launch_or_find(timeout_sec=None) or not guard.hwnd:
            print(f"[-] Clash Royale window not found. Reason: {guard.halt_reason}")
            guard.close()
            return

        # Focus window and allow brief startup geometry settle before locking baseline_rect
        self.executor.focus_window(guard.hwnd)
        time.sleep(1.5)
        settled_rect = RECT()
        if ctypes.windll.user32.GetWindowRect(guard.hwnd, ctypes.byref(settled_rect)):
            guard.baseline_rect = (
                settled_rect.left,
                settled_rect.top,
                settled_rect.right,
                settled_rect.bottom,
            )
            guard.baseline_client_rect = guard.get_client_screen_rect()

        print(f"[+] Attached to HWND {guard.hwnd} | Baseline: {guard.baseline_rect}")
        if self.recorder is not None:
            print(f"[+] Recording gameplay & decisions to: {self.recorder.session_dir}")

        cap_thread = LatestFrameCapture(
            hwnd=guard.hwnd,
            rect_provider=guard.get_client_screen_rect,
            target_capture_hz=30.0,
            min_variance_threshold=15.0,
            mask_pii=False,
        ).start()

        start_ts = time.time()
        last_loop_ts = start_ts

        try:
            while not guard.is_halted:
                tick_t0 = time.perf_counter()
                now_loop_ts = time.time()
                if max_duration_sec > 0 and (now_loop_ts - start_ts) >= max_duration_sec:
                    print(f"\n[DONE] Reached max duration ({max_duration_sec:.0f}s).")
                    break

                u32 = ctypes.windll.user32
                if not guard.hwnd or not u32.IsWindow(guard.hwnd):
                    print("\n[HALT] Clash Royale window closed.")
                    break
                if u32.IsIconic(guard.hwnd):
                    print("\n[HALT] Clash Royale window minimized.")
                    break

                pkt = cap_thread.get_latest(timeout_sec=0.08)
                if pkt is None:
                    continue
                frame = pkt["frame"]
                rect = pkt["rect"] or guard.get_client_screen_rect()
                if frame is None or not rect:
                    continue

                dt_loop = max(0.016, now_loop_ts - last_loop_ts)
                last_loop_ts = now_loop_ts
                dyn_tick_steps = max(2, min(16, int(round(dt_loop * 60.0))))

                res = self.process_frame(
                    frame,
                    elapsed_sec=now_loop_ts - start_ts,
                    client_origin=(rect[0], rect[1]),
                    tick_steps=dyn_tick_steps,
                    client_size=(rect[2] - rect[0], rect[3] - rect[1]),
                    hwnd=guard.hwnd,
                    preview_every_n=30,
                    t_capture=pkt["t_capture"],
                )
                dec = res["decision"]
                ex = res["execution"]
                c_stats = cap_thread.stats()
                lat = res.get("latency_breakdown", {})
                if ex.get("executed"):
                    print(
                        f"\n[EXEC {time.strftime('%H:%M:%S')}] {ex['status']} | {ex.get('reason')} "
                        f"| c2t={lat.get('capture_to_tap_ms')}ms | drop={c_stats['dropped_frames_pct']}%"
                    )
                sys.stdout.write(
                    f"\r[{time.strftime('%H:%M:%S')}] {res['screen']} ({res['confidence']*100:.0f}%) "
                    f"| Elx:{res['elixir']:4.1f} | Threats:{res['enemy_threats_count']} "
                    f"| Plan:{dec['action']} Slot:{dec.get('slot')} "
                    f"| c2d:{lat.get('capture_to_decision_ms', 0):.0f}ms drop:{c_stats['dropped_frames_pct']:.0f}% "
                    f"| {ex['status']}     "
                )
                sys.stdout.flush()
                loop_elapsed = time.perf_counter() - tick_t0
                sleep_left = target_period - loop_elapsed
                if sleep_left > 0.001:
                    time.sleep(sleep_left)
        except KeyboardInterrupt:
            print("\n[STOP] Stopped by user.")
        finally:
            final_cap_stats = cap_thread.stats()
            cap_thread.stop()
            guard.close()
            if self.recorder is not None:
                rec_summary = self.recorder.close()
                print(f"\n[RECORDING SAVED] {rec_summary}")
                try:
                    from src.learning.exchange_ledger import ExchangeLedgerBuilder
                    ev_path = rec_summary.get("events_log")
                    sess_dir = rec_summary.get("session_dir")
                    if ev_path and os.path.exists(ev_path):
                        ex_out = os.path.join(sess_dir, "exchanges.jsonl") if sess_dir else None
                        reserve = float(self.planner.learning_tables.management_params.get("reserve_floor", 2.0))
                        ledger = ExchangeLedgerBuilder().build_from_events(ev_path, output_jsonl_path=ex_out, reserve_floor=reserve)
                        upd = self.planner.learning_tables.update_from_exchanges(
                            ledger["exchanges"], metrics=ledger["metrics"], match_id=ledger["match_id"]
                        )
                        print(f"[LEARNING LEDGER] {ledger['metrics']} | updates={upd['counter_updates']}/{upd['placement_updates']}")
                except Exception as exc:
                    print(f"[LEARNING LEDGER WARN] Failed post-match update: {exc}")
            print(f"\n[CAPTURE STATS] {final_cap_stats}")
            print(f"[SHUTDOWN] Stats: {self.executor.stats}")
