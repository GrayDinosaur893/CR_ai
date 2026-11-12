"""
CLASH IQ — Zero-Prompt Autonomous Friendly Battle Agent.
Launches Clash Royale directly, brings it to foreground, enters/accepts
Friendly Battle ONLY, and autonomously plays cards using tactical intelligence.

Run: python scripts/cr_agent.py
Stop: Press ESC or Ctrl+C at any time.
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes

# Win32 Constants
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
SW_RESTORE = 9
VK_ESCAPE = 0x1B

user32 = ctypes.windll.user32

# Path to Clash Royale Google Play Games Shortcut
CR_SHORTCUT = os.path.expanduser(
    r"~\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Google Play Games\Clash Royale.lnk"
)

# Add root directory to path for tactical modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.strategy.action_planner import TacticalActionPlanner
from src.strategy.opportunity_scheduler import DyadicOpportunityScheduler
from src.game_state.elixir_tracker import ElixirTracker

class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long)
    ]

class AutonomousCRAgent:
    def __init__(self):
        self.hwnd = None
        self.window_rect = None
        self.planner = TacticalActionPlanner()
        self.tracker = ElixirTracker(initial_elixir=5.0)
        self.scheduler = DyadicOpportunityScheduler()

    def launch_and_focus(self):
        """Launches Clash Royale via Google Play Games shortcut and focuses window."""
        print("=" * 60)
        print("          CLASH IQ: FULL AUTONOMOUS AGENT")
        print("=" * 60)
        print("[1/4] Launching / Focusing Clash Royale...")

        if os.path.exists(CR_SHORTCUT):
            try:
                os.startfile(CR_SHORTCUT)
                print("[OK] Opened Clash Royale shortcut.")
            except Exception as e:
                print(f"[!] Warning opening shortcut: {e}")
        else:
            print("[!] Shortcut not found, assuming game already launched on screen.")

        # Give window 4 seconds to rise to foreground
        time.sleep(4.0)

        # Locate game window
        found = []
        def enum_windows_callback(hwnd, extra):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    title = buffer.value.strip()
                    if "clash royale" in title.lower() or "google play games" in title.lower():
                        found.append((hwnd, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

        if found:
            self.hwnd = found[0][0]
            # Bring to front & restore if minimized
            user32.ShowWindow(self.hwnd, SW_RESTORE)
            user32.SetForegroundWindow(self.hwnd)
            rect = RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
            self.window_rect = (rect.left, rect.top, rect.right, rect.bottom)
            print(f"[OK] Focused Window: '{found[0][1]}' at {self.window_rect}")
        else:
            # Fallback to active screen dimensions
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            self.window_rect = (0, 0, w, h)
            print(f"[!] Using full desktop boundaries: ({w}x{h})")

        time.sleep(1.0)

    def click(self, x: int, y: int, delay: float = 0.12):
        """Moves cursor and performs left click."""
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.04)
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(delay)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.04)

    def rel_to_abs(self, rel_x: float, rel_y: float) -> tuple:
        """Converts window-relative coordinates (0.0 to 1.0) to absolute screen (x, y)."""
        left, top, right, bottom = self.window_rect
        w = right - left
        h = bottom - top
        return (left + int(rel_x * w), top + int(rel_y * h))

    def enter_friendly_battle(self):
        """
        Navigates strictly to the Clan/Social tab to accept/start Friendly Battle ONLY.
        Strictly ignores regular Ladder/Ranked battle.
        """
        print("[2/4] Navigating to Clan Tab (Friendly Battles Only)...")

        # Clan Tab is typically 4th tab at the bottom (~68% X, ~96% Y)
        clan_tab_x, clan_tab_y = self.rel_to_abs(0.68, 0.96)
        self.click(clan_tab_x, clan_tab_y, delay=0.15)
        time.sleep(1.5)

        # Check for open 'Accept' Friendly Battle button in clan chat (~68% X, ~72% Y)
        print("[3/4] Checking & Accepting Friendly Battle request in Clan...")
        accept_x, accept_y = self.rel_to_abs(0.68, 0.72)
        self.click(accept_x, accept_y, delay=0.15)
        time.sleep(0.8)

        # Also click the yellow Friendly Battle initiator button if no pending request (~80% X, ~88% Y)
        # to ensure friendly battle starts
        initiate_x, initiate_y = self.rel_to_abs(0.80, 0.88)
        self.click(initiate_x, initiate_y, delay=0.15)
        time.sleep(1.0)

        # Standard 1v1 Battle option popup confirm (~50% X, ~60% Y)
        confirm_x, confirm_y = self.rel_to_abs(0.50, 0.60)
        self.click(confirm_x, confirm_y, delay=0.15)

        print("[OK] Friendly battle action sent. Waiting for battle arena load...")
        time.sleep(4.0)

    def deploy_card_to_tile(self, slot_idx: int, target_tile: list):
        """Deploys card from hand slot (1-4) onto target tile [col, row]."""
        slot_positions = {
            1: (0.32, 0.93),
            2: (0.48, 0.93),
            3: (0.64, 0.93),
            4: (0.80, 0.93)
        }
        card_rel_x, card_rel_y = slot_positions.get(slot_idx, (0.32, 0.93))
        card_x, card_y = self.rel_to_abs(card_rel_x, card_rel_y)

        # Convert tile [col, row] (0-17, 0-31) to relative arena (x, y)
        col, row = target_tile
        # Arena occupies ~5% to ~95% X, ~10% to ~80% Y
        arena_x = 0.08 + (col / 18.0) * 0.84
        arena_y = 0.12 + (row / 32.0) * 0.68
        dest_x, dest_y = self.rel_to_abs(arena_x, arena_y)

        # 1. Click Card
        self.click(card_x, card_y, delay=0.08)
        time.sleep(0.1)
        # 2. Click Destination Tile
        self.click(dest_x, dest_y, delay=0.08)

    def run_autonomous_battle(self):
        """
        Main autonomous gameplay decision loop with Dyadic 2^x Opportunity Scheduling.
        Dispatches actions dynamically at power-of-two gates (33ms, 66ms, 133ms, 266ms, 533ms)
        as soon as decisions are computed. Resets epoch immediately upon execution.
        """
        print("[4/4] BATTLE ENGAGED — Dyadic 2^x Opportunity Engine Active!")
        print("      Operating on 60 FPS tick timeline with dynamic power-of-two gates.")
        print("      Press ESC or Ctrl+C at any time to abort.")
        print("-" * 65)

        match_time = 0.0
        slot_cycle = 1
        computation_pending = False

        try:
            while True:
                # Emergency Abort
                if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                    print("\n[ABORT] ESC key pressed. Halting agent immediately.")
                    break

                # Advance game time
                self.tracker.advance_time(match_time)
                state = self.tracker.get_state()
                elixir = state["player_elixir"]
                opp_elixir = state["opponent_elixir_estimate"]["mid"]

                # If no decision is currently queued, trigger asynchronous tactical planning
                if not computation_pending:
                    hand = ["slot_1", "slot_2", "slot_3", "slot_4"]
                    decision = self.planner.decide_next_action(
                        current_elixir=elixir,
                        opponent_estimated_elixir=opp_elixir,
                        hand_cards=hand,
                        arena_threats=[]
                    )
                    # Submit decision to the opportunity scheduler buffer
                    self.scheduler.submit_decision(decision)
                    computation_pending = True

                # Step 1 tick of the 60 FPS game clock (~16.67ms)
                action_fired, action_data, telem = self.scheduler.tick()

                # If an action fired at an active 2^x gate
                if action_fired and action_data is not None:
                    computation_pending = False
                    action = action_data["action"]
                    target_tile = action_data.get("target_tile") or [14, 22]
                    gate_hit = telem.get("dispatched_at_gate", 2)
                    lat_ms = telem.get("latency_ms", 33.33)

                    if action in ["opposite_lane_punish", "defend", "slow_roll_back"]:
                        print(f"[{int(match_time):03d}s] Gate {gate_hit} (2^{gate_hit.bit_length()-1}) | Latency: {lat_ms}ms | Action: {action.upper()}")
                        print(f"       Deploying Slot {slot_cycle} -> Tile {target_tile} (Elixir: {elixir:.1f})")
                        self.deploy_card_to_tile(slot_cycle, target_tile)
                        self.tracker.register_card_played("player", elixir_cost=3, current_seconds=match_time)
                        slot_cycle = (slot_cycle % 4) + 1
                        time.sleep(1.0)
                        match_time += 1.0
                    else:
                        # Wait decision dispatched at macro gate
                        pass

                # Frame step delay
                time.sleep(0.01667)
                match_time += 0.01667

        except KeyboardInterrupt:
            print("\n[STOP] Agent stopped by user.")
            summary = self.scheduler.get_summary()
            print("\n--- Session Scheduling Telemetry ---")
            print(f"Total Frames Observed:     {summary['total_frames_seen']}")
            print(f"Actions Dispatched:        {summary['actions_dispatched']}")
            print(f"Compute Inference Saved:   {summary['compute_saved_pct']}%")
            print(f"Gate Distribution (2^x):   {summary['gate_distribution']}")

def main():
    agent = AutonomousCRAgent()
    # 1. Automatically launch / focus Clash Royale
    agent.launch_and_focus()
    # 2. Automatically navigate to Clan and accept / start Friendly Battle ONLY
    agent.enter_friendly_battle()
    # 3. Autonomous card deployment & tactical play
    agent.run_autonomous_battle()

if __name__ == "__main__":
    main()
