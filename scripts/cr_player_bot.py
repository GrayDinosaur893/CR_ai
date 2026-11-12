"""
CLASH IQ — Autonomous Friendly Battle & Card Deployment Controller.
Run this script in your interactive terminal while Clash Royale is on screen.
Uses pure Windows Win32 API (built-in ctypes) with zero external dependencies.
"""

import sys
import os
import time
import ctypes
from ctypes import wintypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.strategy.opportunity_scheduler import DyadicOpportunityScheduler

# Win32 Constants
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
VK_ESCAPE = 0x1B

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [
        ('left', ctypes.c_long),
        ('top', ctypes.c_long),
        ('right', ctypes.c_long),
        ('bottom', ctypes.c_long)
    ]

class ClashRoyaleBot:
    def __init__(self, window_title_keyword: str = "Clash Royale"):
        self.keyword = window_title_keyword
        self.hwnd = None
        self.window_rect = None
        self.scheduler = DyadicOpportunityScheduler()

    def find_game_window(self) -> bool:
        """Finds the Clash Royale or Google Play Games PC window."""
        found = []

        def enum_windows_callback(hwnd, extra):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buffer = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buffer, length + 1)
                    title = buffer.value.strip()
                    if self.keyword.lower() in title.lower() or "google play games" in title.lower():
                        found.append((hwnd, title))
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.c_int)
        user32.EnumWindows(WNDENUMPROC(enum_windows_callback), 0)

        if found:
            self.hwnd = found[0][0]
            rect = RECT()
            user32.GetWindowRect(self.hwnd, ctypes.byref(rect))
            self.window_rect = (rect.left, rect.top, rect.right, rect.bottom)
            print(f"[OK] Found Window: '{found[0][1]}' at {self.window_rect}")
            return True
        else:
            print("[!] Could not auto-detect window title. Using full primary screen coordinates.")
            # Default to full screen if window title is hidden by crosvm
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            self.window_rect = (0, 0, w, h)
            return True

    def click(self, x: int, y: int, delay: float = 0.1):
        """Simulates an instant mouse click at absolute screen coordinates."""
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.05)
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(delay)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        time.sleep(0.05)

    def rel_to_abs(self, rel_x: float, rel_y: float) -> tuple:
        """Converts window-relative coordinates (0.0 to 1.0) to screen (x, y)."""
        left, top, right, bottom = self.window_rect
        w = right - left
        h = bottom - top
        abs_x = left + int(rel_x * w)
        abs_y = top + int(rel_y * h)
        return (abs_x, abs_y)

    def accept_friendly_battle(self):
        """Clicks the 'Battle' / 'Accept' button in clan chat."""
        # Typically located around center-right of clan chat tab
        btn_x, btn_y = self.rel_to_abs(0.68, 0.72)
        print(f"[*] Clicking Accept Friendly Battle button at ({btn_x}, {btn_y})...")
        self.click(btn_x, btn_y, delay=0.15)

    def deploy_card(self, slot_index: int, target_rel_x: float, target_rel_y: float):
        """
        Deploys card from slot (1, 2, 3, or 4) to an arena target.
        """
        slot_positions = {
            1: (0.32, 0.93),
            2: (0.48, 0.93),
            3: (0.64, 0.93),
            4: (0.80, 0.93)
        }

        if slot_index not in slot_positions:
            slot_index = 1

        card_rel_x, card_rel_y = slot_positions[slot_index]
        card_x, card_y = self.rel_to_abs(card_rel_x, card_rel_y)
        dest_x, dest_y = self.rel_to_abs(target_rel_x, target_rel_y)

        print(f"[*] Action: Selecting Card Slot {slot_index} -> Deploying to ({dest_x}, {dest_y})")
        # 1. Click card in hand
        self.click(card_x, card_y, delay=0.1)
        time.sleep(0.12)
        # 2. Click destination tile in arena
        self.click(dest_x, dest_y, delay=0.1)

    def start_autoplay_loop(self):
        """
        Deploys cards using Dyadic 2^x Opportunity Scheduling (33ms, 66ms, 133ms, 266ms, 533ms).
        Resets dynamic window instantly upon action to eliminate robotic timing.
        """
        print("\n" + "=" * 65)
        print("   CLASH IQ AUTONOMOUS BOT — DYADIC 2^x ENGINE ACTIVE")
        print("   [SAFEGUARD] Press ESC or Ctrl+C in this terminal to STOP")
        print("=" * 65)

        left_bridge = (0.28, 0.52)
        right_bridge = (0.72, 0.52)
        center_defend = (0.50, 0.65)
        back_king = (0.50, 0.82)
        drop_targets = [right_bridge, left_bridge, center_defend, right_bridge]

        current_slot = 1
        turn = 0
        computation_pending = False

        try:
            while True:
                if user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                    print("\n[STOP] ESC key detected! Stopping bot.")
                    break

                if not computation_pending:
                    target = drop_targets[turn % len(drop_targets)]
                    self.scheduler.submit_decision({
                        "slot": current_slot,
                        "target": target,
                        "turn": turn + 1
                    })
                    computation_pending = True

                action_fired, action_data, telem = self.scheduler.tick()

                if action_fired and action_data is not None:
                    computation_pending = False
                    slot = action_data["slot"]
                    tx, ty = action_data["target"]
                    gate = telem.get("dispatched_at_gate", 2)
                    lat = telem.get("latency_ms", 33.33)

                    print(f"\n[Turn {action_data['turn']}] Gate {gate} (2^{gate.bit_length()-1}) | Latency: {lat}ms -> Deploying Slot {slot}...")
                    self.deploy_card(slot, tx, ty)

                    current_slot = (current_slot % 4) + 1
                    turn += 1
                    time.sleep(2.0)

                time.sleep(0.01667)

        except KeyboardInterrupt:
            print("\n[STOP] Bot stopped by user.")
            summary = self.scheduler.get_summary()
            print("\n--- Session Scheduling Telemetry ---")
            print(f"Total Frames Observed:     {summary['total_frames_seen']}")
            print(f"Actions Dispatched:        {summary['actions_dispatched']}")
            print(f"Compute Inference Saved:   {summary['compute_saved_pct']}%")

if __name__ == "__main__":
    bot = ClashRoyaleBot()
    bot.find_game_window()

    print("\nSelect Mode:")
    print("1. Accept Friendly Battle (Clicks Accept button)")
    print("2. Start Auto-Play (Deploys cards autonomously)")
    print("3. Accept & Immediately Start Auto-Play")

    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    if choice == "1":
        time.sleep(2)
        bot.accept_friendly_battle()
    elif choice == "2":
        print("Switch to Clash Royale window now! Starting in 3 seconds...")
        time.sleep(3)
        bot.start_autoplay_loop()
    elif choice == "3":
        print("Accepting battle in 2 seconds...")
        time.sleep(2)
        bot.accept_friendly_battle()
        print("Starting auto-play in 3 seconds...")
        time.sleep(3)
        bot.start_autoplay_loop()
    else:
        print("Invalid choice.")
