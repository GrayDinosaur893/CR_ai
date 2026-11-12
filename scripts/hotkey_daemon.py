"""
CLASH IQ — Global Hotkey Daemon.
Runs silently in background. Listens for global keypresses across Windows:
- F9  : Launch & Start 2^x Autonomous Bot
- F10 : Live Screenshot & Component Detection
- ESC : Safe Abort
"""

import sys
import os
import time
import ctypes

user32 = ctypes.windll.user32

VK_F9 = 0x78
VK_F10 = 0x79
VK_ESCAPE = 0x1B

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_hotkey_daemon():
    print("=" * 65)
    print("     CLASH IQ GLOBAL HOTKEY DAEMON INITIALIZED")
    print("=" * 65)
    print("  * [F9]  -> Launch & Start 2^x Autonomous Bot")
    print("  * [F10] -> Take Live Screenshot & Detect Components")
    print("  * [ESC] -> Emergency Abort / Stop")
    print("=" * 65)
    print("[WAITING] Listening for hotkeys globally... (Press Ctrl+C to close daemon)")

    try:
        while True:
            # Check F9
            if user32.GetAsyncKeyState(VK_F9) & 0x8000:
                print("\n>>> [HOTKEY DETECTED: F9] Starting Autonomous Bot...")
                time.sleep(0.5)
                from scripts.cr_agent import AutonomousCRAgent
                agent = AutonomousCRAgent()
                agent.launch_and_focus()
                agent.enter_friendly_battle()
                agent.run_autonomous_battle()
                print("\n[READY] Resumed hotkey listening...")

            # Check F10
            elif user32.GetAsyncKeyState(VK_F10) & 0x8000:
                print("\n>>> [HOTKEY DETECTED: F10] Grabbing Screenshot & Detecting Components...")
                time.sleep(0.5)
                from scripts.run_perception_and_export_sheet import run_pipeline
                run_pipeline()
                print("\n[READY] Resumed hotkey listening...")

            # Check ESC
            elif user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
                print("\n>>> [HOTKEY DETECTED: ESC] Daemon stopped.")
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[EXIT] Hotkey daemon terminated.")

if __name__ == "__main__":
    run_hotkey_daemon()
