"""
CLASH IQ — Master Unified Simultaneous Pipeline.
Single command to launch the full simultaneous AI engine:
  - Multi-threaded Capture (~60 Hz)
  - Simultaneous Screen & In-Battle Perception (~20 Hz)
  - Live Real-Time ASCII Telemetry Stream (5 Hz)
  - Per-frame Geometry Invariant Guard
  - F12 Emergency Hard Killswitch (os._exit(0))
"""

import os
import sys

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.unified_engine import UnifiedClashEngine

def main():
    engine = UnifiedClashEngine()
    engine.run_simultaneous()

if __name__ == "__main__":
    main()
