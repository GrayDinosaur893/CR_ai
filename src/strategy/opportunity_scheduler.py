"""
CLASH IQ — Hierarchical Binary Exponential (2^x) Opportunity Scheduler.

Implements the user's innovative dynamic timing architecture:
1. Decomposes 60 frames into dyadic power blocks: 60 = 28 + 32 = 32 + 16 + 8 + 4
2. Dispatches decisions at powers-of-two execution gates:
   - Frame 2  = 33.33 ms (Reflex Gate)
   - Frame 4  = 66.67 ms (Micro Gate)
   - Frame 8  = 133.33 ms (Tactical Gate)
   - Frame 16 = 266.67 ms (Complex Evaluation Gate)
   - Frame 32 = 533.33 ms (Macro / Wait Gate)
3. Immediately executes as soon as a decision is ready at the next available opportunity.
4. Resets the window epoch instantly upon execution to prevent robotic fixed-interval rhythms.
5. Provides emergency preemption interrupt for sudden in-game threats.
"""

from typing import Dict, Any, List, Optional, Tuple
import time

class DyadicOpportunityScheduler:
    FRAME_DURATION_MS = 1000.0 / 60.0  # 16.667 ms per frame
    TOTAL_CYCLE_FRAMES = 60
    # Recursive dyadic decomposition: 60 = 32 + 28 (where 28 = 16 + 8 + 4)
    DYADIC_BLOCKS = [32, 16, 8, 4]

    def __init__(self):
        # Current active block index in [32, 16, 8, 4]
        self.block_idx = 0
        self.current_block_size = self.DYADIC_BLOCKS[0]
        # Frame counter relative to the current block epoch (0 to current_block_size)
        self.epoch_frame = 0
        self.global_frame_count = 0

        # Asynchronous decision buffer
        self.pending_decision: Optional[Dict[str, Any]] = None
        self.decision_ready = False
        self.decision_start_frame = 0

        # Performance & telemetry statistics
        self.stats = {
            "total_frames_seen": 0,
            "actions_dispatched": 0,
            "gate_hits": {2: 0, 4: 0, 8: 0, 16: 0, 32: 0},
            "emergency_interrupts": 0,
            "total_compute_frames_skipped": 0
        }

    def reset_epoch(self, reason: str = "action_executed"):
        """
        Resets the dynamic window epoch to 0.
        Called as soon as an action is executed or an emergency interrupt triggers.
        """
        self.block_idx = 0
        self.current_block_size = self.DYADIC_BLOCKS[0]
        self.epoch_frame = 0
        self.decision_ready = False
        self.pending_decision = None

    def submit_decision(self, decision: Dict[str, Any]):
        """
        Called asynchronously by the vision/planning worker when calculation finishes.
        """
        self.pending_decision = decision
        self.decision_ready = True

    def trigger_emergency_interrupt(self, reason: str = "sudden_threat"):
        """
        Preempts any active slow window and snaps back to Frame 0
        so the next frame-2 gate (33.33ms) can catch the threat.
        """
        self.stats["emergency_interrupts"] += 1
        self.reset_epoch(reason=reason)

    def is_opportunity_gate(self, local_frame: int) -> bool:
        """
        Checks if the given local frame is a power-of-two execution gate:
        2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32
        """
        if local_frame <= 0 or local_frame > self.current_block_size:
            return False
        # Power of 2 check: (x & (x - 1)) == 0 and x >= 2
        return (local_frame >= 2) and ((local_frame & (local_frame - 1)) == 0)

    def tick(self) -> Tuple[bool, Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Advances the 60 FPS clock by exactly 1 frame (16.67ms).
        Returns:
            (is_action_dispatched, dispatched_action, telemetry_metadata)
        """
        self.global_frame_count += 1
        self.stats["total_frames_seen"] += 1
        self.epoch_frame += 1

        is_gate = self.is_opportunity_gate(self.epoch_frame)
        dispatched_action = None
        action_fired = False

        telemetry = {
            "global_frame": self.global_frame_count,
            "epoch_frame": self.epoch_frame,
            "active_block_size": self.current_block_size,
            "is_opportunity_gate": is_gate,
            "gate_ms": round(self.epoch_frame * self.FRAME_DURATION_MS, 2),
            "decision_ready": self.decision_ready
        }

        # Check if we are on a power-of-2 opportunity gate
        if is_gate:
            if self.decision_ready and self.pending_decision is not None:
                # ACTION READY! Fire immediately at this opportunity!
                dispatched_action = self.pending_decision
                action_fired = True
                gate_frame = self.epoch_frame
                self.stats["actions_dispatched"] += 1
                if gate_frame in self.stats["gate_hits"]:
                    self.stats["gate_hits"][gate_frame] += 1

                telemetry["dispatched_at_gate"] = gate_frame
                telemetry["latency_ms"] = round(gate_frame * self.FRAME_DURATION_MS, 2)

                # Reset epoch immediately for the next action cycle
                self.reset_epoch(reason="action_dispatched")
                return (True, dispatched_action, telemetry)
            else:
                # Gate passed but decision not ready yet: let calculation continue
                telemetry["gate_status"] = "waiting_for_compute"
        else:
            # Not a gate: zero compute frame skip
            self.stats["total_compute_frames_skipped"] += 1

        # Check if active block completed without action
        if self.epoch_frame >= self.current_block_size:
            # Advance to next recursive block in 32 -> 16 -> 8 -> 4
            self.block_idx = (self.block_idx + 1) % len(self.DYADIC_BLOCKS)
            self.current_block_size = self.DYADIC_BLOCKS[self.block_idx]
            self.epoch_frame = 0

        return (action_fired, dispatched_action, telemetry)

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive compute and reaction metrics."""
        total_frames = max(1, self.stats["total_frames_seen"])
        skipped = self.stats["total_compute_frames_skipped"]
        savings_pct = (skipped / total_frames) * 100.0
        return {
            "total_frames_seen": total_frames,
            "actions_dispatched": self.stats["actions_dispatched"],
            "compute_saved_pct": round(savings_pct, 1),
            "gate_distribution": self.stats["gate_hits"],
            "emergency_interrupts": self.stats["emergency_interrupts"]
        }
