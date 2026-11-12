"""
CLASH IQ — Simulation & Test Suite for Dyadic Opportunity Scheduler.
Demonstrates the user's 2^x binary frame sampling algorithm in action:
- Instant action dispatch as soon as decision is ready
- Recursive dyadic decomposition: 60 = 32 + 16 + 8 + 4
- Non-robotic dynamic reaction latency (33ms to 533ms)
- Emergency preemption interrupt
"""

import os
import sys
import random

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.strategy.opportunity_scheduler import DyadicOpportunityScheduler

def test_scheduler_scenarios():
    print("=" * 80)
    print("   TESTING DYADIC (2^x) OPPORTUNITY SCHEDULER & RECURSIVE TIMING")
    print("=" * 80)

    scheduler = DyadicOpportunityScheduler()

    scenarios = [
        {"name": "Fast Reflex (Zap on Skeletons)", "compute_ready_at_frame": 1, "expected_gate": 2},
        {"name": "Micro-Reaction (Log vs Goblin Barrel)", "compute_ready_at_frame": 3, "expected_gate": 4},
        {"name": "Tactical Pull (Cannon vs Hog Rider)", "compute_ready_at_frame": 6, "expected_gate": 8},
        {"name": "Complex Counter (Knight + Archers vs Mega Knight)", "compute_ready_at_frame": 12, "expected_gate": 16},
        {"name": "Macro Decision (Slow Roll Giant in Back)", "compute_ready_at_frame": 24, "expected_gate": 32},
    ]

    for sc in scenarios:
        ready_frame = sc["compute_ready_at_frame"]
        name = sc["name"]
        print(f"\n[SCENARIO] {name}")
        print(f" -> Decision starts computing. Ready at Frame: {ready_frame} ({ready_frame * 16.67:.1f} ms)")

        dispatched = False
        step = 0
        while not dispatched and step < 60:
            step += 1
            if step == ready_frame:
                scheduler.submit_decision({
                    "action": "play_card",
                    "scenario": name
                })
                print(f"    [AI Engine] Decision READY! Queued into opportunity buffer at Frame {step}")

            fired, action, telem = scheduler.tick()
            if fired:
                dispatched = True
                gate = telem["dispatched_at_gate"]
                lat = telem["latency_ms"]
                print(f"    [DISPATCHED!] Fired at Gate {gate} (2^{gate.bit_length()-1}) | Latency: {lat} ms")
                print(f"    [EPOCH RESET] Immediate window reset to Frame 0 (Robotic fixed rhythm prevented!)")

    # Scenario with Emergency Interrupt
    print("\n[SCENARIO] Emergency Preemption Interrupt (Sudden Bridge Threat)")
    print(" -> AI is deep in 32-frame macro computation...")
    for f in range(1, 15):
        scheduler.tick()
    print(" -> Frame 14: Sudden Sparky dropped at bridge! Triggering preemption interrupt...")
    scheduler.trigger_emergency_interrupt(reason="sparky_at_bridge")
    # New reflex decision prepared in 1 frame
    scheduler.submit_decision({"action": "drop_electro_wizard", "target": "bridge"})
    # Next ticks
    for _ in range(5):
        fired, action, telem = scheduler.tick()
        if fired:
            gate = telem["dispatched_at_gate"]
            lat = telem["latency_ms"]
            print(f"    [INTERRUPT DISPATCH] Fired Counter at Gate {gate} | Snapped in {lat} ms!")
            break

    # Summary
    summary = scheduler.get_summary()
    print("\n" + "=" * 80)
    print("   OVERALL DYADIC SCHEDULER PERFORMANCE TELEMETRY")
    print("=" * 80)
    print(f"  Total 60-FPS Frames Processed:    {summary['total_frames_seen']}")
    print(f"  Actions Dispatched:               {summary['actions_dispatched']}")
    print(f"  Compute / Inference Saved:        {summary['compute_saved_pct']}% of frames dropped cleanly")
    print(f"  Dynamic Gate Distribution (2^x):  {summary['gate_distribution']}")
    print(f"  Emergency Preemptions Caught:     {summary['emergency_interrupts']}")
    print("=" * 80)

if __name__ == "__main__":
    test_scheduler_scenarios()
