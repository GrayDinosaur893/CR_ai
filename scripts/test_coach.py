import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.llm_client.openrouter import OpenRouterClient, DEFAULT_FREE_MODELS
from src.coach.replay_coach import ReplayCoach
from src.game_state.elixir_tracker import ElixirTracker

def test_pipeline():
    print("=== CLASH IQ: Pipeline Test ===")
    
    # 1. Test Elixir Tracker
    print("\n1. Testing Elixir Tracker logic...")
    tracker = ElixirTracker(initial_elixir=5.0)
    tracker.advance_time(10.0) # 10s elapsed
    tracker.register_card_played(player="opponent", elixir_cost=8, current_seconds=12.0) # Golem
    tracker.register_card_played(player="player", elixir_cost=4, current_seconds=13.0) # Hog Rider
    state = tracker.get_state()
    print(f"Elixir State after plays: {state}")
    
    # 2. Check OpenRouter Client setup
    print("\n2. Checking OpenRouter Client...")
    client = OpenRouterClient()
    if not client.is_configured():
        print("[!] OPENROUTER_API_KEY environment variable is NOT set.")
        print(f"    Available top free models on OpenRouter: {DEFAULT_FREE_MODELS}")
        print("    To run live inference:")
        print("    1. Sign up / login at https://openrouter.ai/keys")
        print("    2. Create a free API key")
        print("    3. Set it in PowerShell: $env:OPENROUTER_API_KEY='sk-or-v1-...'")
        print("    4. Re-run this script!")
        return

    print(f"[OK] OpenRouter API Key detected. Using model: {client.model}")
    print("\n3. Testing live coaching inference via OpenRouter...")
    coach = ReplayCoach(client=client)
    sample_events = [
        {"timestamp": 12.0, "player": "opponent", "card": "golem", "elixir_cost": 8, "lane": "left"},
        {"timestamp": 13.0, "player": "player", "card": "hog_rider", "elixir_cost": 4, "lane": "right"},
        {"timestamp": 13.5, "player": "player", "card": "goblins", "elixir_cost": 2, "lane": "right"}
    ]
    sample_commentary = [
        {"timestamp": 13.0, "text": "Opponent invested heavy in the back, we immediately punish opposite lane!"}
    ]
    
    try:
        result = coach.analyze_replay_events(
            events=sample_events,
            commentary=sample_commentary,
            player_name="Player",
            opponent_name="Opponent"
        )
        print("\n--- AI Coach Output ---")
        output_text = result["raw_output"] or ""
        # Print safely on Windows console
        try:
            print(output_text)
        except UnicodeEncodeError:
            print(output_text.encode(sys.stdout.encoding, errors="replace").decode(sys.stdout.encoding))
            
        print(f"\n[SUCCESS] Model '{result['model_used']}' inference completed successfully!")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    test_pipeline()
