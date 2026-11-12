# NECESSARY.MD — NEXT-LEVEL COMPUTER VISION & OBJECT RECOGNITION SPECIFICATION

> **Target:** Zero-hallucination, sub-100ms latency, high-precision visual perception engine for Clash Royale gameplay recordings and real-time streams.

---

## 1. The Core Challenge of Clash Royale CV

Clash Royale is one of the most visually deceptive mobile games for standard off-the-shelf vision models:
1. **Extreme Scale Variance:** A massive Golem (150x200 px) vs tiny Skeletons or Bats (15x15 px).
2. **Heavy Particle Occlusion:** Poison, Freeze, Rage tint, Fireball smoke, Rocket trails, and Log debris frequently occlude troops and tile markers.
3. **Spawn Clutter & Animations:** Units do not instantly render; they emerge from spawn clouds, drop-in bounce animations, and circular spawn timer clocks.
4. **Isometric Projection & Perspective Distortion:** Screen pixels do not map linearly to arena coordinates; top lane units appear physically smaller than bottom lane units.
5. **Evolutions & Skin Variations:** Evolution glows, star-level cosmetics, and tower skins change visual signatures without changing canonical card IDs.

To achieve **"Next-Level"** accuracy, CLASH IQ rejects naive end-to-end vision models in favor of a **Decoupled Multi-Stage Perception Architecture**.

---

## 2. Multi-Stage Perception Architecture

```
                      Raw Video Frame (1080x1920 or 720x1280)
                                      |
              +-----------------------+-----------------------+
              |                                               |
      [ROI Slicer Engine]                            [Arena Perspective Warper]
              |                                               |
   +----------+----------+                        (Homography Matrix H)
   |                     |                                    |
[UI Top ROI]       [UI Bottom ROI]                   [Arena Grid (18x32 Tiles)]
   |                     |                                    |
   |-- Match Timer OCR   |-- Player Hand (4 Cards)            |-- YOLOv10-X / RT-DETR
   |-- Tower HP OCR      |-- Next Card Queue                      Troop & Building Detector
   |-- Overtime State    |-- Elixir Bar (Digit + Fill)        |-- ByteTrack / BoT-SORT
                         |-- Card Drop Glow Event                 Multi-Object Trajectory
                                                              |-- Spawn Clock & Ring Detector
                                                              |-- Spell AOE Projector
                                                              |
                                                              v
                                                   Timestamped Visual Events
                                                {"card": "hog_rider", "tile": [14, 22]}
```

---

## 3. Component Breakdown & Model Specifications

### A. Arena Coordinate Homography (Pixel -> 18x32 Arena Tiles)
Clash Royale's playable arena is strictly defined as an **18-column by 32-row tile grid**:
- Bridges are fixed at standard tile coordinates (Left: cols 3-4, Right: cols 14-15; rows 15-16).
- River divides row 15 and row 16.
- Princess Towers are centered on fixed tile anchors.

* **Implementation:** Four-point perspective transform (Homography matrix $H$):
  $$P_{tile} = H \cdot P_{pixel}$$
* Calibration points are detected automatically via the river banks, bridge wooden pegs, and king tower corners.
* **Output:** Every detected unit has an exact tile position `(tile_x, tile_y)` rather than vague screen pixel coordinates.

---

### B. Arena Entity Detection (Troops, Buildings, Spells)
* **Backbone:** **YOLOv10-X** or **RT-DETR-X (Real-Time Detection Transformer)** fine-tuned specifically on Clash Royale sprites across all 110+ cards.
* **Why RT-DETR/YOLOv10:**
  - NMS-free architecture avoids duplicate bounding boxes when swarm units (Skeleton Army, Goblin Gang) overlap.
  - Multi-scale Feature Pyramid Network (BiFPN) captures both 15px Bats and 200px P.E.K.K.A.
* **Label Hierarchy:**
  - `class_id`: Canonical card ID (e.g., `electro_giant`, `princess`, `the_log`).
  - `variant`: `standard`, `evolution`, `mirrored`.
  - `owner`: `player` (Blue / bottom), `opponent` (Red / top). Determined by health-bar anchor color (HSV filter: Blue $[100-130^\circ]$, Red $[0-15^\circ \cup 345-360^\circ]$).

---

### C. Temporal Multi-Object Tracking (MOT)
Static per-frame detection fails during spell flashes and bridge crowding.
* **Tracker:** **ByteTrack + BoT-SORT (Camera Motion Compensation)**
* **Features:**
  - Tracks unit ID over time (`track_id: 42 -> knight, alive for 8.4 seconds`).
  - Distinguishes new deployments from already living troops.
  - Tracks death events when bounding box disappears and elixir/death particle is detected.

---

### D. Deployment Spawn Clock & Drop Detector
A card deployment is an atomic event occurring at exact time $t_0$:
1. **Spawn Ring Detector:** When a card is placed, a grey circular loading clock appears for 1.0s (or 3.0s for Golem/3M) before the troop can move.
2. **Circular Hough Transform + Template Matcher:** Detects the radial clock animation anchor with $\ge 99.2\%$ precision.
3. **Card Drop Flash in Hand:** Bottom UI hand slot flashes yellow/white when released, confirming the exact frame of placement.

---

### E. Bottom UI Bar (Player Hand, Cycle, & Elixir OCR)
* **Hand Slots (4 cards + Next card slot):**
  - High-resolution cropped 120x150 px cards.
  - Identification via **CosFace / ArcFace Metric Embedding** or **Perceptual Hash + Feature Bank**:
    - Pre-computed embeddings of all 110 card portraits.
    - Cosine similarity $\ge 0.96$ matches the card in under 2ms.
* **Elixir Bar Reading:**
  - Double verification:
    1. **Digital OCR:** CRNN / SVTR model reading the integer digit (0 to 10) in the purple drop icon.
    2. **Linear Fill Gauge:** Masking purple pixel width across the 10-segment bar gives decimal precision (e.g. `7.4 elixir`).

---

### F. Tower HP & Match Clock OCR
* **Match Clock:**
  - Top center / top right: format `M:SS`.
  - Standard SVTR / lightweight MobileNet-V4 OCR trained on Supercell's custom clash font (Supercell-Magic).
* **Tower HP:**
  - Princess Towers (Left & Right) and King Tower HP numbers.
  - Detects damage spikes, spell chip damage, and exact destruction timestamps.

---

## 4. Dataset Generation & Annotation Pipeline

To train this next-level vision engine without requiring 10,000 hours of manual clicking:

### 1. Synthetic Clash Royale Compositor (`cr_synth`)
* Extract transparent PNG sprites of all troops, buildings, and animations from game assets.
* Synthetically overlay troops onto arena backgrounds with realistic lighting, shadows, rage effects, and scaling.
* Automatically generate 50,000+ perfectly labeled bounding boxes with:
  - Exact bounding boxes `[x_min, y_min, x_max, y_max]`
  - Tile coordinates `[tile_x, tile_y]`
  - Swarm clusters and occlusions

### 2. Semi-Supervised Self-Training
* Run pretrained detector on high-resolution tournament replays (TV Royale).
* Use temporal consistency: If a unit is detected as `musketeer` for 10 consecutive frames, propagate the label to occluded intermediate frames.
* Human annotators only review uncertain detections (confidence $< 0.85$).

---

## 5. Performance & Quality Benchmarks (Target Metrics)

| Task | Target Metric | Minimum Acceptable Threshold |
| :--- | :--- | :--- |
| **Card Deployment Detection** | F1-Score | $\ge 0.98$ |
| **Deployment Timestamp Accuracy** | Error Margin | $\le \pm 0.1 \text{ seconds}$ |
| **Arena Tile Localization** | Grid Distance Error | $\le 0.5 \text{ tiles}$ |
| **Troop Identification Accuracy** | Top-1 Accuracy | $\ge 99.0\%$ |
| **Swarm Count Precision (Skarmy/Minions)**| IoU | $\ge 0.85$ |
| **Elixir Reading Precision** | Absolute Error | $\le 0.2 \text{ elixir}$ |
| **Frame Processing Throughput** | Inference Speed | $\ge 60 \text{ FPS (CUDA)}$ / $\ge 25 \text{ FPS (CPU)}$ |

---

## 6. Implementation Roadmap for Vision Module (`src/vision/`)

1. **`src/vision/roi.py`**: Automated resolution-agnostic screen anchor & ROI detector.
2. **`src/vision/homography.py`**: Perspective matrix mapping pixel $(x, y)$ to arena tile $(tx, ty)$.
3. **`src/vision/hand_reader.py`**: Real-time hand cards, next queue, and elixir reader.
4. **`src/vision/entity_detector.py`**: YOLOv10 / RT-DETR inference wrapper for in-arena troops.
5. **`src/vision/deployment_detector.py`**: Spawn clock ring & drop frame finder.
6. **`src/vision/tracker.py`**: ByteTrack temporal unit trajectory manager.
