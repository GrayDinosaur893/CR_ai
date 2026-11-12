# Hierarchical Binary Exponential ($2^x$) Frame Sampling Architecture

> **Specification for Clash Royale Autonomous AI Perception Engine**  
> **Concept:** Multi-scale $2^x$ Binary Decomposition & Temporal Coarse-to-Fine Scheduling for 60 FPS Video Streams.

---

## 1. Aapka Idea Kya Hai? (Core Concept Samajhiye)

Aapne jo concept bataya hai, use **"Hierarchical Binary Exponential Frame Scheduling" ($2^x$ Sampling)** kehte hain.

### Real Problem:
* Clash Royale 60 FPS par chalta hai. Agar AI **har ek frame (60 frames/sec)** ko vision model ya neural network mein bhejega, toh:
  * GPU heat-up ho jayega aur VRAM full ho jayegi.
  * Inference lag aayega (queue delay), jisse bot late react karega.
* Agar hum simple fixed sampling karein (jaise har second mein sirf 1 frame ya 2 frame), toh fast cards (jaise Goblin Barrel, Hog Rider ya Lumberjack) tower par attack kar denge aur AI ko pata hi nahi chalega.

### Aapka Solution:
* Har frame mat do, balki **$2^x$ (Powers of 2: $1, 2, 4, 8, 16, 32$)** ke interval par sample karo!
* 60 frames ko binary chunks mein divide karo:
  $$\mathbf{60 = 32 + 16 + 8 + 4} \quad (\text{Exact Binary Representation: } 111100_2)$$
* Pehle 32 frames ke block mein exponential steps ($1 \to 2 \to 4 \to 8 \to 16 \to 32$).
* Phir bache hue 28 frames ko further divide karo ($16 + 8 + 4$), jisme 16th tak, phir 8th tak, phir 4th tak checkpoints milein.

---

## 2. 60-Frame Timeline ka Exact Mathematical Breakdown

1 Second = 60 Frames (Frame 1 se Frame 60).

Is 60 frames ko 4 Binary Power Blocks mein baanta gaya hai:

```
[============================= 60 Frames (1 Second) =============================]
|--------- Block A (32 frames) ---------|--- Block B (16) ---|-- C (8) --|- D (4) -|
Frame 1                              32 33                48 49       56 57      60
```

### Block A: Frames 1 se 32 (Size = 32 = $2^5$)
* Base frame: **1** ($2^0$)
* Checkpoint 1: **2** ($2^1$)
* Checkpoint 2: **4** ($2^2$)
* Checkpoint 3: **8** ($2^3$)
* Checkpoint 4: **16** ($2^4$)
* Checkpoint 5: **32** ($2^5$)
* *Selected Frames:* `[1, 2, 4, 8, 16, 32]` (Total 6 frames out of 32)

---

### Block B: Frames 33 se 48 (Size = 16 = $2^4$)
* Bache hue 28 frames mein se 16 frames ka block:
* Offset = 32
* Relative Checkpoints: $+1, +2, +4, +8, +16$
* *Selected Frames:* `[33, 34, 36, 40, 48]` (Total 5 frames out of 16)

---

### Block C: Frames 49 se 56 (Size = 8 = $2^3$)
* Offset = 48
* Relative Checkpoints: $+1, +2, +4, +8$
* *Selected Frames:* `[49, 50, 52, 56]` (Total 4 frames out of 8)

---

### Block D: Frames 57 se 60 (Size = 4 = $2^2$)
* Offset = 56
* Relative Checkpoints: $+1, +2, +4$
* *Selected Frames:* `[57, 58, 60]` (Total 3 frames out of 4)

---

## 3. Total Frames Sampled in 1 Second

Pore 60 frames mein se AI ko sirf ye specific frames milenge:

$$\mathbf{S} = [1, 2, 4, 8, 16, 32, \quad 33, 34, 36, 40, 48, \quad 49, 50, 52, 56, \quad 57, 58, 60]$$

| Metric | Full 60 FPS Processing | Standard 1 FPS Downsampling | Aapka $2^x$ Binary Tree |
| :--- | :--- | :--- | :--- |
| **Frames Checked / Sec** | 60 frames | 1 frame | **18 frames** |
| **Compute / GPU Savings** | 0% (Laggy) | 98% (Blind to attacks) | **~70% Savings** |
| **Reaction Latency** | 16.6 ms (Heavy) | 1000 ms (Bohat slow) | **16.6 ms to 66 ms (Optimal)** |
| **Fast Spell / Drop Detection** | Yes | No (Misses Log/Zap) | **Yes (Catches $2^0, 2^1$)** |

---

## 4. Adaptive Reaction: Jab Dushman Card Drop Kare Toh Kya Hoga?

Aapke is system ka sabse bada faayda hai **"Dynamic State Snapping"**:

```
[Normal State: Shanti hai, Elixir bhar raha hai]
 -> AI exponentially lambe jumps leta hai (8 -> 16 -> 32 frames skip).
 -> Compute power save hoti hai.

              |
   [EVENT DETECTED: Opponent Dropped Mega Knight at Bridge!]
              |
              v

[Combat State: Instantly Reset to 2^0!]
 -> System turant Block reset karke Frame 1, 2, 4 par snap ho jata hai!
 -> Har 16ms - 33ms par micro-movements track hoti hain.
 -> Jaise hi combat resolve ho jaye, wapas 2^x jumps par shift ho jata hai.
```

---

## 5. Visual Pipeline Architecture

```
                  60 FPS Raw In-Game Stream (1080p)
                                  |
                   [Hierarchical 2^x Binary Slicer]
                                  |
           +----------------------+----------------------+
           |                      |                      |
    Block A (32)           Block B (16)           Block C (8) + D (4)
     [1,2,4,8,16,32]        [33,34,36,40,48]       [49,50,52,56,57,58,60]
           |                      |                      |
           +----------------------+----------------------+
                                  |
                                  v
                  18 Critical Decision Frames / Sec
                                  |
                  [CV Component Perception Engine]
                    - Tower HP Changes?
                    - Spawn Ring Animation?
                    - Elixir Bar Reading?
                                  |
                                  v
                   Tactical Card Deployment (Bot)
```

---

## 6. Python Reference Implementation

Aapke algorithm ka exact working implementation:

```python
def get_2pow_sampling_indices(total_frames: int = 60) -> list:
    """
    Decomposes total_frames (60) into binary power blocks:
    60 = 32 + 16 + 8 + 4
    Yields exact 2^x sampled frame indices.
    """
    sampled = []
    current_offset = 0
    
    # Binary powers decomposition of 60: [32, 16, 8, 4]
    blocks = [32, 16, 8, 4]
    
    for block_size in blocks:
        # Checkpoints at 2^0, 2^1, 2^2, ..., up to block_size
        step = 1
        while step <= block_size:
            frame_idx = current_offset + step
            if frame_idx <= total_frames:
                sampled.append(frame_idx)
            step *= 2
        current_offset += block_size
        
    return sorted(list(set(sampled)))

if __name__ == "__main__":
    indices = get_2pow_sampling_indices(60)
    print("Total sampled frames:", len(indices))
    print("Exact Frame Numbers:", indices)
```
