# KAL KA KAAM — CLASH ROYALE AI ROADMAP
> **Target Date:** 2026-11-10  
> **Status:** Pinned Agenda & Ground Truth Tasks  
> **Rule:** Koi fake ya modeled numbers nahi. Sirf actual measured values aur live terminal output.

---

## 1. Core Issues Jo Kal Pehle Fix Karne Hain

### A. Elixir Rate Error (+0.8/s vs Real Rates)
- **Game Reality:** Single Elixir = `~0.357/s` ($1\text{ per } 2.8\text{s}$), Double Elixir = `~0.714/s` ($1\text{ per } 1.4\text{s}$).
- **Hamara Galat Output:** $+0.8/\text{s}$ reading aa rahi thi jo single elixir se 2.2x tez hai.
- **Asli Wajah:** Windows 11 ka **125% Display Scaling / DPI** (Window size: `696x1140 px`). Bar width hardcoded 22 px/elixir nahi ho sakti, current window resolution se dynamically calibrate karni padegi.

### B. Sabse Sasta aur Asli Test: Game ka Droplet Digit
- Elixir bar ke saath-saath **droplet ke andar ka integer digit (0 to 10)** read karna hai.
- **The Ground-Truth Rule:** Jab bar 5.0 cross kare, droplet ka digit theek usi frame par 4 se 5 hona chahiye.
- Agar dono match nahi huye $\to$ scale ya offset galat hai.

### C. Unmeasured Claims Hatana
- Capture FPS intended nahi, **measured p95 latency aur real unique frame rate** log karna hai.
- Python GIL ki reality accept karni hai (simultaneous threads I/O aur OpenCV level par hain, pure parallel Python code nahi).
- Confidence scores ka clear mathematical formula dena hai.

---

## 2. Kal Ke 4 Step-by-Step Tasks

### Task 1: Droplet Digit OCR / Template Reader (`src/vision/digit_reader.py`)
- Droplet icon ke center se integer (0, 1, 2, ... 10) template matching / pixel contour se nikaalna.
- Bar fill value aur integer digit ka real-time cross-check.

### Task 2: 125% Display Scaling Calibration
- Google Play Games PC window (`HWND 525850`, `696x1140`) par physical bar ki exact pixel width measure karna.
- Width formula: `Total Bar Pixels / 10.0 = Real px per elixir`.

### Task 3: Live Training Camp Test (15 Raw Terminal Lines)
- Training Camp match lagakar live engine chalana.
- Asli terminal output ki 10–15 continuous lines directly log aur paste karna.
- Dekhna ki single elixir mein rate sach mein `+0.357/s` ke paas rehta hai ya nahi.

### Task 4: Card Drag Flatness Verification
- Jab player card ko hand se drag/hold karta hai arena par, toh bar live freeze hota hai ya regenerate hota rehta hai?
- Ise live test karke definitive answer log karna.

---

## 3. Exit Criteria for Tomorrow
- [ ] Bar fill level aur Droplet digit dono 100% sync mein aane chahiye.
- [ ] Regeneration rate match ke actual mode (Single: ~0.36/s, Double: ~0.71/s) ke barabar measure hona chahiye.
- [ ] Real terminal logs paste hone chahiye, koi assumption ya modeled copy-paste nahi.
