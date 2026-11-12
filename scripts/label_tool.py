"""
INTERACTIVE OPENCV LABELING TOOL
Allows human-in-the-loop ground truth annotation for Clash Royale frames.

Controls:
- Left Click + Drag: Draw bounding box.
- Press '1'-'8' after drawing to assign common component names:
    1: card_slot_1
    2: card_slot_2
    3: card_slot_3
    4: card_slot_4
    5: elixir_bar
    6: elixir_droplet
    7: timer
    8: battle_button
- Press 't' to type a custom name in terminal.
- Press 'u' to undo last drawn box.
- Press 'c' to clear all boxes for current frame.
- Press 'n' / 'Right Arrow' to save and go to next frame.
- Press 'p' / 'Left Arrow' to go to previous frame.
- Press 'q' / 'Esc' to save and exit.
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
from typing import List, Dict, Any

COMPONENT_SHORTCUTS = {
    ord('1'): "card_slot_1",
    ord('2'): "card_slot_2",
    ord('3'): "card_slot_3",
    ord('4'): "card_slot_4",
    ord('5'): "elixir_bar",
    ord('6'): "elixir_droplet",
    ord('7'): "timer",
    ord('8'): "battle_button",
}

class LabelingTool:
    def __init__(self, image_paths: List[str], output_json: str):
        self.image_paths = image_paths
        self.output_json = output_json
        self.current_idx = 0
        self.annotations = self._load_existing_annotations()

        self.drawing = False
        self.ix, self.iy = -1, -1
        self.cx, self.cy = -1, -1
        self.temp_box = None
        self.current_boxes = [] # List of {"name": str, "box_px": [x1,y1,x2,y2], "box_pct": [x1,y1,x2,y2]}

    def _load_existing_annotations(self) -> Dict[str, Any]:
        if os.path.exists(self.output_json):
            try:
                with open(self.output_json, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"version": "1.0", "frames": {}}

    def _save_annotations(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_json)), exist_ok=True)
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.cx, self.cy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.cx, self.cy = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, x2 = min(self.ix, x), max(self.ix, x)
            y1, y2 = min(self.iy, y), max(self.iy, y)
            if (x2 - x1) > 4 and (y2 - y1) > 4:
                self.temp_box = [x1, y1, x2, y2]
                print(f"\n[BOX DRAWN] Pixel coordinates: [{x1}, {y1}, {x2}, {y2}]")
                print("Press 1-8 for preset component, or 't' to type custom name:")
                for k, v in COMPONENT_SHORTCUTS.items():
                    print(f"  {chr(k)}: {v}")

    def run(self):
        if not self.image_paths:
            print("[-] No image files found to label.")
            return

        window_name = "CR Human Labeling Tool (Fixed UI Annotator)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while 0 <= self.current_idx < len(self.image_paths):
            img_path = self.image_paths[self.current_idx]
            fname = os.path.basename(img_path)
            frame = cv2.imread(img_path)
            if frame is None:
                self.current_idx += 1
                continue

            h, w = frame.shape[:2]

            # Load any existing boxes for this frame
            frame_entry = self.annotations.get("frames", {}).get(fname, {})
            self.current_boxes = frame_entry.get("components", [])

            while True:
                display = frame.copy()

                # Draw saved boxes
                for item in self.current_boxes:
                    name = item["name"]
                    x1, y1, x2, y2 = item["box_px"]
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, name, (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                # Draw currently dragging box
                if self.drawing:
                    cv2.rectangle(display, (self.ix, self.iy), (self.cx, self.cy), (0, 165, 255), 2)

                # Draw pending assigned box
                if self.temp_box:
                    x1, y1, x2, y2 = self.temp_box
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, "ASSIGN KEY 1-8 or 't'", (x1, max(18, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # HUD Info Bar
                hud = f"Frame {self.current_idx + 1}/{len(self.image_paths)}: {fname} ({w}x{h}) | Boxes: {len(self.current_boxes)}"
                cv2.rectangle(display, (0, 0), (w, 24), (20, 20, 20), -1)
                cv2.putText(display, hud, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in COMPONENT_SHORTCUTS and self.temp_box:
                    comp_name = COMPONENT_SHORTCUTS[key]
                    self._add_box(comp_name, self.temp_box, w, h)
                    self.temp_box = None

                elif key == ord('t') and self.temp_box:
                    custom_name = input("\nEnter component name: ").strip()
                    if custom_name:
                        self._add_box(custom_name, self.temp_box, w, h)
                    self.temp_box = None

                elif key == ord('u'): # Undo last box
                    if self.current_boxes:
                        removed = self.current_boxes.pop()
                        print(f"Removed: {removed['name']}")
                    self.temp_box = None

                elif key == ord('c'): # Clear all
                    self.current_boxes = []
                    self.temp_box = None
                    print("Cleared all boxes for this frame.")

                elif key in (ord('n'), 83): # 'n' or Right Arrow -> Next frame
                    self._commit_frame(fname, w, h)
                    self.current_idx += 1
                    self.temp_box = None
                    break

                elif key in (ord('p'), 81): # 'p' or Left Arrow -> Prev frame
                    self._commit_frame(fname, w, h)
                    self.current_idx = max(0, self.current_idx - 1)
                    self.temp_box = None
                    break

                elif key in (ord('q'), 27): # 'q' or Esc -> Quit
                    self._commit_frame(fname, w, h)
                    self._save_annotations()
                    cv2.destroyAllWindows()
                    print("\n[+] Labels successfully saved. Exiting labeling tool.")
                    return

    def _add_box(self, name: str, box_px: List[int], w: int, h: int):
        x1, y1, x2, y2 = box_px
        box_pct = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]
        self.current_boxes.append({
            "name": name,
            "box_px": box_px,
            "box_pct": box_pct
        })
        print(f"[SAVED] {name} -> px: {box_px}, pct: {box_pct}")

    def _commit_frame(self, fname: str, w: int, h: int):
        if "frames" not in self.annotations:
            self.annotations["frames"] = {}
        self.annotations["frames"][fname] = {
            "width": w,
            "height": h,
            "components": self.current_boxes
        }
        self._save_annotations()

if __name__ == "__main__":
    search_dir = os.path.join("data", "observer_logs")
    frame_files = sorted(glob.glob(os.path.join(search_dir, "**", "frames", "*.png"), recursive=True))

    # Also include any observer test frames
    test_frames = sorted(glob.glob(os.path.join("data", "observer_tests", "*.png")))
    frame_files.extend(test_frames)

    out_file = os.path.join("data", "annotations", "manual_labels.json")
    print(f"Found {len(frame_files)} frames to label.")
    print(f"Annotations output target: {out_file}")

    tool = LabelingTool(frame_files, out_file)
    tool.run()
