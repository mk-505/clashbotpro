"""
Live Perception Loop
=====================
Captures frames from MuMu window and runs the full SAR pipeline (state/action/reward extraction).

Usage:
    python live_perception.py

Requirements:
    - mumu_config.py must exist with MUMU_WINDOW_POS and MUMU_WINDOW_SIZE
    - katacr.mumu_adapter and SARBuilder available

The script will show a debug window with detections, state info and FPS.
Press 'q' to exit.
"""

import time
import cv2
import numpy as np

# load config
try:
    from mumu_config import MUMU_WINDOW_POS, MUMU_WINDOW_SIZE
except ImportError:
    raise RuntimeError(
        "mumu_config.py not found. Run mumu_calibration.py first.")

from katacr.mumu_adapter import MuMuAdapter
from katacr.policy.perceptron.torch_sar_builder import SARBuilder


def format_state(state: dict) -> str:
    # Simple summarization for display
    parts = []
    if 'time' in state:
        parts.append(f"Time:{state['time']}")
    if 'elixir' in state:
        parts.append(f"Elixir:{state['elixir']}")
    if 'cards' in state:
        parts.append("Cards:" + ",".join(str(c) for c in state['cards']))
    return " | ".join(parts)


def main():
    adapter = MuMuAdapter(window_pos=MUMU_WINDOW_POS,
                          window_size=MUMU_WINDOW_SIZE)
    sar = SARBuilder(verbose=False)

    cv2.namedWindow('Live', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    print("Starting live perception. Press 'q' to quit.")
    last_time = time.time()
    while True:
        frame = adapter.capture()
        if frame is None:
            print("Frame capture failed, retrying...")
            time.sleep(0.1)
            continue

        result = sar.update(frame)
        if result is None:
            # skip frame
            continue
        info, dt_list = result
        s, a, r, build_dt = sar.get_sar()

        # render debug overlay
        debug = frame.copy()
        if info is not None and 'arena' in info:
            arena_img = info['arena'].show_box(show_conf=True)
            # overlay at small size
            h, w = arena_img.shape[:2]
            debug[0:h, 0:w] = cv2.resize(arena_img, (w, h))
        # draw state text
        state_text = format_state(s) if s else ''
        cv2.putText(debug, state_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug, f"Action:{a}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)
        cv2.putText(debug, f"Reward:{r}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # FPS
        fps = adapter.get_fps()
        cv2.putText(debug, f"FPS:{fps:.1f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Live', debug)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
