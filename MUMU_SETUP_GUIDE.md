# MuMu Setup & Calibration Guide

## Overview
This guide will help you set up the MuMu emulator integration for live Clash Royale game detection.

## Step 1: Install MuMu Emulator
- Download MuMu emulator from: https://www.mumuplayer.com/
- Install and launch Clash Royale in MuMu

## Step 2: Install Updated Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- `mss` - For cross-platform screen capture
- `pyautogui` - For future mouse control
- Other necessary libraries

## Step 3: Calibrate MuMu Window Position

### Run the Calibration Script
```bash
python mumu_calibration.py
```

### Follow the Instructions:
1. A full-screen screenshot will appear
2. **Click and drag** to select your MuMu game window region
3. Start from the **top-left corner** of the game window (where the time display is)
4. Drag to the **bottom-right corner** 
5. Release the mouse

### What to Expect:
- The script will show you the exact pixel coordinates
- It will create `mumu_config.py` with your settings
- You can verify the region is correct by pressing 'v'

### Output Example:
```
============================================================
CALIBRATION RESULTS
============================================================

MuMu Window Position (top-left corner):
  X: 100
  Y: 100

MuMu Window Size:
  Width:  568
  Height: 896

Aspect Ratio: 0.63

--- Copy this into your MuMu adapter initialization ---

window_pos = (100, 100)
window_size = (568, 896)
```

## Step 4: Verify Configuration

Check that `mumu_config.py` was created in the root directory:

```bash
cat mumu_config.py
```

Should contain:
```python
MUMU_WINDOW_POS = (100, 100)      # Top-left corner
MUMU_WINDOW_SIZE = (568, 896)     # Width, Height
```

## Step 5: Test the MuMu Adapter

Run a quick test to ensure frames are being captured:

```bash
python -c "
from katacr.mumu_adapter import MuMuAdapter
from mumu_config import MUMU_WINDOW_POS, MUMU_WINDOW_SIZE
import cv2

adapter = MuMuAdapter(MUMU_WINDOW_POS, MUMU_WINDOW_SIZE)

print('Testing MuMu adapter...')
for i in range(5):
    frame = adapter.capture()
    if frame is not None:
        print(f'  Frame {i+1}: Shape={frame.shape}, FPS={adapter.get_fps():.1f}')
    else:
        print(f'  Frame {i+1}: Capture failed!')

print('✅ MuMu adapter working!')
"
```

## Step 6: Understanding the Coordinates

### Important Notes:
- **X, Y Position**: Top-left corner of the game window on your screen
- **Width, Height**: The size of the game area (typically 568×896 for standard Clash Royale)
- **Screen vs Window**: These are absolute screen coordinates, not relative to any other window

### Visual Reference:
```
┌─────────────────────────────────────┐
│ (X, Y) ← Top-left corner            │
│   ┌──────────────────────────┐      │
│   │                          │      │
│   │   MuMu Game Window      │ Width │
│   │   (568 × 896)           │      │
│   │                          │      │
│   └──────────────────────────┘      │
│           Height ↓                   │
└─────────────────────────────────────┘
```

## Next Steps

After calibration, you'll be able to use:

1. **Live Detection Loop** - Captures game frames and runs YOLO detection
2. **State/Action/Reward Extraction** - Builds game state from perception
3. **Data Logging** - Saves gameplay data for analysis

## Troubleshooting

### "Capture failed" Error
- Check that MuMu window is visible and not minimized
- Verify the coordinates are correct (try calibrating again)
- Make sure the window isn't partially off-screen

### Wrong Region Captured
- Run the calibration script again
- Be careful to select from the exact top-left to bottom-right of the game area
- The game display should start immediately after selection

### FPS Too Low
- Close other applications consuming CPU
- Reduce screen resolution if possible
- The capture is typically 20-30 FPS depending on system performance

## File Reference

| File | Purpose |
|------|---------|
| `katacr/mumu_adapter.py` | Main MuMu capture class |
| `mumu_calibration.py` | Interactive calibration script |
| `mumu_config.py` | Your generated configuration (auto-created) |
| `requirements.txt` | Updated with mss and pyautogui |

## API Reference

### MuMuAdapter Class

```python
from katacr.mumu_adapter import MuMuAdapter

# Initialize with your coordinates
adapter = MuMuAdapter(
    window_pos=(100, 100),      # Top-left corner
    window_size=(568, 896),     # Width, Height
    backend='mss'               # 'mss' (default) or 'pillow'
)

# Capture a frame (returns BGR numpy array)
frame = adapter.capture()

# Get current FPS
fps = adapter.get_fps()

# Update position dynamically (if window moves)
adapter.update_position((150, 150))

# Context manager support
with MuMuAdapter((100, 100), (568, 896)) as adapter:
    frame = adapter.capture()
```

## Questions?

Check the following files for more details:
- `katacr/mumu_adapter.py` - Full documentation and examples
- `mumu_calibration.py` - Interactive guide in the script itself
