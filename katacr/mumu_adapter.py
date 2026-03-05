"""
MuMu Emulator Adapter - Replaces Scrcpy for game screen capture
Captures frames from MuMu window and provides them in BGR format for processing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time


class MuMuAdapter:
    """
    Captures game frames from MuMu emulator window.
    
    Supports:
    - Windows: Uses mss library
    - macOS: Uses mss library (cross-platform)
    - Linux: Uses mss library (cross-platform)
    
    Usage:
        adapter = MuMuAdapter(window_pos=(100, 100), window_size=(568, 896))
        frame = adapter.capture()  # Returns BGR frame
    """
    
    def __init__(self, window_pos: Tuple[int, int], window_size: Tuple[int, int], backend='mss'):
        """
        Initialize MuMu adapter.
        
        Args:
            window_pos (Tuple[int, int]): Top-left corner (x, y) of MuMu window
            window_size (Tuple[int, int]): Size (width, height) of MuMu window
            backend (str): 'mss' for cross-platform, 'pillow' as alternative
        """
        self.window_pos = window_pos
        self.window_size = window_size
        self.backend = backend
        self.fps_counter = 0
        self.fps_time = time.time()
        self.fps = 0
        
        if backend == 'mss':
            try:
                import mss
                self.mss = mss.mss()
            except ImportError:
                raise ImportError("mss not installed. Install with: pip install mss")
        elif backend == 'pillow':
            from PIL import ImageGrab
            self.ImageGrab = ImageGrab
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def capture(self) -> Optional[np.ndarray]:
        """
        Capture a frame from MuMu window.
        
        Returns:
            np.ndarray: Frame in BGR format (H, W, 3), or None if capture failed
        """
        try:
            if self.backend == 'mss':
                return self._capture_mss()
            elif self.backend == 'pillow':
                return self._capture_pillow()
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def _capture_mss(self) -> Optional[np.ndarray]:
        """Capture using mss library (cross-platform)."""
        x, y = self.window_pos
        w, h = self.window_size
        
        # mss region format: {'left': x, 'top': y, 'width': w, 'height': h}
        monitor = {
            'left': x,
            'top': y,
            'width': w,
            'height': h
        }
        
        # Capture screenshot
        screenshot = self.mss.grab(monitor)
        
        # Convert RGBA to BGR
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        self._update_fps()
        return frame
    
    def _capture_pillow(self) -> Optional[np.ndarray]:
        """Capture using PIL/Pillow (alternative)."""
        x, y = self.window_pos
        w, h = self.window_size
        
        # PIL bbox: (left, top, right, bottom)
        bbox = (x, y, x + w, y + h)
        
        # Capture screenshot
        screenshot = self.ImageGrab.grab(bbox=bbox)
        
        # Convert RGB to BGR
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        self._update_fps()
        return frame
    
    def _update_fps(self):
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        elapsed = current_time - self.fps_time
        
        if elapsed >= 1.0:
            self.fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
    
    def validate_position(self) -> bool:
        """
        Check if window position is valid by attempting capture.
        
        Returns:
            bool: True if capture successful, False otherwise
        """
        frame = self.capture()
        return frame is not None
    
    def update_position(self, window_pos: Tuple[int, int]):
        """Update window position (for dynamic tracking)."""
        self.window_pos = window_pos
    
    def update_size(self, window_size: Tuple[int, int]):
        """Update window size."""
        self.window_size = window_size
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup."""
        if self.backend == 'mss':
            self.mss.__exit__(exc_type, exc_val, exc_tb)
    
    def __repr__(self):
        return f"MuMuAdapter(pos={self.window_pos}, size={self.window_size}, backend={self.backend})"


class MuMuWindowFinder:
    """
    Helper class to find MuMu window automatically (platform-specific).
    
    Supports:
    - Windows: Finds window by name "MuMu"
    - macOS: Limited support (manual specification recommended)
    - Linux: Limited support (manual specification recommended)
    """
    
    @staticmethod
    def find_mumu_window() -> Optional[Tuple[int, int, int, int]]:
        """
        Find MuMu window position and size.
        
        Returns:
            Tuple[int, int, int, int]: (x, y, width, height) or None if not found
        """
        import platform
        system = platform.system()
        
        if system == 'Windows':
            return MuMuWindowFinder._find_windows()
        elif system == 'Darwin':  # macOS
            return MuMuWindowFinder._find_macos()
        elif system == 'Linux':
            return MuMuWindowFinder._find_linux()
        else:
            print(f"Unsupported platform: {system}")
            return None
    
    @staticmethod
    def _find_windows() -> Optional[Tuple[int, int, int, int]]:
        """Find MuMu window on Windows using pygetwindow."""
        try:
            import pygetwindow
            windows = pygetwindow.getWindowsWithTitle('MuMu')
            if windows:
                w = windows[0]
                return (w.left, w.top, w.width, w.height)
        except ImportError:
            print("pygetwindow not installed. Install with: pip install pygetwindow")
        except Exception as e:
            print(f"Error finding MuMu window on Windows: {e}")
        return None
    
    @staticmethod
    def _find_macos() -> Optional[Tuple[int, int, int, int]]:
        """Find MuMu window on macOS using PyObjC."""
        try:
            from AppKit import NSWorkspace, NSScreen
            workspace = NSWorkspace.sharedWorkspace()
            
            # Find MuMu application
            for app in workspace.runningApplications():
                if 'MuMu' in app.localizedName():
                    print(f"Found: {app.localizedName()}")
                    # Note: Getting exact window position on macOS is complex
                    # Return default or require manual input
                    return None
        except ImportError:
            print("PyObjC not available on this macOS installation")
        except Exception as e:
            print(f"Error finding MuMu window on macOS: {e}")
        
        print("macOS auto-detection not fully supported. Use manual calibration.")
        return None
    
    @staticmethod
    def _find_linux() -> Optional[Tuple[int, int, int, int]]:
        """Find MuMu window on Linux using wmctrl."""
        try:
            import subprocess
            result = subprocess.run(['wmctrl', '-l'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'MuMu' in line:
                    print(f"Found: {line}")
                    # Parse window info - format varies by WM
                    return None
        except Exception as e:
            print(f"Error finding MuMu window on Linux: {e}")
        
        print("Linux auto-detection not fully supported. Use manual calibration.")
        return None


if __name__ == '__main__':
    print("MuMu Adapter Test")
    print("=" * 50)
    
    # Try to auto-find MuMu window
    result = MuMuWindowFinder.find_mumu_window()
    
    if result:
        print(f"Auto-detected MuMu window: {result}")
        x, y, w, h = result
        adapter = MuMuAdapter(window_pos=(x, y), window_size=(w, h))
    else:
        print("Could not auto-detect. Using example dimensions.")
        print("For macOS: Typically around (100, 100, 568, 896)")
        adapter = MuMuAdapter(window_pos=(100, 100), window_size=(568, 896))
    
    print(f"\nAdapter: {adapter}")
    
    # Test capture
    print("\nAttempting capture...")
    frame = adapter.capture()
    
    if frame is not None:
        print(f"✅ Capture successful! Frame shape: {frame.shape}")
        print(f"FPS: {adapter.get_fps():.1f}")
    else:
        print("❌ Capture failed. Check window position.")
