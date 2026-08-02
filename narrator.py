import os
import time
import queue
import threading
import ctypes
import numpy as np
from gtts import gTTS

class AudioNarrator:
    def __init__(self):
        self.queue = queue.Queue()
        self.last_spoken = {}  # key: label, val: timestamp
        self.cooldown = 4.0    # 4 seconds cooldown per object category
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        # Clean up any old temporary audio files in the folder on startup
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        try:
            for file in os.listdir("."):
                if file.startswith("temp_announce_") and file.endswith(".mp3"):
                    try:
                        os.remove(file)
                    except Exception:
                        pass
        except Exception:
            pass

    def _worker(self):
        file_counter = 0
        while self.running:
            try:
                # Wait for text to speak
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            if text is None:
                break
                
            try:
                # Generate a unique temp file name to prevent lock conflicts
                file_counter += 1
                temp_file = os.path.abspath(f"temp_announce_{int(time.time())}_{file_counter}.mp3")
                
                # Generate text to speech
                tts = gTTS(text=text, lang='en')
                tts.save(temp_file)
                
                # Play using Windows MCI API (completely native, async to main thread)
                alias = f"announcement_{int(time.time() * 1000)}_{file_counter}"
                
                # Open, play (wait for completion), and close
                ctypes.windll.winmm.mciSendStringW(f'open "{temp_file}" type mpegvideo alias {alias}', None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(f"play {alias} wait", None, 0, 0)
                ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, 0)
                
                # Remove file after playing
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            except Exception as e:
                print(f"Audio error in worker thread: {e}")
            finally:
                self.queue.task_done()

    def speak(self, text):
        self.queue.put(text)

    def announce_objects(self, detections, depth_map_resized):
        # detections: list of tuples (x1, y1, x2, y2, conf, label)
        now = time.time()
        candidates = []
        
        for det in detections:
            if len(det) < 6:
                continue
            x1, y1, x2, y2, conf, label = det
            
            # Check cooldown
            if label in self.last_spoken and (now - self.last_spoken[label]) < self.cooldown:
                continue
                
            # Bounding box center depth lookup
            h, w = depth_map_resized.shape[:2]
            center_x = int(np.clip((x1 + x2) // 2, 0, w - 1))
            center_y = int(np.clip((y1 + y2) // 2, 0, h - 1))
            depth_value = depth_map_resized[center_y, center_x]
            
            candidates.append((depth_value, label))
            
        if candidates:
            # Prioritize: Sort by disparity in DESCENDING order (higher disparity = closer object)
            candidates.sort(key=lambda x: x[0], reverse=True)
            closest_depth, closest_label = candidates[0]
            
            # Record spoken timestamp to enforce cooldown
            self.last_spoken[closest_label] = now
            
            # Sentence formation
            # High values in disparity mean very close objects.
            # In your dataset, values near 800+ represent close range.
            if closest_depth > 800:
                sentence = f"Warning! {closest_label} is very close. {closest_depth:.0f} centimeters."
            else:
                sentence = f"{closest_label} detected at {closest_depth:.0f} centimeters."
            
            self.speak(sentence)

    def stop(self):
        self.running = False
        self.queue.put(None)
