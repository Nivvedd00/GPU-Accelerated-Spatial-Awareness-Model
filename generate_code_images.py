import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

def generate_screenshots():
    os.makedirs("code_screenshots", exist_ok=True)
    
    snippets = {
        "dual_yolo": '''# Dual-Model YOLOv8 Configuration
# Model A: General COCO Obstacles
coco_model = YOLO("yolov8n.pt")

# Model B: Custom DWS Structural Landmarks
yolo_model_path = "yolov8n.pt"  # Fallback
detect_dir = Path("runs/detect")
if detect_dir.exists():
    best_files = list(detect_dir.glob("**/weights/best.pt"))
    if best_files:
        best_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        yolo_model_path = str(best_files[0])

custom_model = YOLO(yolo_model_path)''',
        
        "midas_depth": '''# Monocular Depth Estimation via Intel MiDaS
# Load model & transform from PyTorch Hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas.to(device).eval()

transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

# Run inference on RGB frame
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
input_batch = transform(img_rgb).to(device)
with torch.no_grad():
    depth_map = midas(input_batch)
    depth_map = depth_map.squeeze().cpu().numpy()''',
        
        "box_fusion": '''# Bounding Box & Depth Map Fusion
# Get bounding box center coordinates
center_x = int(np.clip((x1 + x2) // 2, 0, w - 1))
center_y = int(np.clip((y1 + y2) // 2, 0, h - 1))

# Look up disparity value at box center
depth_value = depth_map_resized[center_y, center_x]

# Render bounding box and distance overlay
cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
cv2.putText(
    annotated_frame, 
    f"{label}: {depth_value:.1f} cm", 
    (x1, y1 - 10), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.5, color, 2, cv2.LINE_AA
)''',
        
        "async_narrator": '''# Threaded Asynchronous Audio Narrator
class AudioNarrator:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            text = self.queue.get()
            if text is None: break
            
            # Generate speech
            temp_file = os.path.abspath(f"temp_announce_{int(time.time())}.mp3")
            tts = gTTS(text=text, lang='en')
            tts.save(temp_file)
            
            # Play using native Windows MCI API (completely non-blocking)
            alias = f"announcement_{int(time.time() * 1000)}"
            ctypes.windll.winmm.mciSendStringW(f'open "{temp_file}" type mpegvideo alias {alias}', None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f"play {alias} wait", None, 0, 0)
            ctypes.windll.winmm.mciSendStringW(f"close {alias}", None, 0, 0)
            os.remove(temp_file)''',
            
        "cooldown_priority": '''# Prioritization and Cooldown Logic
# Cooldown filter: 4.0 seconds per category
if label in self.last_spoken and (now - self.last_spoken[label]) < self.cooldown:
    continue

# Bounding box center depth lookup
depth_value = depth_map_resized[center_y, center_x]
candidates.append((depth_value, label))

# Sort: Higher disparity (closer) objects first
if candidates:
    candidates.sort(key=lambda x: x[0], reverse=True)
    closest_depth, closest_label = candidates[0]
    self.last_spoken[closest_label] = now
    
    # Generate Warning or Normal alert sentence
    if closest_depth > 800:
        sentence = f"Warning! {closest_label} is very close."
    else:
        sentence = f"{closest_label} detected at {closest_depth:.0f} cm."
    self.speak(sentence)'''
    }
    
    # Generate syntax highlighted images using Pygments
    for name, code in snippets.items():
        formatter = ImageFormatter(font_name='Consolas', font_size=18, line_number_chars=0, style='monokai')
        png_bytes = highlight(code, PythonLexer(), formatter)
        out_path = os.path.join("code_screenshots", f"{name}.png")
        with open(out_path, "wb") as f:
            f.write(png_bytes)
        print(f"Generated code screenshot: {out_path}")

if __name__ == "__main__":
    generate_screenshots()
