import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def main():
    print("=== DWS Dual Model Live Webcam Demo ===")
    print("Initializing models. Please wait...")
    
    # 1. Load MiDaS depth model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas.to(device)
        midas.eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
        print("  [OK] MiDaS Depth model loaded.")
    except Exception as e:
        print(f"  [ERROR] Failed to load MiDaS: {e}")
        return
        
    # 2. Load COCO YOLO model for general objects (chairs, bottles, etc.)
    print("Loading COCO YOLO model...")
    try:
        coco_model = YOLO("yolov8n.pt")
        print("  [OK] COCO YOLO model loaded.")
    except Exception as e:
        print(f"  [ERROR] Failed to load COCO YOLO: {e}")
        return

    # 3. Load custom DWS YOLO model for doors, windows, stairs
    yolo_model_path = "yolov8n.pt"  # fallback
    detect_dir = Path("runs/detect")
    if detect_dir.exists():
        best_files = list(detect_dir.glob("**/weights/best.pt"))
        if best_files:
            # Sort by modification time to get the absolute newest weights
            best_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            yolo_model_path = str(best_files[0])
            
    print(f"Loading custom DWS YOLO model from: {yolo_model_path}...")
    try:
        custom_model = YOLO(yolo_model_path)
        print("  [OK] Custom DWS YOLO model loaded.")
    except Exception as e:
        print(f"  [ERROR] Failed to load Custom DWS YOLO: {e}")
        return

    # Optional Audio Feedback configuration
    audio_choice = input("Enable text-to-speech audio feedback? (y/n): ").strip().lower()
    use_audio = audio_choice == 'y'

    if use_audio:
        from narrator import AudioNarrator
        print("Audio feedback enabled.")
        narrator = AudioNarrator()
        narrator.speak("System initialized. Dual object detection and depth mapping activated.")
    else:
        print("Audio feedback disabled. Running in silent mode.")

    # 4. Open webcam or Direct IP stream
    print("\n====================================================")
    print("Select your camera input type:")
    print("1. Local Webcam / USB Camera / Virtual Drivers")
    print("2. Direct DroidCam IP Stream (Bypasses all driver locks)")
    print("====================================================")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        ip_input = input("Enter the WiFi IP shown on your phone DroidCam app (e.g. 192.168.1.5): ").strip()
        port = input("Enter the Port shown on your phone (default: 4747): ").strip()
        
        # Smart parsing in case IP was entered as IP:Port
        if ":" in ip_input:
            parts = ip_input.split(":")
            ip = parts[0]
            if not port and len(parts) > 1:
                port = parts[1]
        else:
            ip = ip_input
            
        if not port:
            port = "4747"
        
        # Connect directly to the phone's HTTP network stream
        stream_url = f"http://{ip}:{port}/video"
        print(f"Connecting directly to DroidCam stream at: {stream_url}...")
        cap = cv2.VideoCapture(stream_url)
    else:
        index_str = input("Enter camera index (0 for default, 1 or 2 for virtual cams): ").strip()
        try:
            index = int(index_str) if index_str else 0
        except ValueError:
            index = 0
        print(f"Opening local webcam at index {index}...")
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print("  [ERROR] Could not open camera source.")
        if use_audio:
            narrator.stop()
        return
        
    window_name = "DWS Live Depth & Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("Camera connected. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        h, w = frame.shape[:2]
        
        # Step 1: Run YOLO object detection on both models
        coco_results = coco_model.predict(source=frame, conf=0.5, verbose=False)[0]
        custom_results = custom_model.predict(source=frame, conf=0.3, verbose=False)[0]
        
        # Step 2: Run MiDaS depth estimation
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img_rgb).to(device)
        with torch.no_grad():
            depth_map = midas(input_batch)
            depth_map = depth_map.squeeze().cpu().numpy()
            
        depth_map_resized = cv2.resize(depth_map, (w, h))
        
        # Step 3: Draw boxes and distance text
        annotated_frame = frame.copy()
        
        # Define color codes: Blue (255, 0, 0) for COCO objects, Green (0, 255, 0) for Custom DWS objects
        all_detections = [
            (coco_results, 0.5, (255, 0, 0)),
            (custom_results, 0.3, (0, 255, 0))
        ]
        
        frame_detections = []  # Gather all detections on this frame for audio processing

        for result, conf_thresh, color in all_detections:
            for det in result.boxes.data.tolist():
                if len(det) < 6:
                    continue
                x1, y1, x2, y2, conf, cls = det
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                label = result.names[int(cls)]
                
                # Add to detections for narrator
                frame_detections.append((x1, y1, x2, y2, conf, label))

                # Bounding box center depth lookup
                center_x = int(np.clip((x1 + x2) // 2, 0, w - 1))
                center_y = int(np.clip((y1 + y2) // 2, 0, h - 1))
                depth_value = depth_map_resized[center_y, center_x]
                
                # Draw rectangle
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label overlay (displaying depth)
                text = f"{label}: {depth_value:.1f} cm"
                cv2.putText(
                    annotated_frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA
                )
            
        # Trigger background audio announcement (only if enabled)
        if use_audio:
            narrator.announce_objects(frame_detections, depth_map_resized)

        # Display the live window
        cv2.imshow(window_name, annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    if use_audio:
        narrator.stop()  # Clean up and stop the background thread
    print("Live demo closed.")

if __name__ == "__main__":
    main()
