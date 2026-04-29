import cv2
import numpy as np
import threading
import time
import queue
from deepface import DeepFace
from collections import defaultdict
import datetime
import os
import signal
import sys
import json

# --- Configuration ---
# RTSP stream URL
STREAM_URL = 'rtsp://192.168.1.18:554/stream1'

# Colors for bounding boxes (BGR)
COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 100, 0),    # Dark Green
    'fear': (128, 0, 128),     # Purple
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (0, 165, 255), # Orange
    'neutral': (200, 200, 200),# Gray
    'Analyzing...': (255, 255, 255)
}

running = True

def signal_handler(sig, frame):
    global running
    print("\n[INFO] Termination signal received (Ctrl+C). Saving reports and safely exiting...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# --- Asynchronous Video Stream Grabber ---
# This prevents the RTSP stream from buffering and lagging behind.
class VideoStreamWidget:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.status, self.frame = self.capture.read()
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.started:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            else:
                time.sleep(0.1)

    def read(self):
        if self.frame is not None:
            return self.status, self.frame.copy()
        return self.status, None

    def release(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()

# --- Tracker ---
class SimpleTracker:
    def __init__(self):
        self.faces = {} 
        self.next_id = 1
        self.max_distance = 80 # px distance for matching
        self.analysis_queue = queue.Queue()

    def update(self, new_boxes):
        updated_faces = {}
        unmatched_boxes = list(new_boxes)
        
        for fid, data in self.faces.items():
            if not unmatched_boxes:
                break
                
            best_match = None
            best_dist = float('inf')
            
            cx1 = data['box'][0] + data['box'][2]/2
            cy1 = data['box'][1] + data['box'][3]/2
            
            for box in unmatched_boxes:
                cx2 = box[0] + box[2]/2
                cy2 = box[1] + box[3]/2
                dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
                
                if dist < self.max_distance and dist < best_dist:
                    best_dist = dist
                    best_match = box
            
            if best_match is not None:
                unmatched_boxes = [b for b in unmatched_boxes if b != best_match]
                data['box'] = best_match
                updated_faces[fid] = data

        for box in unmatched_boxes:
            updated_faces[self.next_id] = {
                'box': box,
                'emotion': 'Analyzing...',
                'scores': {},
                'history': [],
                'last_analyze_time': 0
            }
            self.next_id += 1
            
        self.faces = updated_faces
        return self.faces

# --- Analysis Threads ---
def emotion_worker(tracker, worker_id):
    while running:
        try:
            # wait briefly to allow graceful shutdown
            face_img, face_id = tracker.analysis_queue.get(timeout=1.0)
            if face_img is None: 
                break
            
            try:
                # DeepFace analyze
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(result, list):
                    result = result[0]
                
                emotion = result['dominant_emotion']
                scores = result['emotion']
                
                if face_id in tracker.faces:
                    tracker.faces[face_id]['emotion'] = emotion
                    tracker.faces[face_id]['scores'] = scores
                    tracker.faces[face_id]['history'].append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'emotion': emotion
                    })
                    
            except Exception as e:
                pass
                
        except queue.Empty:
            pass
        except Exception as e:
            pass

def generate_reports(tracker):
    print("\nGenerating Detailed Reports (Markdown and JSON)...")
    
    # 1. Generate Markdown Report
    report_lines = ["# Crowd Emotion Analysis Report", f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    
    # 2. Generate JSON Report
    json_data = {
        "metadata": {
            "generated_on": datetime.datetime.now().isoformat(),
            "total_unique_people": len(tracker.faces)
        },
        "people": {}
    }
    
    for fid, data in tracker.faces.items():
        report_lines.append(f"## Person ID: {fid}")
        history = data['history']
        
        if not history:
            report_lines.append("No emotion data recorded.\n")
            continue
            
        emotions_only = [h['emotion'] for h in history]
        dominant = max(set(emotions_only), key=emotions_only.count)
        
        # Markdown
        report_lines.append(f"- **Dominant Emotion Overall**: {dominant.capitalize()}")
        report_lines.append(f"- **Total Frames Analyzed**: {len(history)}")
        report_lines.append(f"- **Timeline Summary**: ")
        
        step = max(1, len(history) // 10)
        for i in range(0, len(history), step):
            entry = history[i]
            time_str = entry['timestamp'].split('T')[1].split('.')[0]
            report_lines.append(f"  - {time_str}: {entry['emotion']}")
        report_lines.append("\n")
        
        # JSON
        json_data["people"][str(fid)] = {
            "dominant_emotion": dominant,
            "total_frames_analyzed": len(history),
            "timeline": history
        }
        
    with open("analysis_report.md", "w") as f:
        f.write("\n".join(report_lines))
        
    with open("analysis_report.json", "w") as f:
        json.dump(json_data, f, indent=4)
        
    print("[INFO] Reports saved successfully: analysis_report.md & analysis_report.json")

# --- Main Application ---
def main():
    global running
    print(f"[INFO] Initializing high-performance stream from: {STREAM_URL}")
    
    stream = VideoStreamWidget(STREAM_URL)
    
    time.sleep(2) # Give camera time to warm up
    status, frame = stream.read()
    
    if not status or frame is None:
        print(f"[WARN] Failed to open stream {STREAM_URL}. Falling back to webcam (0)...")
        stream.release()
        stream = VideoStreamWidget(0)
        time.sleep(2)
        status, frame = stream.read()
        if not status or frame is None:
            print("[ERROR] Could not open any video source.")
            return

    # Load high-accuracy OpenCV SSD Face Detector
    prototxt_path = "deploy.prototxt"
    model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
        print("[ERROR] High Accuracy Face Model files are missing!")
        print("They should have been downloaded by the AI. Re-run or download manually.")
        return
        
    face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    tracker = SimpleTracker()
    
    # Start 2 worker threads for faster emotion analysis (handles crowds better)
    for i in range(2):
        t = threading.Thread(target=emotion_worker, args=(tracker, i), daemon=True)
        t.start()

    print("\n[INFO] Application Live. Press 'q' or 'Ctrl+C' to quit and save reports.")
    
    frame_count = 0
    start_time = time.time()
    
    while running:
        status, frame = stream.read()
        if not status or frame is None:
            time.sleep(0.01)
            continue

        # Resize for display and processing
        frame = cv2.resize(frame, (1024, 768))
        ih, iw = frame.shape[:2]
        
        # Detect faces with OpenCV SSD (High Accuracy)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        face_boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # Filter out weak detections
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Expand box slightly for better emotion context
                x = max(0, startX - int((endX - startX) * 0.1))
                y = max(0, startY - int((endY - startY) * 0.1))
                w = min(iw - x, int((endX - startX) * 1.2))
                h = min(ih - y, int((endY - startY) * 1.2))
                
                if w > 20 and h > 20:
                    face_boxes.append([x, y, w, h])
        
        tracked_faces = tracker.update(face_boxes)
        
        num_people = len(tracked_faces)
        emotion_counts = defaultdict(int)
        
        current_time = time.time()
        for fid, data in tracked_faces.items():
            x, y, w, h = data['box']
            emotion = data['emotion']
            color = COLORS.get(emotion, (255, 255, 255))
            
            if emotion != 'Analyzing...':
                emotion_counts[emotion] += 1
            
            # Request new emotion analysis every 1 second
            if current_time - data.get('last_analyze_time', 0) > 1.0:
                face_img = frame[y:y+h, x:x+w].copy()
                if face_img.size > 0:
                    try:
                        # Queue size limit to avoid memory bloat
                        if tracker.analysis_queue.qsize() < 10:
                            tracker.analysis_queue.put_nowait((face_img, fid))
                            data['last_analyze_time'] = current_time
                    except queue.Full:
                        pass

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Label background
            label = f"ID:{fid} {emotion.capitalize()}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - lh - 10), (x + lw, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            if num_people == 1 and data['scores']:
                y_offset = 120
                cv2.rectangle(frame, (10, 80), (300, 300), (0, 0, 0), -1)
                cv2.putText(frame, "Detailed Analysis", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for emo, score in data['scores'].items():
                    text = f"{emo.capitalize()}: {score:.1f}%"
                    cv2.putText(frame, text, (20, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(emo, (255,255,255)), 2)
                    y_offset += 25

        if num_people > 1:
            summary = f"Crowd Density: {num_people} | " + " | ".join([f"{k.capitalize()}: {v}" for k, v in emotion_counts.items()])
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
            cv2.putText(frame, summary, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # FPS calculation
        fps = frame_count / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame_count += 1

        cv2.imshow("Crowd Behavior Analyzer (Real-Time)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            break

    # Cleanup
    stream.release()
    cv2.destroyAllWindows()
    
    generate_reports(tracker)

if __name__ == "__main__":
    main()

