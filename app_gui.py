import cv2
import numpy as np
import threading
import time
import queue
from deepface import DeepFace
from collections import defaultdict
import datetime
import os
import json
import traceback
from concurrent.futures import ThreadPoolExecutor

import customtkinter as ctk
from PIL import Image

# Optimize OpenCV for CPU multithreading so it doesn't fight with TensorFlow
cv2.setNumThreads(2)

# --- Configuration ---
STREAM_URL = 'rtsp://192.168.1.18:554/stream1'

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

BEHAVIOR_MAP = {
    "Man": {
        "angry": {
            "high": "HIGH RISK: Intense male aggression detected (Over 70%). Potential confrontation likely.",
            "low": "NOTICE: Male displaying mild frustration or annoyance."
        },
        "sad": {
            "high": "ALERT: Male exhibiting severe distress, disappointment or depression.",
            "low": "NOTICE: Male displaying signs of fatigue or slight sadness."
        },
        "happy": {
            "high": "INSIGHT: Male is highly engaged, agreeable, and expressing extreme joy.",
            "low": "INSIGHT: Male is displaying mild contentment or polite agreement."
        },
        "surprise": {
            "high": "INSIGHT: Male is highly shocked or startled.",
            "low": "INSIGHT: Male is mildly surprised or processing unexpected information."
        },
        "fear": {
            "high": "ALERT: Male is exhibiting intense fear, panic, or anxiety.",
            "low": "NOTICE: Male is displaying mild apprehension."
        },
        "disgust": {
            "high": "NOTICE: Male is showing strong aversion or disgust.",
            "low": "NOTICE: Male is mildly displeased."
        },
        "neutral": {
            "high": "INSIGHT: Male is completely disengaged and unreactive. Very low emotional pulse.",
            "low": "INSIGHT: Male is calm and composed."
        }
    },
    "Woman": {
        "angry": {
            "high": "HIGH RISK: Intense female aggression or anger detected (Over 70%).",
            "low": "NOTICE: Female displaying mild frustration or annoyance."
        },
        "sad": {
            "high": "ALERT: Female exhibiting severe distress, disappointment or depression.",
            "low": "NOTICE: Female displaying signs of fatigue or slight sadness."
        },
        "happy": {
            "high": "INSIGHT: Female is highly engaged, agreeable, and expressing extreme joy.",
            "low": "INSIGHT: Female is displaying mild contentment or polite agreement."
        },
        "surprise": {
            "high": "INSIGHT: Female is highly shocked or startled.",
            "low": "INSIGHT: Female is mildly surprised or processing unexpected information."
        },
        "fear": {
            "high": "ALERT: Female is exhibiting intense fear, panic, or anxiety.",
            "low": "NOTICE: Female is displaying mild apprehension."
        },
        "disgust": {
            "high": "NOTICE: Female is showing strong aversion or disgust.",
            "low": "NOTICE: Female is mildly displeased."
        },
        "neutral": {
            "high": "INSIGHT: Female is completely disengaged and unreactive. Very low emotional pulse.",
            "low": "INSIGHT: Female is calm and composed."
        }
    }
}

if not os.path.exists("recordings"):
    os.makedirs("recordings")

# --- Core Logic Classes ---
class VideoStreamWidget:
    def __init__(self, src, fallback_src=0):
        self.src = src
        self.fallback_src = fallback_src
        self.capture = None
        self.status = False
        self.frame = None
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        self.capture = cv2.VideoCapture(self.src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.capture.isOpened():
            print(f"[WARN] Failed to open stream {self.src}. Trying webcam...")
            self.capture = cv2.VideoCapture(self.fallback_src)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.started:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            else:
                time.sleep(0.1)

    def read(self):
        if self.frame is not None:
            return self.status, self.frame.copy()
        return False, None

    def release(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.capture:
            self.capture.release()

class SimpleTracker:
    def __init__(self):
        self.faces = {} 
        self.next_id = 1
        self.max_distance = 100
        self.analysis_queue = queue.Queue(maxsize=20)

    def update(self, new_boxes):
        updated_faces = {}
        unmatched_boxes = list(new_boxes)
        
        for fid, data in self.faces.items():
            if not unmatched_boxes: break
                
            best_match, best_dist = None, float('inf')
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
                'box': box, 'emotion': 'Analyzing...', 'gender': 'Analyzing...', 
                'emotion_conf': 0.0, 'gender_conf': 0.0,
                'scores': {}, 'history': [], 'last_analyze_time': 0
            }
            self.next_id += 1
            
        self.faces = updated_faces
        return self.faces

# --- Desktop Application GUI ---
class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("NeuroSight Pro - AI Behavioral Analytics")
        self.geometry("1600x900")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Grid Layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Variables
        self.is_running = False
        self.stream = None
        self.tracker = None
        self.face_net = None
        self.start_time = 0
        self.fps_history = []
        self.video_writer = None
        self.current_ctk_image = None
        self.ai_pool = None
        
        # Pause & Hover variables
        self.is_paused = False
        self.paused_frame = None
        self.paused_faces = {}
        self.hovered_face_id = None
        
        # Special Mode variables
        self.presentation_mode = False
        self.presentation_stats = {'engaged': 0, 'bored': 0, 'frustrated': 0, 'total_readings': 0}

        # Keyboard & Mouse Bindings
        self.bind('<space>', self.toggle_pause)

        self.setup_ui()
        self.load_models()

    def load_models(self):
        prototxt_path = "deploy.prototxt"
        model_path = "res10_300x300_ssd_iter_140000.caffemodel"
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            self.face_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            print("[ERROR] OpenCV SSD Face models not found.")

    def setup_ui(self):
        # --- Left Sidebar (Controls & Emotions) ---
        self.sidebar_left = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_left.grid(row=0, column=0, sticky="nsew")
        self.sidebar_left.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_left, text="NeuroSight Pro", font=ctk.CTkFont(size=26, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))
        
        self.subtitle_label = ctk.CTkLabel(self.sidebar_left, text="Real-Time Behavioral Analytics", font=ctk.CTkFont(size=13), text_color="gray")
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        self.start_btn = ctk.CTkButton(self.sidebar_left, text="Start Feed & Record", command=self.toggle_stream, height=45, font=ctk.CTkFont(size=14, weight="bold"))
        self.start_btn.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        self.mode_btn = ctk.CTkButton(self.sidebar_left, text="Start Presentation Mode", command=self.toggle_presentation_mode, height=45, fg_color="#8A2BE2", hover_color="#5D1049", font=ctk.CTkFont(size=14, weight="bold"))
        self.mode_btn.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

        self.stats_frame = ctk.CTkFrame(self.sidebar_left, fg_color="#1E1E1E", corner_radius=10)
        self.stats_frame.grid(row=4, column=0, padx=20, pady=20, sticky="ew")
        
        ctk.CTkLabel(self.stats_frame, text="Live Emotion Analytics", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 10))
        
        self.density_label = ctk.CTkLabel(self.stats_frame, text="Crowd Density: 0", font=ctk.CTkFont(size=15))
        self.density_label.pack(anchor="w", padx=20, pady=5)
        
        self.fps_label = ctk.CTkLabel(self.stats_frame, text="UI Performance: 0.0 FPS", font=ctk.CTkFont(size=15), text_color="#00ffcc")
        self.fps_label.pack(anchor="w", padx=20, pady=5)
        
        self.emotion_summary = ctk.CTkTextbox(self.stats_frame, height=180, fg_color="#2b2b2b", text_color="#FFFFFF", font=ctk.CTkFont(size=14))
        self.emotion_summary.pack(fill="x", padx=15, pady=15)
        self.emotion_summary.insert("0.0", "Waiting for data...")
        self.emotion_summary.configure(state="disabled")

        self.report_btn = ctk.CTkButton(self.sidebar_left, text="Generate Reports", command=self.generate_reports, fg_color="#1E90FF", hover_color="#1874CD", height=45)
        self.report_btn.grid(row=9, column=0, padx=20, pady=20, sticky="ew")

        # --- Main Video Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#0D0D0D")
        self.main_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.main_frame, text="Feed Offline", font=ctk.CTkFont(size=30, weight="bold"), text_color="gray")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        
        # Bind mouse movement for hover analysis
        self.video_label.bind('<Motion>', self.on_mouse_hover)

        # --- Right Sidebar (Gender & Presentation Stats) ---
        self.sidebar_right = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_right.grid(row=0, column=2, sticky="nsew")
        self.sidebar_right.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(self.sidebar_right, text="Demographics", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(30, 20))
        
        self.gender_frame = ctk.CTkFrame(self.sidebar_right, fg_color="#1E1E1E", corner_radius=10)
        self.gender_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.gender_frame, text="Male", font=ctk.CTkFont(size=15)).pack(anchor="w", padx=15, pady=(15, 0))
        self.male_count_label = ctk.CTkLabel(self.gender_frame, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color="#1E90FF")
        self.male_count_label.pack(anchor="w", padx=15)
        
        ctk.CTkLabel(self.gender_frame, text="Female", font=ctk.CTkFont(size=15)).pack(anchor="w", padx=15, pady=(10, 0))
        self.female_count_label = ctk.CTkLabel(self.gender_frame, text="0", font=ctk.CTkFont(size=24, weight="bold"), text_color="#FF69B4")
        self.female_count_label.pack(anchor="w", padx=15, pady=(0, 15))

        self.pres_frame = ctk.CTkFrame(self.sidebar_right, fg_color="#1E1E1E", corner_radius=10)
        self.pres_frame.grid(row=2, column=0, padx=20, pady=30, sticky="ew")
        
        ctk.CTkLabel(self.pres_frame, text="Psychological State", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 10))
        self.state_label = ctk.CTkLabel(self.pres_frame, text="Standby...", font=ctk.CTkFont(size=15), text_color="gray")
        self.state_label.pack(pady=10)
        
        self.engaged_bar = ctk.CTkProgressBar(self.pres_frame, progress_color="#00FF00")
        self.engaged_bar.pack(padx=15, pady=5)
        self.engaged_bar.set(0)
        self.engaged_label = ctk.CTkLabel(self.pres_frame, text="Engaged: 0%")
        self.engaged_label.pack()

        self.bored_bar = ctk.CTkProgressBar(self.pres_frame, progress_color="#FFA500")
        self.bored_bar.pack(padx=15, pady=5)
        self.bored_bar.set(0)
        self.bored_label = ctk.CTkLabel(self.pres_frame, text="Bored: 0%")
        self.bored_label.pack()

        # Instructions for Spacebar
        ctk.CTkLabel(self.sidebar_right, text="Hit SPACEBAR to pause feed & analyze", font=ctk.CTkFont(size=12, slant="italic"), text_color="gray").grid(row=9, column=0, pady=20)

    def toggle_pause(self, event=None):
        if not self.is_running: return
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.state_label.configure(text="PAUSED - Hover to Analyze", text_color="#FFA500")
        else:
            self.state_label.configure(text="Monitoring Audience..." if self.presentation_mode else "Standby...", text_color="#00FF00" if self.presentation_mode else "gray")
            self.hovered_face_id = None

    def on_mouse_hover(self, event):
        if not self.is_paused or self.paused_frame is None:
            self.hovered_face_id = None
            return
            
        fw = self.main_frame.winfo_width()
        fh = self.main_frame.winfo_height()
        label_w = fw - 20
        label_h = fh - 20
        
        # Calculate scaled coordinates from label to original 1024x768 frame
        scale_x = 1024.0 / max(1, label_w)
        scale_y = 768.0 / max(1, label_h)
        
        vid_x = int(event.x * scale_x)
        vid_y = int(event.y * scale_y)
        
        found = False
        for fid, data in self.paused_faces.items():
            x, y, w, h = data['box']
            if x <= vid_x <= x + w and y <= vid_y <= y + h:
                self.hovered_face_id = fid
                found = True
                break
                
        if not found:
            self.hovered_face_id = None

    def get_behavioral_insight(self, gender, emotion, conf):
        if gender not in BEHAVIOR_MAP or emotion not in BEHAVIOR_MAP[gender]:
            return "Gathering psychological insight..."
        
        level = "high" if conf > 70 else "low"
        return BEHAVIOR_MAP[gender][emotion][level]

    def toggle_presentation_mode(self):
        if not self.presentation_mode:
            self.presentation_mode = True
            self.mode_btn.configure(text="End Presentation", fg_color="#DC143C", hover_color="#B22222")
            self.presentation_stats = {'engaged': 0, 'bored': 0, 'frustrated': 0, 'total_readings': 0}
            self.state_label.configure(text="Monitoring Audience...", text_color="#00FF00")
        else:
            self.presentation_mode = False
            self.mode_btn.configure(text="Start Presentation Mode", fg_color="#8A2BE2", hover_color="#5D1049")
            self.state_label.configure(text="Analysis Complete", text_color="#1E90FF")
            self.generate_presentation_report()

    def generate_presentation_report(self):
        total = self.presentation_stats['total_readings']
        if total == 0: return
        engaged_pct = (self.presentation_stats['engaged'] / total) * 100
        bored_pct = (self.presentation_stats['bored'] / total) * 100
        frust_pct = (self.presentation_stats['frustrated'] / total) * 100
        
        report = [
            "# Presentation Psychological Analysis",
            f"**Total Audience Emotional Pulses Analyzed**: {total}",
            f"- **Highly Engaged/Interested**: {engaged_pct:.1f}%",
            f"- **Bored/Disengaged**: {bored_pct:.1f}%",
            f"- **Frustrated/Confused**: {frust_pct:.1f}%",
            "",
            "## Summary Verdict"
        ]
        
        if engaged_pct > 60:
            report.append("Excellent presentation! The audience was highly engaged and captivated.")
        elif bored_pct > 40:
            report.append("Audience showed signs of boredom. Consider adding more interactive elements or varying your tone.")
        else:
            report.append("Mixed reactions. The presentation held attention but lacked a strong emotional hook.")

        with open("presentation_analysis.md", "w") as f:
            f.write("\n".join(report))
        print("Presentation report saved to presentation_analysis.md")

    def warmup_ai_worker(self):
        """ Pre-loads TensorFlow models sequentially to prevent thread-crash on startup """
        try:
            print("[INFO] Pre-heating DeepFace Neural Engines...")
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.analyze(dummy_img, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
            print("[INFO] Neural Engines Online.")
        except Exception as e:
            pass
        finally:
            self.ai_ready = True

    def toggle_stream(self):
        if not self.is_running:
            self.start_btn.configure(text="Stop Feed & Record", fg_color="#DC143C", hover_color="#B22222")
            self.video_label.configure(text="Connecting to camera...", image=None)
            self.is_running = True
            self.is_paused = False
            self.hovered_face_id = None
            self.ai_ready = False
            
            self.stream = VideoStreamWidget(STREAM_URL)
            self.tracker = SimpleTracker()
            self.start_time = time.time()
            self.fps_history = []
            
            self.ai_pool = ThreadPoolExecutor(max_workers=3)
            
            self.detector_thread = threading.Thread(target=self.face_detector_worker, daemon=True)
            self.detector_thread.start()
            
            threading.Thread(target=self.deepface_queue_manager, daemon=True).start()
            threading.Thread(target=self.warmup_ai_worker, daemon=True).start()
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_writer = cv2.VideoWriter(f"recordings/feed_{timestamp}.avi", fourcc, 30.0, (1024, 768))

            self.update_video_ui()
        else:
            self.start_btn.configure(text="Start Feed & Record", fg_color=["#3a7ebf", "#1f538d"], hover_color=["#325882", "#14375e"])
            self.is_running = False
            if self.stream: self.stream.release()
            if self.video_writer: self.video_writer.release()
            if self.ai_pool: self.ai_pool.shutdown(wait=False)
            self.video_label.configure(image=None, text="Feed Offline")

    def face_detector_worker(self):
        while self.is_running:
            # Skip heavy detector logic if we are paused, save CPU
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            status, frame = self.stream.read()
            if status and frame is not None:
                frame = cv2.resize(frame, (1024, 768))
                ih, iw = frame.shape[:2]
                
                if self.face_net:
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.face_net.setInput(blob)
                    detections = self.face_net.forward()
                    
                    face_boxes = []
                    for i in range(0, detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            margin_x = int((endX - startX) * 0.2)
                            margin_y = int((endY - startY) * 0.2)
                            
                            x = max(0, startX - margin_x)
                            y = max(0, startY - margin_y)
                            w = min(iw - x, (endX - startX) + margin_x*2)
                            h = min(ih - y, (endY - startY) + margin_y*2)
                            
                            if w > 20 and h > 20:
                                face_boxes.append([x, y, w, h])
                    
                    tracked_faces = self.tracker.update(face_boxes)
                    
                    current_time = time.time()
                    for fid, data in tracked_faces.items():
                        if not self.ai_ready:
                            data['emotion'] = 'Booting AI...'
                            data['gender'] = 'Booting AI...'
                            continue
                            
                        frequency = 1.0 if len(tracked_faces) == 1 else 2.0
                        
                        if current_time - data.get('last_analyze_time', 0) > frequency:
                            x, y, w, h = data['box']
                            face_img = frame[y:y+h, x:x+w].copy()
                            if face_img.size > 0 and self.tracker.analysis_queue.qsize() < 5:
                                self.tracker.analysis_queue.put_nowait((face_img, fid))
                                data['last_analyze_time'] = current_time

            time.sleep(0.1)

    def deepface_queue_manager(self):
        while self.is_running:
            try:
                face_img, face_id = self.tracker.analysis_queue.get(timeout=1.0)
                if face_img is not None:
                    self.ai_pool.submit(self.run_deepface_inference, face_img, face_id)
            except queue.Empty:
                pass

    def run_deepface_inference(self, face_img, face_id):
        if not self.is_running: return
        try:
            result = DeepFace.analyze(face_img, actions=['emotion', 'gender'], enforce_detection=False, silent=True)
            if isinstance(result, list): result = result[0]
            
            emotion = result['dominant_emotion']
            gender = result['dominant_gender'] 
            
            emo_conf = result['emotion'].get(emotion, 0.0)
            gen_conf = result['gender'].get(gender, 0.0)
            
            if face_id in self.tracker.faces:
                self.tracker.faces[face_id]['emotion'] = emotion
                self.tracker.faces[face_id]['gender'] = gender
                self.tracker.faces[face_id]['emotion_conf'] = emo_conf
                self.tracker.faces[face_id]['gender_conf'] = gen_conf
                self.tracker.faces[face_id]['scores'] = result['emotion']
                self.tracker.faces[face_id]['history'].append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'emotion': emotion,
                    'gender': gender
                })

                if self.presentation_mode:
                    self.presentation_stats['total_readings'] += 1
                    if emotion in ['happy', 'surprise']:
                        self.presentation_stats['engaged'] += 1
                    elif emotion in ['neutral', 'sad']:
                        self.presentation_stats['bored'] += 1
                    elif emotion in ['angry', 'disgust', 'fear']:
                        self.presentation_stats['frustrated'] += 1

        except Exception as e:
            pass

    def draw_hover_insight(self, frame, data):
        """ Draws a beautiful tooltip overlay on the frame when hovering over a face """
        x, y, w, h = data['box']
        gender = data.get('gender', 'Analyzing...')
        emotion = data.get('emotion', 'Analyzing...')
        emo_conf = data.get('emotion_conf', 0.0)
        
        insight_msg = self.get_behavioral_insight(gender, emotion, emo_conf)
        
        # Overlay Box Parameters - Smaller and tighter
        box_width = 380
        box_height = 90
        
        # Position below the face by default
        oy = y + h + 15
        if oy + box_height > 768 - 10:  # If it goes off the bottom edge, put it above the face
            oy = max(10, y - box_height - 35) # 35 to account for the ID label above the face
            
        # Center horizontally with the face
        ox = x + (w // 2) - (box_width // 2)
        ox = max(10, min(ox, 1024 - box_width - 10)) # Keep within screen bounds
        
        # Create a translucent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (ox, oy), (ox + box_width, oy + box_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Draw Border
        cv2.rectangle(frame, (ox, oy), (ox + box_width, oy + box_height), (0, 255, 255), 2)
        
        # Draw Header
        title = f"SPECIAL ANALYSIS: ID {data.get('id', '?')}"
        cv2.putText(frame, title, (ox + 15, oy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        
        # Text Wrapping for Insight Message
        words = insight_msg.split(' ')
        lines = []
        current_line = []
        for word in words:
            current_line.append(word)
            (lw, _), _ = cv2.getTextSize(" ".join(current_line), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            if lw > box_width - 30:
                current_line.pop()
                lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            lines.append(" ".join(current_line))
            
        # Draw Text Lines
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (ox + 15, oy + 50 + (i*20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def update_video_ui(self):
        if not self.is_running: return

        if self.is_paused:
            # If paused, render the frozen frame
            if self.paused_frame is not None:
                frame = self.paused_frame.copy()
                tracked_faces = self.paused_faces
                
                # Dim the paused frame slightly for effect
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (1024, 768), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                cv2.putText(frame, "PAUSED - HOVER OVER FACES", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            else:
                self.after(15, self.update_video_ui)
                return
        else:
            # Live Feed
            status, frame = self.stream.read()
            if status and frame is not None:
                frame = cv2.resize(frame, (1024, 768))
                self.paused_frame = frame.copy()
                self.paused_faces = self.tracker.faces.copy()
                tracked_faces = self.paused_faces
            else:
                self.after(15, self.update_video_ui)
                return
        
        if not self.ai_ready and not self.is_paused:
            # Pulsating effect for "Pre-Heating" text
            alpha = (np.sin(time.time() * 5) + 1) / 2.0 
            color = (int(0 * alpha), int(165 * alpha), int(255 * alpha))
            cv2.putText(frame, "PRE-HEATING NEURAL ENGINES...", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        num_people = len(tracked_faces)
        emotion_counts = defaultdict(int)
        gender_counts = {'Male': 0, 'Female': 0}
        
        for fid, data in tracked_faces.items():
            data['id'] = fid # Inject ID for hover logic
            x, y, w, h = data['box']
            emotion = data['emotion']
            gender = data.get('gender', 'Analyzing...')
            emo_conf = data.get('emotion_conf', 0.0)
            gen_conf = data.get('gender_conf', 0.0)
            
            color = COLORS.get(emotion, (255, 255, 255))
            
            if emotion != 'Analyzing...': emotion_counts[emotion] += 1
            if gender == 'Man': gender_counts['Male'] += 1
            elif gender == 'Woman': gender_counts['Female'] += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            g_str = "M" if gender == "Man" else ("F" if gender == "Woman" else "?")
            if emotion == 'Analyzing...':
                label = f"ID:{fid} | Loading AI Models..."
            else:
                label = f"ID:{fid} | {g_str} ({int(gen_conf)}%) | {emotion.upper()} ({int(emo_conf)}%)"
                
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y = y - 15 if y - 15 > lh else y + h + 20
            cv2.rectangle(frame, (x, text_y - lh - 5), (x + lw + 10, text_y + 5), color, -1)
            cv2.putText(frame, label, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            # If hovering, draw special insight box
            if self.is_paused and self.hovered_face_id == fid:
                self.draw_hover_insight(frame, data)

        if not self.is_paused and self.video_writer:
            self.video_writer.write(frame)

        # UI Updates
        if not self.is_paused:
            t = time.time()
            self.fps_history.append(t)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            inst_fps = len(self.fps_history) / max(0.001, t - self.fps_history[0])
            self.fps_label.configure(text=f"UI Performance: {inst_fps:.1f} FPS")

        self.density_label.configure(text=f"Crowd Density: {num_people}")
        self.male_count_label.configure(text=str(gender_counts['Male']))
        self.female_count_label.configure(text=str(gender_counts['Female']))

        if self.presentation_mode and self.presentation_stats['total_readings'] > 0:
            tot = self.presentation_stats['total_readings']
            eng = self.presentation_stats['engaged'] / tot
            bor = self.presentation_stats['bored'] / tot
            self.engaged_bar.set(eng)
            self.bored_bar.set(bor)
            self.engaged_label.configure(text=f"Engaged: {int(eng*100)}%")
            self.bored_label.configure(text=f"Bored: {int(bor*100)}%")

        summary_text = ""
        for emo, count in emotion_counts.items():
            summary_text += f"{emo.capitalize()}: {count} people\n"
        
        self.emotion_summary.configure(state="normal")
        self.emotion_summary.delete("0.0", "end")
        self.emotion_summary.insert("0.0", summary_text if summary_text else "Tracking...")
        self.emotion_summary.configure(state="disabled")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        fw = self.main_frame.winfo_width()
        fh = self.main_frame.winfo_height()
        if fw > 10 and fh > 10:
            self.current_ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(fw-20, fh-20))
            self.video_label.configure(image=self.current_ctk_image, text="")

        self.after(15, self.update_video_ui)

    def generate_reports(self):
        if not self.tracker: return
        print("Generating Reports...")
        json_data = {"metadata": {"generated_on": datetime.datetime.now().isoformat(), "total_unique_people": len(self.tracker.faces)}, "people": {}}
        
        for fid, data in self.tracker.faces.items():
            history = data['history']
            if not history: continue
            emotions_only = [h['emotion'] for h in history]
            dominant = max(set(emotions_only), key=emotions_only.count)
            genders_only = [h['gender'] for h in history if h.get('gender') and h['gender'] != 'Analyzing...']
            dom_gen = max(set(genders_only), key=genders_only.count) if genders_only else "Unknown"

            json_data["people"][str(fid)] = {"dominant_emotion": dominant, "dominant_gender": dom_gen, "total_frames_analyzed": len(history), "timeline": history}
            
        with open("analysis_report.json", "w") as f:
            json.dump(json_data, f, indent=4)
        print("Report Saved to analysis_report.json")

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
