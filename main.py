import cv2
import numpy as np
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional
import os
import argparse


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    """All settings"""
    
    # Paths
    MODEL_TFLITE = r"C:\Users\SUBBU\Downloads\Read Automation Internship\model_unquant.tflite"
    LABELS = r"C:\Users\SUBBU\Downloads\Read Automation Internship\labels.txt"
    MODEL_OPENVINO_XML = r"C:\Users\SUBBU\Downloads\Read Automation Internship\openvino_model\bottle_cap_inspection.xml"
    MODEL_OPENVINO_BIN = r"C:\Users\SUBBU\Downloads\Read Automation Internship\openvino_model\bottle_cap_inspection.bin"
    
    # Camera
    CAMERA_ID = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    
    # ROI
    ROI_WIDTH = 0.70
    ROI_HEIGHT = 0.75
    
    # Trigger
    TRIGGER_THRESHOLD = 15
    TRIGGER_MIN_PIXELS = 150
    
    # AI Thresholds - VERY LENIENT
    PASS_THRESHOLD = 0.40  # 40%
    WARNING_THRESHOLD = 0.40  # 40%
    CRITICAL_THRESHOLD = 0.40  # 40%
    
    # Display
    RESULT_HOLD_TIME = 2.0
    INPUT_SIZE = (224, 224)
    
    # Colors (BGR)
    COLOR_PASS = (0, 255, 0)
    COLOR_CRITICAL = (0, 0, 255)
    COLOR_WARNING = (0, 165, 255)
    COLOR_WAITING = (255, 255, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)


# ============================================================================
# DATA CLASSES
# ============================================================================
class DefectType(Enum):
    PASS = "PASS"
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    NONE = "NONE"


@dataclass
class Result:
    defect_type: DefectType
    confidence: float
    timestamp: float
    all_probs: dict


@dataclass
class Metrics:
    total: int = 0
    pass_count: int = 0
    critical_count: int = 0
    warning_count: int = 0
    fps: float = 0.0


# ============================================================================
# INFERENCE ENGINE
# ============================================================================
class InferenceEngine:
    """AI Engine - TFLite or OpenVINO"""
    
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        print(f"[INIT] Loading {engine_type.upper()} model...")
        
        # Load labels
        with open(Config.LABELS, 'r') as f:
            self.labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        
        if engine_type == "tflite":
            self._init_tflite()
        else:
            self._init_openvino()
        
        print(f"[INIT] Classes: {self.labels}")
    
    def _init_tflite(self):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=Config.MODEL_TFLITE)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("[INIT] TFLite Ready")
    
    def _init_openvino(self):
        from openvino.runtime import Core
        ie = Core()
        model = ie.read_model(Config.MODEL_OPENVINO_XML, Config.MODEL_OPENVINO_BIN)
        self.compiled_model = ie.compile_model(model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        print("[INIT] OpenVINO Ready")
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        img = cv2.resize(image, Config.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # CLAHE enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        
        # Normalize
        img = (img.astype(np.float32) / 127.5) - 1.0
        return np.expand_dims(img, axis=0)
    
    def predict(self, image: np.ndarray) -> dict:
        input_tensor = self.preprocess(image)
        
        if self.engine_type == "tflite":
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            output = self.compiled_model([input_tensor])[self.output_layer][0]
        
        # Softmax
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / np.sum(exp_output)
        
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}


# ============================================================================
# TRIGGER SYSTEM
# ============================================================================
class TriggerSystem:
    """Motion detection"""
    
    def __init__(self):
        self.prev_frame = None
        self.is_triggered = False
        self.last_motion_time = 0
    
    def check(self, roi: np.ndarray) -> Tuple[bool, bool]:
        """Returns: (should_inspect, object_present)"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, False
        
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, Config.TRIGGER_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        changed = np.sum(thresh > 0)
        
        has_object = np.mean(gray) > 30
        
        if changed > Config.TRIGGER_MIN_PIXELS:
            self.is_triggered = True
            self.last_motion_time = time.time()
            self.prev_frame = gray
            return True, True
        
        time_since = time.time() - self.last_motion_time
        object_present = has_object and time_since < Config.RESULT_HOLD_TIME
        
        if not object_present:
            self.is_triggered = False
        
        self.prev_frame = gray
        return False, object_present


# ============================================================================
# HMI
# ============================================================================
class HMI:
    """Display interface"""
    
    @staticmethod
    def draw_header(frame, engine):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (40, 40, 40), -1)
        cv2.putText(frame, "READ AUTOMATION", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 255), 3)
        color = (0, 255, 100) if engine == "openvino" else (255, 255, 255)
        cv2.putText(frame, f"Engine: {engine.upper()}", (500, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    @staticmethod
    def draw_metrics(frame, metrics):
        x, y = frame.shape[1] - 350, 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (x-10, y-10), (frame.shape[1]-10, y+160), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        texts = [
            f"FPS: {metrics.fps:.1f}",
            f"Total: {metrics.total}",
            f"PASS: {metrics.pass_count}",
            f"CRITICAL: {metrics.critical_count}",
            f"WARNING: {metrics.warning_count}",
            f"Yield: {(metrics.pass_count/metrics.total*100 if metrics.total > 0 else 0):.1f}%"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(frame, text, (x, y + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLOR_WHITE, 1)
    
    @staticmethod
    def draw_roi(frame, coords, triggered):
        x1, y1, x2, y2 = coords
        color = Config.COLOR_RED if triggered else Config.COLOR_GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if triggered else 2)
        label = "INSPECTION ZONE" if triggered else "TRIGGER ZONE"
        cv2.putText(frame, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    @staticmethod
    def draw_result(frame, result, is_waiting):
        cx = frame.shape[1] // 2
        cy = frame.shape[0] // 2 + 100
        
        if is_waiting:
            cv2.putText(frame, "WAITING FOR BOTTLE...", (cx-250, cy),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, Config.COLOR_WAITING, 3)
        elif result:
            if result.defect_type == DefectType.PASS:
                status, color = "PASS", Config.COLOR_PASS
            elif result.defect_type == DefectType.CRITICAL:
                status, color = "CRITICAL DEFECT", Config.COLOR_CRITICAL
            elif result.defect_type == DefectType.WARNING:
                status, color = "WARNING - CHECK CAP", Config.COLOR_WARNING
            else:
                return
            
            cv2.putText(frame, status, (cx-200, cy),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 4)
            cv2.putText(frame, f"Confidence: {result.confidence*100:.1f}%", (cx-150, cy+40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, Config.COLOR_WHITE, 2)


# ============================================================================
# MAIN SYSTEM
# ============================================================================
class InspectionSystem:
    """Complete inspection system"""
    
    def __init__(self, engine_type="tflite"):
        print("\n" + "="*70)
        print("READ AUTOMATION - INDUSTRIAL QC SYSTEM v2.0")
        print(f"ENGINE: {engine_type.upper()}")
        print("="*70)
        
        self.engine_type = engine_type
        self.ai = InferenceEngine(engine_type)
        self.trigger = TriggerSystem()
        self.metrics = Metrics()
        
        self.fps_tracker = deque(maxlen=30)
        self.last_result = None
        self.result_time = 0
        self.last_inference_time = 0
        
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        
        print("[INIT] Ready! Press 'q' to quit\n")
    
    def get_roi_coords(self, frame_shape):
        h, w = frame_shape[:2]
        roi_w = int(w * Config.ROI_WIDTH)
        roi_h = int(h * Config.ROI_HEIGHT)
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        return (x1, y1, x1 + roi_w, y1 + roi_h)
    
    def classify(self, probs: dict) -> Result:
        """SIMPLE classification logic"""
        
        # Get highest probability
        top_class = max(probs.keys(), key=lambda k: probs[k])
        top_conf = probs[top_class]
        
        print(f"\n{'='*60}")
        print(f"[DETECTION]")
        for cls, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 20)
            print(f"  {cls:20s}: {prob*100:5.1f}% {bar}")
        
        # Classify based on highest class
        if top_class == "OK_Good" and top_conf > Config.PASS_THRESHOLD:
            defect_type = DefectType.PASS
            self.metrics.pass_count += 1
            print(f"[RESULT] PASS ✅")
        
        elif top_class == "Defect_Missing" and top_conf > Config.CRITICAL_THRESHOLD:
            defect_type = DefectType.CRITICAL
            self.metrics.critical_count += 1
            print(f"[RESULT] CRITICAL ❌")
        
        elif top_class == "Defect_High" and top_conf > Config.WARNING_THRESHOLD:
            defect_type = DefectType.WARNING
            self.metrics.warning_count += 1
            print(f"[RESULT] WARNING ⚠️")
        
        else:
            defect_type = DefectType.NONE
            print(f"[RESULT] IGNORED (too uncertain)")
        
        print(f"{'='*60}")
        
        if defect_type != DefectType.NONE:
            self.metrics.total += 1
        
        return Result(
            defect_type=defect_type,
            confidence=top_conf,
            timestamp=time.time(),
            all_probs=probs
        )
    
    def run(self):
        frame_start = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                roi_coords = self.get_roi_coords(frame.shape)
                x1, y1, x2, y2 = roi_coords
                roi = frame[y1:y2, x1:x2]
                
                should_inspect, object_present = self.trigger.check(roi)
                current_time = time.time()
                
                # Run AI when triggered
                if should_inspect and (current_time - self.last_inference_time) > 0.3:
                    probs = self.ai.predict(roi)
                    result = self.classify(probs)
                    
                    if result.defect_type != DefectType.NONE:
                        self.last_result = result
                        self.result_time = current_time
                    
                    self.last_inference_time = current_time
                
                # Display logic
                is_waiting = not self.trigger.is_triggered and not object_present
                show_result = (self.last_result and 
                             (object_present or (current_time - self.result_time) < Config.RESULT_HOLD_TIME))
                
                if not object_present and (current_time - self.result_time) > Config.RESULT_HOLD_TIME:
                    self.last_result = None
                
                # Draw UI
                HMI.draw_header(frame, self.engine_type)
                HMI.draw_metrics(frame, self.metrics)
                HMI.draw_roi(frame, roi_coords, self.trigger.is_triggered)
                HMI.draw_result(frame, self.last_result if show_result else None, is_waiting)
                
                cv2.imshow("READ Automation QC System", frame)
                
                # Update FPS
                frame_time = time.time() - frame_start
                self.fps_tracker.append(1.0 / frame_time if frame_time > 0 else 0)
                self.metrics.fps = np.mean(self.fps_tracker)
                frame_start = time.time()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("SESSION SUMMARY")
            print("="*70)
            print(f"Total Inspected: {self.metrics.total}")
            print(f"PASS: {self.metrics.pass_count}")
            print(f"CRITICAL: {self.metrics.critical_count}")
            print(f"WARNING: {self.metrics.warning_count}")
            print(f"Yield: {(self.metrics.pass_count/self.metrics.total*100 if self.metrics.total > 0 else 0):.1f}%")
            print(f"Average FPS: {self.metrics.fps:.1f}")
            print("="*70)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='tflite', choices=['tflite', 'openvino'])
    args = parser.parse_args()
    
    try:
        system = InspectionSystem(args.engine)
        system.run()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()