from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import math
import time
from ultralytics import YOLO
import mediapipe as mp
import json
import torch
import logging
import asyncio
from collections import deque

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt").to("cuda")
model.to("cuda")

# Load Silero VAD model
logger.info("Loading Silero VAD model...")
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
vad_model = vad_model.to("cuda") if torch.cuda.is_available() else vad_model
(get_speech_timestamps, _, vad_collect_chunks, _, _) = utils
logger.info("Silero VAD model loaded successfully")

# Object classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Find the index of "cell phone" in the classNames list
cell_phone_index = classNames.index("cell phone")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Sensitivity for detecting subtle movements
SENSITIVITY_THRESHOLD = 10  # Adjust based on your requirements

def get_head_direction(landmarks, width, height):
    """
    Detect subtle head movements for looking left, right, or straight.
    """
    left_eye_x = landmarks[33].x * width
    right_eye_x = landmarks[263].x * width
    nose_tip_x = landmarks[1].x * width
    midpoint_x = (left_eye_x + right_eye_x) / 2
    diff = nose_tip_x - midpoint_x

    if diff > SENSITIVITY_THRESHOLD:
        return "Looking Left"
    elif diff < -SENSITIVITY_THRESHOLD:
        return "Looking Right"
    else:
        return "Looking Straight"

@app.websocket("/video")
async def video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive video frames and send detection results.
    """
    await websocket.accept()
    logger.info("Video WebSocket client connected")
    last_direction_print_time = time.time()

    try:
        while True:
            # Receive frame as bytes
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array and decode the frame
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Received an invalid frame")
                continue

            # Perform YOLO inference
            results = model(img, stream=True)

            # Process detections
            cell_phone_boxes = []
            people_count = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Class index
                    cls = int(box.cls[0])

                    if cls == cell_phone_index:
                        # Bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cell_phone_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    elif cls == classNames.index("person"):
                        people_count += 1

            # Perform head direction detection
            head_direction = "Unknown"
            height, width, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_img)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    head_direction = get_head_direction(face_landmarks.landmark, width, height)

            # Send data back to the client
            data = {
                "cell_phone_boxes": cell_phone_boxes,
                "head_direction": head_direction,
                "people_count": people_count
            }

            await websocket.send_text(json.dumps(data))

    except WebSocketDisconnect:
        logger.info("Video WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in video processing: {e}")
    finally:
        await websocket.close()

class AudioProcessor:
    """
    Class to manage audio processing state and detection logic
    """
    def __init__(self, sampling_rate=16000, window_size_ms=200):
        self.sampling_rate = sampling_rate
        self.window_size_samples = int(sampling_rate * window_size_ms / 1000)
        self.audio_buffer = deque(maxlen=10)  # Store up to 10 chunks for context
        self.speech_detected = False
        self.speech_prob_threshold = 0.5
        self.speech_history = []  # Store recent speech detection results
        self.history_size = 5  # Number of frames to keep for smoothing
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.min_speech_frames = 3  # Minimum frames to confirm speech
        self.min_silence_frames = 10  # Minimum frames to confirm silence
        self.debug_counter = 0
    
    def add_audio_chunk(self, audio_chunk):
        """Add an audio chunk to the buffer"""
        self.audio_buffer.append(audio_chunk)
    
    def process_current_buffer(self):
        """Process the current audio buffer to detect speech"""
        # Debug logging periodically
        self.debug_counter += 1
        debug_log = (self.debug_counter % 50 == 0)
        
        if not self.audio_buffer:
            return False
        
        # Get the most recent audio chunk
        audio_np = self.audio_buffer[-1]
        
        try:
            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio_float)
            
            # Ensure tensor is the right shape
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Move to CUDA if available
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
            
            # Get speech probability
            speech_prob = vad_model(audio_tensor, self.sampling_rate).item()
            
            # Update speech history for smoothing
            self.speech_history.append(speech_prob > self.speech_prob_threshold)
            if len(self.speech_history) > self.history_size:
                self.speech_history.pop(0)
            
            # Determine current speech state with hysteresis
            current_frame_has_speech = speech_prob > self.speech_prob_threshold
            
            if current_frame_has_speech:
                self.consecutive_speech_frames += 1
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_frames += 1
                self.consecutive_speech_frames = 0
            
            # Apply state machine logic for speech detection
            if not self.speech_detected and self.consecutive_speech_frames >= self.min_speech_frames:
                self.speech_detected = True
                logger.info(f"Speech detected! Probability: {speech_prob:.3f}")
            elif self.speech_detected and self.consecutive_silence_frames >= self.min_silence_frames:
                self.speech_detected = False
                logger.info(f"Speech ended. Silence for {self.consecutive_silence_frames} frames")
            
            if debug_log:
                logger.debug(f"Audio processing stats: prob={speech_prob:.3f}, " 
                          f"speech_frames={self.consecutive_speech_frames}, "
                          f"silence_frames={self.consecutive_silence_frames}, "
                          f"detected={self.speech_detected}")
            
            return self.speech_detected
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {str(e)}")
            return False

@app.websocket("/audio")
async def audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Audio WebSocket client connected")
    
    # Create audio processor instance
    audio_processor = AudioProcessor()
    last_status_time = time.time()
    status_interval = 0.2  # Send status every 200ms
    
    try:
        while True:
            try:
                # Receive audio data as bytes - THIS CAN THROW WebSocketDisconnect
                audio_data = await websocket.receive_bytes()
                
                # Convert bytes to numpy array of int16
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Add to audio processor
                audio_processor.add_audio_chunk(audio_np)
                
                # Process and get speech detection result
                is_speaking = audio_processor.process_current_buffer()
                
                # Only send updates at the specified interval to avoid flooding
                current_time = time.time()
                if current_time - last_status_time >= status_interval:
                    status_text = "Speech detected" if is_speaking else "No speech detected"
                    
                    # Send results back to client
                    await websocket.send_json({
                        "type": "speech_detection",
                        "speech_detected": is_speaking,
                        "status": status_text,
                        "timestamp": current_time
                    })
                    
                    last_status_time = current_time
                
            except WebSocketDisconnect:
                # Move this to the inner try block to catch disconnect during receive
                logger.info("Audio WebSocket client disconnected")
                break  # Break the outer loop
            except asyncio.exceptions.CancelledError:
                logger.info("Audio task cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                # Continue processing next chunk instead of breaking

    except Exception as e:
        # This should now only catch other types of exceptions
        logger.error(f"Fatal error in audio websocket: {str(e)}", exc_info=True)
    finally:
        logger.info("Closing audio WebSocket connection")
        try:
            await websocket.close()
        except RuntimeError:
            # Handle case where socket is already closed
            pass

# To run this code, use: uvicorn filename:app --reload