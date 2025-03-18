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
import torchaudio
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Load YOLO model
model = YOLO("yolo-Weights/yolov8n.pt").to("cuda")
model.to("cuda")

# # Load Silero VAD model
# logger.info("Loading Silero VAD model...")
# vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
# vad_model = vad_model.to("cuda") if torch.cuda.is_available() else vad_model
# (get_speech_timestamps, _, vad_collect_chunks, _, _) = utils
# logger.info("Silero VAD model loaded successfully")

# Load Silero VAD model once at startup
# print("Loading Silero VAD model...")
# vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=True)
# (get_speech_timestamps, _, _, _, _) = utils
# print("Silero VAD model loaded.")

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

# @app.websocket("/audio")
# async def audio_endpoint(websocket: WebSocket):
#     """
#     WebSocket endpoint to receive audio chunks and perform speech detection using Silero VAD.
#     """
#     await websocket.accept()
#     logger.info("Audio WebSocket client connected")

#     try:
#         while True:
#             # Receive audio chunk as bytes
#             audio_data = await websocket.receive_bytes()
            
#             # Convert bytes to tensor
#             audio_np = np.frombuffer(audio_data, dtype=np.int16)
#             audio_tensor = torch.from_numpy(audio_np).float() / 32768.0  # Normalize to [-1, 1]
#             audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension

#             # Process with Silero VAD
#             speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)

#             # Determine if speech is detected
#             is_speaking = len(speech_timestamps) > 0
#             status_text = "Speech detected" if is_speaking else "No speech detected"

#             # Send result back to frontend
#             await websocket.send_json({
#                 "type": "speech_detection",
#                 "speech_detected": is_speaking,
#                 "status": status_text,
#             })

#     except WebSocketDisconnect:
#         logger.info("Audio WebSocket disconnected")
#     except Exception as e:
#         logger.error(f"Error in audio WebSocket: {e}")
#     finally:
#         await websocket.close()

# @app.websocket("/audio")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Client connected")
    
#     chunk_count = 0
#     try:
#         while True:
#             # Receive raw audio data
#             audio_data = await websocket.receive_bytes()
#             chunk_count += 1
            
#             print(f"Received audio chunk {chunk_count}, size: {len(audio_data)} bytes")
            
#             # Convert bytes to numpy array of int16
#             audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
#             # Convert to float32 and normalize to [-1, 1]
#             audio_float = audio_np.astype(np.float32) / 32768.0
            
#             # Convert to PyTorch tensor
#             audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)  # Add channel dimension
            
#             # Run Silero VAD
#             speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
            
#             # Log speech detection
#             if speech_timestamps:
#                 print(f"Speech detected in chunk {chunk_count}!")
            
#             # Send results back to client
#             await websocket.send_json({"speech_timestamps": speech_timestamps})

#     except Exception as e:
#         print(f"Error in websocket connection: {e}")
#         await websocket.close()
#     finally:
#         print("Client disconnected")



# To run this code, use: uvicorn filename:app --reload