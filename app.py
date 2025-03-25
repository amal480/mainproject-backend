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

# Initialize Mediapipe Face Mesh for head detection (do not modify)
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

# ------------------- Added Eye Gaze Direction Functionality -------------------

# Initialize a separate MediaPipe Face Mesh instance for eye gaze detection with refined landmarks
face_mesh_gaze = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Landmark indices for eye features
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_CORNERS = [33, 133]   # [right corner, left corner] of left eye
RIGHT_EYE_CORNERS = [362, 263] # [left corner, right corner] of right eye

def get_gaze_direction(eye_corners, iris_center, img_w, img_h):
    # Convert normalized coordinates to pixel values
    left_corner = (eye_corners[1].x * img_w, eye_corners[1].y * img_h)
    right_corner = (eye_corners[0].x * img_w, eye_corners[0].y * img_h)
    iris = (iris_center.x * img_w, iris_center.y * img_h)

    # Calculate horizontal direction
    eye_width = right_corner[0] - left_corner[0]
    if eye_width == 0:
        return "center"
    h_ratio = (iris[0] - left_corner[0]) / eye_width

    # Determine horizontal direction
    if h_ratio < 0.4:
        horizontal = "left"
    elif h_ratio > 0.6:
        horizontal = "right"
    else:
        horizontal = "center"

    return horizontal

@app.websocket("/gaze")
async def gaze_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive video frames and send eye gaze direction detection results.
    """
    await websocket.accept()
    logger.info("Gaze WebSocket client connected")
    try:
        while True:
            # Receive frame as bytes
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array and decode the frame
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("Received an invalid frame for gaze detection")
                continue

            # Flip and convert the frame for consistent view
            frame = cv2.flip(img, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            # Process eye gaze detection
            gaze_direction = "center"
            results = face_mesh_gaze.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get left eye features
                    left_corners = [face_landmarks.landmark[i] for i in LEFT_EYE_CORNERS]
                    left_iris = face_landmarks.landmark[LEFT_IRIS]
                    left_h = get_gaze_direction(left_corners, left_iris, img_w, img_h)

                    # Get right eye features
                    right_corners = [face_landmarks.landmark[i] for i in RIGHT_EYE_CORNERS]
                    right_iris = face_landmarks.landmark[RIGHT_IRIS]
                    right_h = get_gaze_direction(right_corners, right_iris, img_w, img_h)

                    # Combine left and right eye results
                    if left_h == right_h and left_h != "center":
                        h_dir = left_h
                    elif left_h != right_h:
                        h_dir = f"{left_h}/{right_h}"
                    else:
                        h_dir = "center"
                    gaze_direction = h_dir

            # Send the gaze direction back to the client
            data = {"gaze_direction": gaze_direction}
            await websocket.send_text(json.dumps(data))

    except WebSocketDisconnect:
        logger.info("Gaze WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in gaze detection: {e}")
    finally:
        await websocket.close()

# ------------------- End of Added Functionality -------------------

# The remaining audio-related endpoints remain commented out.
# To run this code, use: uvicorn filename:app --reload



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