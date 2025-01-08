from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import mediapipe as mp

app = FastAPI()

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
        return "Looking Right"
    elif diff < -SENSITIVITY_THRESHOLD:
        return "Looking Left"
    else:
        return "Looking Straight"

@app.websocket("/video")
async def video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive video frames and perform head movement detection.
    """
    await websocket.accept()
    try:
        while True:
            # Receive frame as bytes
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array and decode the frame
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if img is None:
                print("Received an invalid frame")
                continue

            # Convert the frame to RGB for Mediapipe
            height, width, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Perform face landmarks detection
            results = face_mesh.process(rgb_img)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get head direction
                    direction = get_head_direction(face_landmarks.landmark, width, height)

                    # Optionally, annotate the frame
                    cv2.putText(img, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    print(f"Head Direction: {direction}")  # Log direction

            # Optional: Save or display the processed frame
            # Uncomment this to save or debug locally
            # cv2.imshow("Processed Frame", img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()

# To run this code, use: uvicorn filename:app --reload
