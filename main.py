from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

app = FastAPI()

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/video")
async def video_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive video frames and print success on receipt.
    """
    await websocket.accept()
    try:
        while True:
            # Receive frame as bytes
            frame_bytes = await websocket.receive_bytes()

            # Convert bytes to a NumPy array
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            # Decode the frame using OpenCV to ensure it's valid (optional)
            img = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if img is None:
                print("Received an invalid frame")
            else:
                print("Received a valid frame")
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await websocket.close()
