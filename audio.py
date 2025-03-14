# server.py
import io
import torch
import numpy as np
from fastapi import FastAPI, WebSocket

app = FastAPI()

# Load Silero VAD model once at startup
print("Loading Silero VAD model...")
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad")
(get_speech_timestamps, _, _, _, _) = utils
print("Silero VAD model loaded.")

@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    chunk_count = 0
    try:
        while True:
            # Receive raw audio data
            audio_data = await websocket.receive_bytes()
            chunk_count += 1
            
            print(f"Received audio chunk {chunk_count}, size: {len(audio_data)} bytes")
            
            # Convert bytes to numpy array of int16
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize to [-1, 1]
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)  # Add channel dimension
            
            # Run Silero VAD
            speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000)
            
            # Log speech detection
            if speech_timestamps:
                print(f"Speech detected in chunk {chunk_count}!")
            
            # Send results back to client
            await websocket.send_json({"speech_timestamps": speech_timestamps})

    except Exception as e:
        print(f"Error in websocket connection: {e}")
        await websocket.close()
    finally:
        print("Client disconnected")
