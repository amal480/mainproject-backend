from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

app = FastAPI()

@app.post("/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    # Read image from file
    file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return {"faces": len(faces), "coordinates": [tuple(face) for face in faces]}
