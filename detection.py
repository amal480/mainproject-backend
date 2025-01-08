import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Camera capture
cap = cv2.VideoCapture(0)

# Define sensitivity threshold for detecting subtle head movements
SENSITIVITY_THRESHOLD = 10  # Adjust as needed for smaller/larger movements

def get_head_direction(landmarks, width, height):
    """
    Detect subtle head movements for looking left, right, or straight.

    :param landmarks: Facial landmarks
    :param width: Width of the video frame
    :param height: Height of the video frame
    :return: Direction string ("Looking Left", "Looking Right", "Looking Straight")
    """
    # Calculate positions of key landmarks
    left_eye_x = landmarks[33].x * width   # Left eye corner
    right_eye_x = landmarks[263].x * width # Right eye corner
    nose_tip_x = landmarks[1].x * width    # Nose tip

    # Calculate the midpoint between eyes
    midpoint_x = (left_eye_x + right_eye_x) / 2

    # Calculate the difference between the nose tip and the midpoint
    diff = nose_tip_x - midpoint_x

    # Detect subtle movements based on the sensitivity threshold
    if diff > SENSITIVITY_THRESHOLD:
        return "Looking Right"
    elif diff < -SENSITIVITY_THRESHOLD:
        return "Looking Left"
    else:
        return "Looking Straight"

drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Flip frame horizontally for a mirrored view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec,  # For individual landmarks
                drawing_spec   # For connections
            )

            # Determine head direction with subtle movement detection
            direction = get_head_direction(face_landmarks.landmark, width, height)

            # Display the direction on the frame
            cv2.putText(frame, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Head Movement Detection', frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
