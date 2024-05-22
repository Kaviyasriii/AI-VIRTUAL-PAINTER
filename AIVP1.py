import cv2
import numpy as np
import mediapipe as mp

# Initialize the canvas
canvas_height, canvas_width = 720, 1280
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Initialize Mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initial brush parameters
brush_color = (0, 0, 255)  # Red
brush_thickness = 5

# Color palette (BGR format)
colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "black": (0, 0, 0)
}

def draw_palette(canvas):
    palette_height = 50
    offset = 10
    for i, (color_name, color_value) in enumerate(colors.items()):
        cv2.rectangle(canvas, (offset + i * 100, 0), (offset + (i + 1) * 100 - 10, palette_height), color_value, -1)

draw_palette(canvas)

cap = cv2.VideoCapture(0)
drawing = False
last_x, last_y = None, None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (canvas_width, canvas_height))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * canvas_width), int(index_finger_tip.y * canvas_height)
            
            if y < 50:  # Check if we are selecting a color from the palette
                for i, (color_name, color_value) in enumerate(colors.items()):
                    if 10 + i * 100 <= x <= 10 + (i + 1) * 100 - 10:
                        brush_color = color_value
            
            if cv2.waitKey(1) & 0xFF == ord(' '):  # Press space to start drawing
                drawing = not drawing
            
            if drawing:
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (x, y), brush_color, brush_thickness)
                last_x, last_y = x, y
            else:
                last_x, last_y = None, None

    frame_with_canvas = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow('Virtual Painter', frame_with_canvas)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):  # Press 'c' to clear canvas
        canvas[50:, :] = 255  # Clear the canvas but keep the palette
    elif key == ord('+'):
        brush_thickness += 1
    elif key == ord('-'):
        brush_thickness = max(1, brush_thickness - 1)

cap.release()
cv2.destroyAllWindows()
