import cv2
import mediapipe as mp
import numpy as np

# Function to list available cameras
def list_cameras():
    available_cameras = []
    for i in range(10):  # Check first 10 indexes
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:  # If camera is available
            available_cameras.append(i)
        cap.release()
    return available_cameras

# Get available cameras and let the user choose
cameras = list_cameras()
if not cameras:
    print("No camera found!")
    exit()

print("Available Cameras:", cameras)
camera_index = int(input(f"Select a camera index from {cameras}: "))

# Initialize Camera
cap = cv2.VideoCapture(camera_index)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Canvas
canvas = None  # Initialize the drawing canvas

# Define colors (Fixed Black & Added Red)
color_options = [(255, 255, 255), (50, 50, 50), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]  # White, Black, Blue, Green, Red, Eraser
draw_color = color_options[0]  # Default drawing color (White)

# Color Selection Blocks
color_positions = [(50, 50), (150, 50), (250, 50), (350, 50), (450, 50), (550, 50)]  # Positions of color blocks
color_labels = ["White", "Black", "Blue", "Green", "Red", "Clear"]  # Labels

# Variables for tracking finger movement
prev_x, prev_y = 0, 0  # Previous coordinates for drawing lines
finger_up = False  # Flag to check if index finger is up
space_active = False  # Flag to track space gesture (pinch gesture)
selected_index = -1  # Store the last selected color block index

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Exit if no frame is captured

    frame = cv2.flip(frame, 1)  # Flip the frame for a mirror effect
    if canvas is None:
        canvas = np.zeros_like(frame)  # Create a blank canvas

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    result = hands.process(rgb_frame)  # Process frame using MediaPipe Hands

    if result.multi_hand_landmarks:  # If a hand is detected
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []  # Store landmark coordinates
            h, w, _ = frame.shape  # Get frame dimensions

            for lm in hand_landmarks.landmark:
                lm_list.append((int(lm.x * w), int(lm.y * h)))  # Convert to pixel coordinates

            if lm_list:
                index_finger_tip = lm_list[8]  # Index Finger Tip
                thumb_tip = lm_list[4]  # Thumb Tip
                middle_finger_tip = lm_list[12]  # Middle Finger Tip

                # Draw a  pointer at the index finger_tip
                cv2.circle(frame, index_finger_tip, 7, draw_color, -1)


                # Detect Pinch Gesture (For Color Selection)
                pinch_distance = np.linalg.norm(np.array(index_finger_tip) - np.array(thumb_tip))

                if pinch_distance < 30:  # If pinch detected
                    space_active = True  # Space gesture activated

                    # Check if pinch is over a color selection block
                    for i, (x, y) in enumerate(color_positions):
                        if (x - 25 < index_finger_tip[0] < x + 25 and y - 25 < index_finger_tip[1] < y + 25):
                            if i < 5:  # Change drawing color
                                draw_color = color_options[i]
                            elif i == 5:  # If 'Clear' is selected
                                canvas = np.zeros_like(frame)  # Reset the canvas
                            selected_index = i  # Mark selected block

                else:
                    space_active = False  # Reset pinch gesture flag

                # Detect if Index Finger is Up (Writing Mode)
                if index_finger_tip[1] < middle_finger_tip[1] and not space_active:
                    if not finger_up:
                        prev_x, prev_y = 0, 0  # Reset previous coordinates
                    finger_up = True  # Index finger is up
                    x, y = index_finger_tip  # Get index finger coordinates
                    if prev_x and prev_y:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 5, cv2.LINE_AA)  # Draw line
                    prev_x, prev_y = x, y  # Update previous coordinates

                else:
                    finger_up = False  # Finger is down
                    prev_x, prev_y = 0, 0  # Reset previous coordinates

                # Eraser Mode: Detect if the whole hand is open
                finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
                finger_folded = sum(
                    1 for tip in finger_tips if lm_list[tip][1] > lm_list[tip - 2][1])  # Count folded fingers

                if finger_folded == 0:  # If all fingers (including thumb) are extended
                    cv2.circle(canvas, (index_finger_tip[0], index_finger_tip[1]), 40, (0, 0, 0), -1)  # Erase

    # Merge the canvas with the webcam frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Draw Color Selection Blocks
    for i, (x, y) in enumerate(color_positions):
        color = color_options[i] if i < 5 else (255, 255, 255)  # Get color for the block
        thickness = 2 if i == selected_index else 1  # Highlight if selected
        cv2.rectangle(frame, (x - 25, y - 25), (x + 25, y + 25), color, -1)  # Draw color block
        cv2.rectangle(frame, (x - 25, y - 25), (x + 25, y + 25), (255, 255, 255), thickness)  # Draw border
        cv2.putText(frame, color_labels[i], (x - 30, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # Label

    # Show the final output frame
    cv2.imshow("Virtual Writing Board", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to exit
        break
    elif key == ord('c'):  # Press 'c' to clear the board
        canvas = np.zeros_like(frame)

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
