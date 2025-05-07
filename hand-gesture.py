import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time
import pickle
import os
from datetime import datetime
import threading
import pyautogui  # For computer control capabilities

# Initialize MediaPipe Hands with improved settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6,  # Increased for better tracking
    model_complexity=1  # Higher model complexity for better accuracy
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define colors in BGR format
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)

# Create folders for saving gestures and recordings
SAVE_DIR = "hand_gesture_data"
RECORDINGS_DIR = os.path.join(SAVE_DIR, "recordings")
CUSTOM_GESTURES_DIR = os.path.join(SAVE_DIR, "custom_gestures")

for directory in [SAVE_DIR, RECORDINGS_DIR, CUSTOM_GESTURES_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Tracking for gesture history and smoothing
gesture_history = deque(maxlen=15)  # Increased for better smoothing
hand_position_history = deque(maxlen=10)  # For position smoothing
gesture_start_time = None
gesture_hold_threshold = 1.5  # seconds to hold a gesture to trigger an action

# Drawing settings
drawing_points = []
stroke_history = []  # To enable undo function
current_stroke = []

# Custom gesture storage
custom_gestures = {}
recording_gesture = False
custom_gesture_samples = []
recording_countdown = 0

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    angle_radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle_degrees = math.degrees(angle_radians)
    
    # Normalize angle to 0-360 range
    if angle_degrees < 0:
        angle_degrees += 360
    
    return angle_degrees

# Function to recognize hand gestures with improved detection
def recognize_gesture(hand_landmarks, hand_type="Right"):
    finger_tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_bases = [2, 5, 9, 13, 17]      # Base of fingers
    fingers = []
    
    # Check hand orientation (right or left)
    is_right_hand = hand_type == "Right"
    
    # Calculate hand center for reference
    hand_center_x = sum([hand_landmarks.landmark[i].x for i in range(21)]) / 21
    hand_center_y = sum([hand_landmarks.landmark[i].y for i in range(21)]) / 21
    
    # Get wrist position for reference
    wrist = hand_landmarks.landmark[0]
    
    # Improved thumb detection based on orientation
    # Thumb orientation is different for left and right hands
    # Also considering the angle between joints for more accuracy
    thumb_tip = hand_landmarks.landmark[finger_tips_ids[0]]
    thumb_ip = hand_landmarks.landmark[3]  # IP joint
    thumb_mcp = hand_landmarks.landmark[2]  # MCP joint
    thumb_cmc = hand_landmarks.landmark[1]  # CMC joint
    
    # Calculate thumb angle
    thumb_angle = calculate_angle(thumb_cmc, thumb_mcp, thumb_tip)
    
    # Improved thumb detection logic
    if is_right_hand:
        # For right hand: thumb is extended if the tip is to the left of the IP joint
        # and the angle is appropriate
        if (thumb_tip.x < thumb_ip.x and 80 < thumb_angle < 270) or calculate_distance(thumb_tip, wrist) > calculate_distance(thumb_mcp, wrist):
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        # For left hand: thumb is extended if the tip is to the right of the IP joint
        # and the angle is appropriate
        if (thumb_tip.x > thumb_ip.x and (thumb_angle < 80 or thumb_angle > 270)) or calculate_distance(thumb_tip, wrist) > calculate_distance(thumb_mcp, wrist):
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Improved detection for other 4 fingers with consideration for curled fingers
    for id in range(1, 5):
        # Get finger joints
        tip = hand_landmarks.landmark[finger_tips_ids[id]]
        dip = hand_landmarks.landmark[finger_tips_ids[id] - 1]
        pip = hand_landmarks.landmark[finger_tips_ids[id] - 2]
        mcp = hand_landmarks.landmark[finger_tips_ids[id] - 3]
        
        # Calculate vertical distances
        mcp_to_pip_distance = calculate_distance(mcp, pip)
        
        # A finger is extended if:
        # 1. The finger tip is higher (lower y-value) than the PIP joint
        # 2. The finger is not bent too much sideways
        if (tip.y < pip.y and  # Tip is above PIP
            abs(tip.x - mcp.x) < 0.2):  # Not bent too much sideways relative to MCP
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Calculate additional parameters for advanced gesture recognition
    # Distance between fingertips for pinch detection
    thumb_to_index_distance = calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8])
    thumb_to_middle_distance = calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[12])
    thumb_to_ring_distance = calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[16])
    thumb_to_pinky_distance = calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[20])
    
    # Index finger tip position for pointing gestures
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    index_finger_mcp = hand_landmarks.landmark[5]
    
    # Get gesture point (for drawing and interaction)
    gesture_point = (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    
    # Get total extended fingers
    total_fingers = fingers.count(1)
    
    # Check for custom gestures first if defined
    if custom_gestures:
        for name, gesture_data in custom_gestures.items():
            # Simple matching algorithm for custom gestures
            # Can be improved with machine learning for better accuracy
            match_score = sum([1 for i, v in enumerate(fingers) if v == gesture_data['fingers'][i]])
            if match_score >= 4:  # At least 4 fingers match
                return name, total_fingers, gesture_point
    
    # Better gesture detection with more detailed patterns
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist", total_fingers, gesture_point
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumb Up", total_fingers, gesture_point
    elif fingers == [0, 1, 0, 0, 0]:
        return "Index Pointing", total_fingers, gesture_point
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign", total_fingers, gesture_point
    elif fingers == [0, 1, 1, 1, 0]:
        return "Three Fingers", total_fingers, gesture_point
    elif fingers == [0, 1, 1, 1, 1]:
        return "Four Fingers", total_fingers, gesture_point
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand", total_fingers, gesture_point
    elif fingers == [1, 1, 0, 0, 1]:
        return "Rock Sign", total_fingers, gesture_point
    elif fingers == [0, 1, 0, 0, 1]:
        return "Spider-Man", total_fingers, gesture_point
    elif fingers == [1, 0, 0, 0, 1]:
        return "Hang Loose", total_fingers, gesture_point
    
    # Enhanced pinch detection with different finger combinations
    if thumb_to_index_distance < 0.05:
        return "Index Pinch", total_fingers, gesture_point
    elif thumb_to_middle_distance < 0.05:
        return "Middle Pinch", total_fingers, gesture_point
    elif thumb_to_ring_distance < 0.05:
        return "Ring Pinch", total_fingers, gesture_point
    elif thumb_to_pinky_distance < 0.05:
        return "Pinky Pinch", total_fingers, gesture_point
    
    # OK sign detection (thumb and index form a circle)
    if 0.03 < thumb_to_index_distance < 0.07 and fingers[1] == 1:
        return "OK Sign", total_fingers, gesture_point
    
    # Dynamic pointing (vector from wrist to index finger)
    if fingers[1] == 1 and all(f == 0 for f in [fingers[0], fingers[2], fingers[3], fingers[4]]):
        pointing_direction = ""
        
        # Determine pointing direction relative to wrist
        if index_finger_tip.y < wrist.y - 0.1:
            pointing_direction = "Up"
        elif index_finger_tip.y > wrist.y + 0.1:
            pointing_direction = "Down"
        
        if index_finger_tip.x < wrist.x - 0.1:
            pointing_direction += " Left" if is_right_hand else " Right"
        elif index_finger_tip.x > wrist.x + 0.1:
            pointing_direction += " Right" if is_right_hand else " Left"
        
        if pointing_direction:
            return f"Pointing {pointing_direction}", total_fingers, gesture_point
    
    # If no specific gesture is recognized, return number of fingers
    return f"{total_fingers} Fingers", total_fingers, gesture_point

# Function to smooth hand position using moving average
def smooth_position(position_history, new_point):
    position_history.append(new_point)
    if len(position_history) < 2:
        return new_point
    
    # Calculate moving average for x and y coordinates
    avg_x = sum(p[0] for p in position_history) / len(position_history)
    avg_y = sum(p[1] for p in position_history) / len(position_history)
    
    return (int(avg_x), int(avg_y))

# Function to display information on screen with improved UI
def display_info(image, gesture, fingers_count, fps, mode, recording_status=""):
    # Create semi-transparent overlay for info panel
    overlay = image.copy()
    
    # Main info panel
    cv2.rectangle(overlay, (10, 10), (320, 160), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.rectangle(image, (10, 10), (320, 160), WHITE, 2)
    
    # FPS with color based on performance
    fps_color = GREEN if fps > 25 else YELLOW if fps > 15 else RED
    cv2.putText(image, f'FPS: {fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)
    
    # Fingers count
    cv2.putText(image, f'Fingers: {fingers_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    
    # Current gesture
    cv2.putText(image, f'Gesture: {gesture}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PURPLE, 2)
    
    # Current mode
    cv2.putText(image, f'Mode: {mode}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
    
    # Show recording status if active
    if recording_status:
        # Flashing recording indicator
        indicator_color = RED if int(time.time() * 2) % 2 == 0 else WHITE
        cv2.circle(image, (width - 30, 30), 10, indicator_color, -1)
        cv2.putText(image, recording_status, (width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, indicator_color, 2)
    
    # Help panel on bottom right
    help_y_start = height - 180
    cv2.rectangle(overlay, (width - 280, help_y_start), (width - 10, height - 10), BLACK, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.rectangle(image, (width - 280, help_y_start), (width - 10, height - 10), WHITE, 1)
    
    # Help text
    help_text = [
        "M: Change Mode",
        "C: Clear Canvas",
        "U: Undo Stroke",
        "S: Save Drawing",
        "R: Record Custom Gesture",
        "P: Take Screenshot",
        "V: Start/Stop Video",
        "ESC: Quit"
    ]
    
    for i, text in enumerate(help_text):
        cv2.putText(image, text, (width - 270, help_y_start + 25 + i*20), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)

# Function for virtual painting/drawing with improved features
def virtual_painter(canvas, finger_point, prev_point, color, thickness=10, mode="pen"):
    if prev_point is None:
        return finger_point
    
    if mode == "pen":
        cv2.line(canvas, finger_point, prev_point, color, thickness)
    elif mode == "brush":
        # Create a gradient brush effect
        for i in range(1, thickness+1, 2):
            alpha = (thickness - i) / thickness
            blend_color = tuple(int((1-alpha)*c + alpha*255) for c in color)
            cv2.line(canvas, finger_point, prev_point, blend_color, i)
    elif mode == "eraser":
        # Use a circular eraser
        cv2.circle(canvas, finger_point, thickness, (0, 0, 0), -1)
    
    return finger_point

# Function to save canvas as image
def save_canvas(canvas):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SAVE_DIR, f"drawing_{timestamp}.png")
    cv2.imwrite(filename, canvas)
    return filename

# Function to record and save custom gesture
def save_custom_gesture(name, fingers_pattern):
    custom_gestures[name] = {
        'fingers': fingers_pattern,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to file
    with open(os.path.join(CUSTOM_GESTURES_DIR, f"{name}.pkl"), 'wb') as f:
        pickle.dump(custom_gestures[name], f)
    
    return True

# Load existing custom gestures if available
def load_custom_gestures():
    if not os.path.exists(CUSTOM_GESTURES_DIR):
        return {}
    
    gestures = {}
    for filename in os.listdir(CUSTOM_GESTURES_DIR):
        if filename.endswith('.pkl'):
            gesture_name = os.path.splitext(filename)[0]
            try:
                with open(os.path.join(CUSTOM_GESTURES_DIR, filename), 'rb') as f:
                    gestures[gesture_name] = pickle.load(f)
            except Exception as e:
                print(f"Error loading gesture {gesture_name}: {e}")
    
    return gestures

# Video recording class
class VideoRecorder:
    def __init__(self, width, height, fps=20):
        self.width = width
        self.height = height
        self.fps = fps
        self.recording = False
        self.output = None
        self.thread = None
        self.frames = []
        self.start_time = None
    
    def start(self):
        if self.recording:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
        self.recording = True
        self.start_time = time.time()
        self.frames = []
        
        # Start recording in a separate thread
        self.thread = threading.Thread(target=self._record_thread)
        self.thread.daemon = True
        self.thread.start()
        
        return filename
    
    def add_frame(self, frame):
        if self.recording:
            self.frames.append(frame.copy())
    
    def _record_thread(self):
        while self.recording:
            if len(self.frames) > 0:
                frame = self.frames.pop(0)
                if self.output is not None:
                    self.output.write(frame)
            else:
                time.sleep(0.01)
    
    def stop(self):
        if not self.recording:
            return
        
        self.recording = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.output:
            self.output.release()
            self.output = None
        
        duration = time.time() - self.start_time if self.start_time else 0
        self.start_time = None
        
        return duration

# Function to handle computer control actions
def perform_computer_action(gesture, position, screen_width, screen_height):
    # Map hand position to screen coordinates
    screen_x = int(position[0] * screen_width / width)
    screen_y = int(position[1] * screen_height / height)
    
    if gesture == "Index Pointing":
        # Move mouse cursor
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
    elif gesture == "Index Pinch":
        # Click
        pyautogui.click(screen_x, screen_y)
    elif gesture == "Peace Sign":
        # Right-click
        pyautogui.rightClick(screen_x, screen_y)
    elif gesture == "Fist" and gesture_hold_threshold > 1.0:
        # Scroll down (when fist held)
        pyautogui.scroll(-5)
    elif gesture == "Open Hand" and gesture_hold_threshold > 1.0:
        # Scroll up (when open hand held)
        pyautogui.scroll(5)

# Accessing camera
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create canvas for painting
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Operation modes
# 0=Detection, 1=Drawing, 2=Volume Control, 3=Computer Control, 4=Gesture Training
mode = 0
mode_names = ["Detection", "Drawing", "Volume Control", "Computer Control", "Gesture Training"]

# Virtual drawing variables
prev_drawing_point = None
drawing_color = GREEN
drawing_thickness = 10
drawing_mode = "pen"  # pen, brush, eraser

# Variables for FPS calculation
prev_time = 0
current_time = 0

# Initialize video recorder
video_recorder = VideoRecorder(width, height)

# Load custom gestures
custom_gestures = load_custom_gestures()
print(f"Loaded {len(custom_gestures)} custom gestures")

# Get screen dimensions for computer control
screen_width, screen_height = pyautogui.size()

# Main application loop
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture image from camera")
        break
    
    # Mirror image horizontally for more intuitive interaction
    image = cv2.flip(image, 1)
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 60
    prev_time = current_time
    
    # Convert image from BGR to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process hand detection
    result = hands.process(image_rgb)
    
    # Display canvas in drawing mode with blending
    if mode == 1:
        # Blend canvas with camera image for better visibility
        image = cv2.addWeighted(image, 0.7, canvas, 0.7, 0)
    
    # Initialize current gesture variables
    gesture_point = None
    gesture_name = "No Gesture"
    fingers_count = 0
    current_hand_type = "Unknown"
    
    # Recording status display
    recording_status = ""
    if video_recorder.recording:
        elapsed = time.time() - video_recorder.start_time if video_recorder.start_time else 0
        recording_status = f"REC {elapsed:.1f}s"
    elif recording_gesture:
        recording_status = f"Recording Gesture: {recording_countdown}s"
    
    # Check if hands are detected
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Determine hand type (Left/Right)
            current_hand_type = "Right"
            if result.multi_handedness:
                current_hand_type = result.multi_handedness[idx].classification[0].label
            
            # Draw hand landmarks with customized style based on mode
            if mode == 0 or mode == 4:  # Full visualization in detection modes
                mp_draw.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            else:  # Minimal visualization in other modes for less distraction
                mp_draw.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    {pair: (BLUE, 1) for pair in mp_hands.HAND_CONNECTIONS}
                )
            
            # Recognize hand gesture
            gesture, count, point = recognize_gesture(hand_landmarks, current_hand_type)
            gesture_name = gesture
            fingers_count = count
            
            # Apply position smoothing for more stable interaction
            if len(hand_position_history) > 0:
                gesture_point = smooth_position(hand_position_history, point)
            else:
                gesture_point = point
                hand_position_history.append(point)
            
            # Add gesture to history for temporal stability
            gesture_history.append(gesture)
            
            # Check for gesture hold time for triggering actions
            most_common_gesture = max(set(gesture_history), key=gesture_history.count)
            if most_common_gesture == gesture and gesture_history.count(gesture) > len(gesture_history) * 0.7:
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                
                hold_time = time.time() - gesture_start_time
                
                # Visual feedback for held gestures
                if hold_time > 0.5:
                    # Draw a progress circle around the index finger
                    progress = min(hold_time / gesture_hold_threshold, 1.0)
                    radius = 30
                    center = gesture_point
                    
                    # Background circle
                    cv2.circle(image, center, radius, (100, 100, 100), 2)
                    
                    # Progress arc
                    start_angle = -90
                    end_angle = start_angle + progress * 360
                    
                    # Draw the progress arc
                    axes = (radius, radius)
                    cv2.ellipse(image, center, axes, 0, start_angle, end_angle, GREEN if progress < 1.0 else BLUE, 3)
            else:
                gesture_start_time = None
            
            # Implement different modes functionality
            if mode == 1:  # Drawing mode
                if gesture == "Index Pointing":
                    # Draw on canvas
                    if prev_drawing_point is None:
                        prev_drawing_point = gesture_point
                        current_stroke = [gesture_point]
                    else:
                        prev_drawing_point = virtual_painter(canvas, gesture_point, prev_drawing_point, 
                                                         drawing_color, drawing_thickness, drawing_mode)
                        current_stroke.append(gesture_point)
                
                elif gesture == "Fist":
                    # Stop drawing and save current stroke
                    if current_stroke:
                        stroke_history.append({
                            'points': current_stroke.copy(),
                            'color': drawing_color,
                            'thickness': drawing_thickness,
                            'mode': drawing_mode
                        })
                        current_stroke = []
                    prev_drawing_point = None
                
                elif gesture == "Index Pinch" and gesture_start_time and (time.time() - gesture_start_time) > 1.0:
                    # Change color with color picker
                    hue = (gesture_point[0] % width) / width * 180  # Hue: 0-180
                    saturation = min(1.0, gesture_point[1] / (height / 2))  # Saturation: 0-1
                    value = 1.0  # Value: always 1 for brightness
                    
                    # Convert HSV to BGR
                    hsv_color = np.uint8([[[hue, saturation * 255, value * 255]]])
                    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                    drawing_color = (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))
                    
                    # Show color preview
                    cv2.circle(image, (width - 50, 80), 30, drawing_color, -1)
                    cv2.circle(image, (width - 50, 80), 30, WHITE, 2)
                
                elif gesture == "Peace Sign" and gesture_start_time and (time.time() - gesture_start_time) > 1.0:
                    # Cycle through drawing modes
                    modes = ["pen", "brush", "eraser"]
                    drawing_mode = modes[(modes.index(drawing_mode) + 1) % len(modes)]
                    
                    # Display mode change
                    mode_icons = {
                        "pen": "✏️", 
                        "brush": "🖌️", 
                        "eraser": "🧽"
                    }
                    cv2.putText(image, f"Mode: {drawing_mode} {mode_icons.get(drawing_mode, '')}", 
                                (width - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
                    
                    # Reset gesture hold timer
                    gesture_start_time = None
                
                elif gesture == "Three Fingers" and gesture_start_time and (time.time() - gesture_start_time) > 1.0:
                    # Change thickness
                    # Map y position to thickness (10-50)
                    drawing_thickness = max(5, min(50, int(50 - (gesture_point[1] / height * 40))))
                    
                    # Show thickness preview
                    cv2.circle(image, (width - 50, 150), drawing_thickness // 2, drawing_color, -1)
                    cv2.putText(image, f"Size: {drawing_thickness}", (width - 200, 160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
            
            elif mode == 2:  # Volume Control mode
                if gesture_point:
                    # Calculate volume percentage based on hand height
                    volume_level = 1.0 - (gesture_point[1] / height)  # 0.0 to 1.0
                    volume_width = int(width * 0.6)
                    volume_height = 40
                    
                    # Create a fancy volume visualization
                    # Background bar
                    cv2.rectangle(image, (int(width*0.2), height-70), (int(width*0.8), height-30), BLACK, -1)
                    cv2.rectangle(image, (int(width*0.2), height-70), (int(width*0.8), height-30), WHITE, 2)
                    
                    # Volume level
                    filled_width = int(width*0.2 + volume_width * volume_level)
                    
                    # Gradient color based on volume level
                    if volume_level < 0.3:
                        bar_color = GREEN
                    elif volume_level < 0.7:
                        bar_color = YELLOW
                    else:
                        bar_color = RED
                    
                    cv2.rectangle(image, (int(width*0.2), height-70), (filled_width, height-30), bar_color, -1)
                    
                    # Volume tick marks
                    for i in range(1, 10):
                        x_pos = int(width*0.2 + volume_width * i / 10)
                        cv2.line(image, (x_pos, height-70), (x_pos, height-30), WHITE, 1)
                    
                    # Volume percentage text
                    vol_text = f"Volume: {int(volume_level * 100)}%"
                    cv2.putText(image, vol_text, (int(width*0.35), height-45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
                    
                    # Additional visualization: audio wave simulation
                    wave_y_base = height - 100
                    wave_width = int(width * 0.6)
                    wave_height = int(50 * volume_level)
                    
                    for i in range(int(width*0.2), int(width*0.2) + wave_width, 10):
                        # Create a wave pattern that's affected by volume level
                        wave_val = math.sin((i + time.time() * 10) * 0.05) * wave_height
                        point1 = (i, int(wave_y_base + wave_val))
                        point2 = (i + 5, int(wave_y_base + math.sin((i + 5 + time.time() * 10) * 0.05) * wave_height))
                        cv2.line(image, point1, point2, bar_color, 2)
            
            elif mode == 3:  # Computer Control mode
                # Perform computer control actions
                if gesture in ["Index Pointing", "Index Pinch", "Peace Sign", "Fist", "Open Hand"]:
                    # Show control overlay
                    cv2.putText(image, "Computer Control Active", (width//2 - 150, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
                    
                    # Visualize control area
                    cv2.rectangle(image, (50, 50), (width-50, height-50), GREEN, 2)
                    
                    # Display control instructions
                    instructions = {
                        "Index Pointing": "Move Cursor",
                        "Index Pinch": "Left Click",
                        "Peace Sign": "Right Click",
                        "Fist": "Scroll Down",
                        "Open Hand": "Scroll Up"
                    }
                    
                    # Show active control
                    if gesture in instructions:
                        cv2.putText(image, f"Action: {instructions[gesture]}", (width//2 - 100, height - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
                    
                    # Perform the actual computer control action if gesture is held long enough
                    if gesture_start_time and (time.time() - gesture_start_time) > 0.3:
                        # Map the hand position to the screen coordinates
                        perform_computer_action(gesture, gesture_point, screen_width, screen_height)
            
            elif mode == 4:  # Gesture Training mode
                if recording_gesture:
                    # Display recording countdown
                    cv2.putText(image, f"Hold position: {recording_countdown}s", (width//2 - 150, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED if recording_countdown <= 2 else GREEN, 2)
                    
                    # Record the current finger pattern for custom gesture
                    if recording_countdown <= 0:
                        # Get the finger pattern from the current hand
                        finger_pattern = []
                        for tip_id in [4, 8, 12, 16, 20]:  # Thumb, Index, Middle, Ring, Pinky
                            if tip_id == 4:  # Thumb
                                if current_hand_type == "Right":
                                    if hand_landmarks.landmark[tip_id].x < hand_landmarks.landmark[tip_id-1].x:
                                        finger_pattern.append(1)
                                    else:
                                        finger_pattern.append(0)
                                else:  # Left hand
                                    if hand_landmarks.landmark[tip_id].x > hand_landmarks.landmark[tip_id-1].x:
                                        finger_pattern.append(1)
                                    else:
                                        finger_pattern.append(0)
                            else:  # Other fingers
                                if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id-2].y:
                                    finger_pattern.append(1)
                                else:
                                    finger_pattern.append(0)
                        
                        # Save the gesture with a default name (can be renamed later)
                        gesture_name = f"Custom_Gesture_{len(custom_gestures) + 1}"
                        save_custom_gesture(gesture_name, finger_pattern)
                        
                        # Reset recording state
                        recording_gesture = False
                        recording_countdown = 0
                        
                        # Confirmation message
                        cv2.putText(image, f"Gesture '{gesture_name}' saved!", (width//2 - 150, height//2 + 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
                else:
                    # Show available custom gestures
                    cv2.putText(image, f"Custom Gestures: {len(custom_gestures)}", (20, height - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
                    
                    # Display instructions for recording new gesture
                    cv2.putText(image, "Press 'R' to record a new gesture", (width//2 - 150, height//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
                    
                    # List existing custom gestures
                    y_pos = 200
                    for name in list(custom_gestures.keys())[:5]:  # Show up to 5 gestures
                        cv2.putText(image, f"- {name}", (width - 250, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
                        y_pos += 30
                    
                    if len(custom_gestures) > 5:
                        cv2.putText(image, f"+ {len(custom_gestures) - 5} more...", (width - 250, y_pos), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
    
    # If hands are not detected, reset tracking variables
    else:
        prev_drawing_point = None
        gesture_start_time = None
        hand_position_history.clear()
    
    # Add frames to video recording if active
    if video_recorder.recording:
        video_recorder.add_frame(image)
    
    # Process recording countdown if active
    if recording_gesture and recording_countdown > 0:
        recording_countdown -= 1/30  # Assuming 30 fps
    
    # Display information on screen
    display_info(image, gesture_name, fingers_count, fps, mode_names[mode], recording_status)
    
    # Display the most consistent recent gesture
    if len(gesture_history) > 0:
        recent_gesture = max(set(gesture_history), key=gesture_history.count)
        confidence = gesture_history.count(recent_gesture) / len(gesture_history)
        
        # Only show high confidence gestures
        if confidence > 0.5:
            cv2.putText(image, f'Detected: {recent_gesture} ({int(confidence*100)}%)', 
                        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    
    # Display the image
    cv2.imshow('Advanced Hand Gesture Recognition', image)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC key to quit
        break
    elif key == ord('m') or key == ord('M'):  # Change mode
        mode = (mode + 1) % len(mode_names)
        # Reset variables when changing mode
        canvas = np.zeros((height, width, 3), dtype=np.uint8) if mode == 1 else canvas
        gesture_history.clear()
        prev_drawing_point = None
    elif key == ord('c') or key == ord('C'):  # Clear canvas in drawing mode
        if mode == 1:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            stroke_history.clear()
    elif key == ord('u') or key == ord('U'):  # Undo last stroke in drawing mode
        if mode == 1 and stroke_history:
            # Redraw all strokes except the last one
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for stroke in stroke_history[:-1]:
                prev_point = None
                for point in stroke['points']:
                    if prev_point:
                        virtual_painter(canvas, point, prev_point, 
                                    stroke['color'], stroke['thickness'], stroke['mode'])
                    prev_point = point
            stroke_history.pop()  # Remove the last stroke
    elif key == ord('s') or key == ord('S'):  # Save canvas
        if mode == 1:
            saved_file = save_canvas(canvas)
            # Display confirmation
            save_overlay = image.copy()
            cv2.rectangle(save_overlay, (width//2 - 200, height//2 - 50), 
                        (width//2 + 200, height//2 + 50), BLACK, -1)
            cv2.addWeighted(save_overlay, 0.7, image, 0.3, 0, image)
            cv2.putText(image, f"Saved as: {os.path.basename(saved_file)}", 
                        (width//2 - 180, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            cv2.imshow('Advanced Hand Gesture Recognition', image)
            cv2.waitKey(1000)  # Show for 1 second
    elif key == ord('r') or key == ord('R'):  # Record new gesture
        if mode == 4 and not recording_gesture:
            recording_gesture = True
            recording_countdown = 3.0  # 3 seconds countdown
    elif key == ord('p') or key == ord('P'):  # Take screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_file = os.path.join(SAVE_DIR, f"screenshot_{timestamp}.png")
        cv2.imwrite(screenshot_file, image)
        
        # Show confirmation
        screenshot_overlay = image.copy()
        cv2.rectangle(screenshot_overlay, (width//2 - 200, height//2 - 50), 
                    (width//2 + 200, height//2 + 50), BLACK, -1)
        cv2.addWeighted(screenshot_overlay, 0.7, image, 0.3, 0, image)
        cv2.putText(image, f"Screenshot saved!", (width//2 - 150, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
        cv2.imshow('Advanced Hand Gesture Recognition', image)
        cv2.waitKey(1000)  # Show for 1 second
    elif key == ord('v') or key == ord('V'):  # Toggle video recording
        if video_recorder.recording:
            duration = video_recorder.stop()
            # Show confirmation
            video_overlay = image.copy()
            cv2.rectangle(video_overlay, (width//2 - 200, height//2 - 50), 
                        (width//2 + 200, height//2 + 50), BLACK, -1)
            cv2.addWeighted(video_overlay, 0.7, image, 0.3, 0, image)
            cv2.putText(image, f"Video saved! Duration: {duration:.1f}s", 
                        (width//2 - 180, height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
            cv2.imshow('Advanced Hand Gesture Recognition', image)
            cv2.waitKey(1000)  # Show for 1 second
        else:
            filename = video_recorder.start()
            print(f"Recording started: {filename}")

# Clean up
video_recorder.stop()
cap.release()
cv2.destroyAllWindows()

# If run directly, display a welcome message
if __name__ == "__main__":
    print("Advanced Hand Gesture Recognition System")
    print("=========================================")
    print(f"Loaded {len(custom_gestures)} custom gestures")
    print(f"Using MediaPipe Hands model with complexity level 1")
    print(f"Press 'M' to cycle through {len(mode_names)} operation modes")
    print("Press 'ESC' to quit")