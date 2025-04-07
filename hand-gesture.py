import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque
import time

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Definisi warna dalam format BGR
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)

# Tracking untuk gesture history
gesture_history = deque(maxlen=10)
drawing_points = []  # Untuk fitur drawing

# Fungsi untuk menghitung jarak antara dua landmark
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

# Fungsi untuk mengenali gesture tangan
def recognize_gesture(hand_landmarks, hand_type="Right"):
    finger_tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_bases = [2, 5, 9, 13, 17]      # Base of fingers
    fingers = []
    
    # Mendeteksi apakah tangan kanan atau kiri
    is_right_hand = hand_type == "Right"
    
    # Thumb: berbeda cara deteksinya tergantung tangan kanan/kiri
    if is_right_hand:
        if hand_landmarks.landmark[finger_tips_ids[0]].x < hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Tangan kiri
        if hand_landmarks.landmark[finger_tips_ids[0]].x > hand_landmarks.landmark[finger_tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # 4 jari lainnya: Up (1) jika ujung jari lebih tinggi dari sendi tengah
    for id in range(1, 5):
        if hand_landmarks.landmark[finger_tips_ids[id]].y < hand_landmarks.landmark[finger_tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    
    # Mengenali gesture khusus
    total_fingers = fingers.count(1)
    
    # Gesture point untuk index finger
    index_finger_tip = hand_landmarks.landmark[8]
    middle_finger_tip = hand_landmarks.landmark[12]
    
    # Mengenali gesture berdasarkan kombinasi jari
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumb Up", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [0, 1, 0, 0, 0]:
        return "Index Pointing", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [0, 1, 1, 1, 0]:
        return "Three Fingers", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [0, 1, 1, 1, 1]:
        return "Four Fingers", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Hand", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [1, 1, 0, 0, 1]:
        return "Rock Sign", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    elif fingers == [0, 1, 0, 0, 1]:
        return "Spider-Man", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    
    # Deteksi Pinch gesture (ibu jari dan telunjuk berdekatan)
    pinch_threshold = 0.05
    if calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8]) < pinch_threshold:
        return "Pinch", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
    
    # OK gesture (ibu jari dan telunjuk membentuk lingkaran)
    if calculate_distance(hand_landmarks.landmark[4], hand_landmarks.landmark[8]) < 0.1 and fingers[1] == 1:
        return "OK Sign", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))
        
    return f"{total_fingers} Fingers", total_fingers, (int(index_finger_tip.x * width), int(index_finger_tip.y * height))

# Fungsi untuk menampilkan informasi di layar
def display_info(image, gesture, fingers_count, fps, mode):
    # Background informasi
    cv2.rectangle(image, (10, 10), (300, 140), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 10), (300, 140), WHITE, 2)
    
    # FPS
    cv2.putText(image, f'FPS: {fps:.1f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
    
    # Jumlah jari
    cv2.putText(image, f'Fingers: {fingers_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)
    
    # Gesture
    cv2.putText(image, f'Gesture: {gesture}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, PURPLE, 2)
    
    # Mode
    cv2.putText(image, f'Mode: {mode}', (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2)
    
    # Instruksi
    cv2.putText(image, "Press 'M': Change Mode", (width - 250, 30), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
    cv2.putText(image, "Press 'C': Clear Drawing", (width - 250, 50), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
    cv2.putText(image, "Press 'ESC': Quit", (width - 250, 70), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)

# Fungsi untuk virtual painting/drawing
def virtual_painter(image, finger_point, prev_point, color):
    if prev_point is not None:
        cv2.line(image, finger_point, prev_point, color, 10)
    return finger_point

# Mengakses kamera
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Buat kanvas untuk painting
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Mode operasi: 0=Detection, 1=Drawing, 2=Volume Control
mode = 0
mode_names = ["Detection", "Drawing", "Volume Control"]

# Variabel untuk virtual drawing
prev_drawing_point = None
drawing_color = GREEN

# Variabel untuk FPS
prev_time = 0
current_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break
    
    # Mirror image horizontally
    image = cv2.flip(image, 1)
    
    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 60
    prev_time = current_time
    
    # Konversi warna gambar dari BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Proses deteksi tangan
    result = hands.process(image_rgb)
    
    # Tampilkan kanvas jika mode drawing
    if mode == 1:
        image = cv2.addWeighted(image, 0.6, canvas, 0.4, 0)
    
    # Initialize current gesture point
    gesture_point = None
    gesture_name = "No Gesture"
    fingers_count = 0
    
    # Jika tangan terdeteksi
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Tentukan tipe tangan (Left/Right)
            hand_type = "Right"
            if result.multi_handedness:
                hand_type = result.multi_handedness[idx].classification[0].label
            
            # Gambar landmark tangan dengan style yang lebih menarik
            mp_draw.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Mengenali gesture tangan
            gesture, count, point = recognize_gesture(hand_landmarks, hand_type)
            gesture_name = gesture
            fingers_count = count
            gesture_point = point
            
            # Tambahkan gesture ke history
            gesture_history.append(gesture)
            
            # Implementasi berbagai mode
            if mode == 1 and gesture == "Index Pointing":  # Mode drawing
                if prev_drawing_point is None:
                    prev_drawing_point = gesture_point
                else:
                    # Gambar pada canvas
                    cv2.line(canvas, prev_drawing_point, gesture_point, drawing_color, 10)
                    prev_drawing_point = gesture_point
            elif mode == 1 and gesture == "Pinch":  # Ganti warna
                drawing_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                prev_drawing_point = None
            elif mode == 1 and gesture == "Fist":  # Reset previous point
                prev_drawing_point = None
            elif mode == 2 and gesture_point:  # Mode volume control (simulasi)
                # Visualize volume level based on hand height
                volume_level = 1.0 - (gesture_point[1] / height)  # 0.0 to 1.0
                volume_width = int(width * 0.6)
                volume_height = 30
                
                # Draw volume bar
                cv2.rectangle(image, (int(width*0.2), height-50), (int(width*0.8), height-20), WHITE, 2)
                cv2.rectangle(image, (int(width*0.2), height-50), (int(width*0.2 + volume_width * volume_level), height-20), GREEN, -1)
                
                # Display volume percentage
                vol_text = f"Volume: {int(volume_level * 100)}%"
                cv2.putText(image, vol_text, (int(width*0.2), height-60), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)
    
    # No hand detected - reset drawing point
    else:
        prev_drawing_point = None
    
    # Display information on screen
    display_info(image, gesture_name, fingers_count, fps, mode_names[mode])
    
    # Display gesture history
    if len(gesture_history) > 0:
        recent_gesture = max(set(gesture_history), key=gesture_history.count)
        cv2.putText(image, f'Recent: {recent_gesture}', (width - 250, 100), cv2.FONT_HERSHEY_PLAIN, 1, WHITE, 1)
    
    # Tampilkan gambar
    cv2.imshow('Hand Gesture Recognition', image)
    
    # Kontrol keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('m') or key == ord('M'):  # Change mode
        mode = (mode + 1) % 3
        canvas = np.zeros((height, width, 3), dtype=np.uint8)  # Clear canvas when changing mode
    elif key == ord('c') or key == ord('C'):  # Clear canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()