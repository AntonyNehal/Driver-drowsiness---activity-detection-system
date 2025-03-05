# import cv2
# import dlib
# import time
# import imutils
# import requests
# import pyttsx3
# import threading
# from scipy.spatial import distance
# from imutils import face_utils
# from pygame import mixer
# from ultralytics import YOLO

# # Initialize mixer for sound alerts
# mixer.init()

# # Initialize text-to-speech engine
# engine = pyttsx3.init()

# # Alert Sounds
# DROWSINESS_SOUND = "music.wav"
# PHONE_SOUND = "phone_alert.wav"
# BOTTLE_SOUND = "bottle_alert.wav"
# EATING_SOUND = "eating.wav"   # New sound for eating detection

# # Load YOLOv8 for object detection (distraction + eating detection)
# model = YOLO("yolov8n.pt")

# # Load face detector and facial landmarks predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Landmark indexes for eyes & mouth
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# # EAR & MAR thresholds
# EYE_THRESH = 0.25
# MAR_THRESH = 0.75
# YAWN_LIMIT = 7
# FRAME_CHECK = 15

# # Tracking variables
# flag = 0
# yawn_count = 0
# show_alert = False
# alert_time = None
# nearby_hotels = []

# # Google Places API Key (replace with your actual key)
# API_KEY = "AIzaSyDKndPBkvqiyi1_vbfpe5nR-341ik_abeM"

# # EAR calculation
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # MAR calculation
# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[2], mouth[10])
#     B = distance.euclidean(mouth[4], mouth[8])
#     C = distance.euclidean(mouth[0], mouth[6])
#     return (A + B) / (2.0 * C)

# # Play alert sound
# def play_alert_sound(sound_file):
#     def play():
#         mixer.music.load(sound_file)
#         mixer.music.play()
#         while mixer.music.get_busy():
#             time.sleep(0.1)
#     threading.Thread(target=play, daemon=True).start()

# # Speak nearby hotels
# def speak_nearby_hotels():
#     def fetch_and_speak():
#         global nearby_hotels
#         lat, lng = get_current_location()
#         if lat and lng:
#             nearby_hotels = get_nearby_hotels(lat, lng)
#         else:
#             nearby_hotels = ["Location fetch failed"]

#         engine.say("Nearby Hotels are")
#         for hotel in nearby_hotels[:5]:
#             engine.say(hotel)
#         engine.runAndWait()

#     threading.Thread(target=fetch_and_speak, daemon=True).start()

# # Get current location
# def get_current_location():
#     try:
#         response = requests.get("http://ip-api.com/json/")
#         data = response.json()
#         return data['lat'], data['lon']
#     except:
#         return None, None

# # Get nearby hotels
# def get_nearby_hotels(lat, lng):
#     url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
#            f"?location={lat},{lng}&radius=5000&type=lodging&key={API_KEY}")
#     try:
#         response = requests.get(url)
#         data = response.json()
#         return [hotel["name"] for hotel in data.get("results", [])] or ["No hotels found nearby"]
#     except:
#         return ["Error fetching hotels"]

# # Video capture
# cap = cv2.VideoCapture("videoplay.mp4")
# cv2.namedWindow("Driver Monitoring System", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Driver Monitoring System", 1024, 768)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = imutils.resize(frame, width=800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Face detection
#     faces = detector(gray)

#     # Object detection (for distractions + eating)
#     results = model(frame)

#     # Object Detection Alerts (Phone, Bottle, Fork)
#     for result in results:
#         for box in result.boxes:
#             label = result.names[int(box.cls[0])]
#             confidence = float(box.conf[0])

#             if label == "cell phone" and confidence > 0.6:
#                 play_alert_sound(PHONE_SOUND)

#             elif label in ["bottle", "cup"] and confidence > 0.6:
#                 play_alert_sound(BOTTLE_SOUND)

#             elif label == "fork" and confidence > 0.5:  # Fork detection for eating
#                 play_alert_sound(EATING_SOUND)

#     # Drowsiness and Yawning Detection
#     drowsy_triggered = False
#     for face in faces:
#         shape = predictor(gray, face)
#         shape = face_utils.shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         mouth = shape[mStart:mEnd]

#         ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
#         mar = mouth_aspect_ratio(mouth)

#         if ear < EYE_THRESH:
#             flag += 1
#             if flag >= FRAME_CHECK:
#                 play_alert_sound(DROWSINESS_SOUND)
#                 drowsy_triggered = True
#         else:
#             flag = 0

#         if mar > MAR_THRESH:
#             yawn_count += 1

#         if (drowsy_triggered or yawn_count >= YAWN_LIMIT) and not show_alert:
#             show_alert = True
#             alert_time = time.time()
#             play_alert_sound(DROWSINESS_SOUND)
#             speak_nearby_hotels()

#     # Display Alert & Hotel Info
#     if show_alert:
#         cv2.putText(frame, "ARE YOU SLEEPY! TAKE REST", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         for idx, hotel in enumerate(nearby_hotels[:5], start=1):
#             cv2.putText(frame, hotel, (50, 50 + idx * 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         if time.time() - alert_time >= 10:
#             show_alert = False
#             yawn_count = 0
#             flag = 0
#             nearby_hotels = []

#     # Show Frame
#     cv2.imshow("Driver Monitoring System", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import dlib
import time
import imutils
import requests
import pyttsx3
import threading
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
from ultralytics import YOLO

# Initialize mixer for sound alerts
mixer.init()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Alert Sounds
DROWSINESS_SOUND = "music.wav"
PHONE_SOUND = "phone_alert.wav"
BOTTLE_SOUND = "bottle_alert.wav"
EATING_SOUND = "eating.wav"

# Load YOLOv8 for object detection
model = YOLO("yolov8n.pt")

# Load face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indexes for eyes & mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Thresholds
EYE_THRESH = 0.25
MAR_THRESH = 0.80   # Increased to detect larger yawns
YAWN_LIMIT = 7
FRAME_CHECK = 15

# Tracking variables
flag = 0
yawn_count = 0
show_alert = False
alert_time = None
nearby_hotels = []

# Cooldown timers (to prevent repeat alerts)
last_phone_alert = 0
last_bottle_alert = 0
last_eating_alert = 0
ALERT_COOLDOWN = 10  # seconds

# Google Places API Key
API_KEY = "AIzaSyDKndPBkvqiyi1_vbfpe5nR-341ik_abeM"

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# MAR calculation
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Generic sound player with thread (non-blocking)
def play_alert_sound(sound_file):
    def play():
        mixer.music.load(sound_file)
        mixer.music.play()
        while mixer.music.get_busy():
            time.sleep(0.1)
    threading.Thread(target=play, daemon=True).start()

# Custom alert functions with cooldown
def handle_phone_alert():
    global last_phone_alert
    if time.time() - last_phone_alert > ALERT_COOLDOWN:
        last_phone_alert = time.time()
        play_alert_sound(PHONE_SOUND)

def handle_bottle_alert():
    global last_bottle_alert
    if time.time() - last_bottle_alert > ALERT_COOLDOWN:
        last_bottle_alert = time.time()
        play_alert_sound(BOTTLE_SOUND)

def handle_eating_alert():
    global last_eating_alert
    if time.time() - last_eating_alert > ALERT_COOLDOWN:
        last_eating_alert = time.time()
        play_alert_sound(EATING_SOUND)

# Speak nearby hotels
def speak_nearby_hotels():
    def fetch_and_speak():
        global nearby_hotels
        lat, lng = get_current_location()
        if lat and lng:
            nearby_hotels = get_nearby_hotels(lat, lng)
        else:
            nearby_hotels = ["Location fetch failed"]

        engine.say("Nearby Hotels are")
        for hotel in nearby_hotels[:5]:
            engine.say(hotel)
        engine.runAndWait()

    threading.Thread(target=fetch_and_speak, daemon=True).start()

# Get current location
def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data['lat'], data['lon']
    except:
        return None, None

# Get nearby hotels
def get_nearby_hotels(lat, lng):
    url = (f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
           f"?location={lat},{lng}&radius=5000&type=lodging&key={API_KEY}")
    try:
        response = requests.get(url)
        data = response.json()
        return [hotel["name"] for hotel in data.get("results", [])] or ["No hotels found nearby"]
    except:
        return ["Error fetching hotels"]

# Video capture
cap = cv2.VideoCapture("videoplay.mp4")
cv2.namedWindow("Driver Monitoring System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Driver Monitoring System", 1024, 768)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = detector(gray)

    # Object detection
    results = model(frame)

    # Object detection alerts
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if label == "cell phone" and confidence > 0.6:
                handle_phone_alert()

            elif label in ["bottle", "cup"] and confidence > 0.6:
                handle_bottle_alert()

            elif label == "fork" and confidence > 0.5:
                handle_eating_alert()

    # Drowsiness & yawning detection
    drowsy_triggered = False
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        if ear < EYE_THRESH:
            flag += 1
            if flag >= FRAME_CHECK:
                play_alert_sound(DROWSINESS_SOUND)
                drowsy_triggered = True
        else:
            flag = 0

        if mar > MAR_THRESH:
            yawn_count += 1

        if (drowsy_triggered or yawn_count >= YAWN_LIMIT) and not show_alert:
            show_alert = True
            alert_time = time.time()
            play_alert_sound(DROWSINESS_SOUND)
            speak_nearby_hotels()

    # Display alerts & hotel info
    if show_alert:
        cv2.putText(frame, "ARE YOU SLEEPY! TAKE REST", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for idx, hotel in enumerate(nearby_hotels[:5], start=1):
            cv2.putText(frame, hotel, (50, 50 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if time.time() - alert_time >= 10:
            show_alert = False
            yawn_count = 0
            flag = 0
            nearby_hotels = []

    # Show frame
    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
