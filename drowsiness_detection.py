


# from scipy.spatial import distance
# from imutils import face_utils
# from pygame import mixer
# import imutils
# import dlib
# import cv2
# import time
# import torch
# from ultralytics import YOLO

# # Initialize mixer for sound alerts
# mixer.init()

# # Load alert sounds
# drowsiness_sound = "music.wav"
# phone_sound = "phone_alert.wav"
# bottle_sound = "bottle_alert.wav"

# # Load YOLOv8 model for object detection
# model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model

# # Function to calculate Eye Aspect Ratio (EAR)
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # Function to calculate Mouth Aspect Ratio (MAR)
# def mouth_aspect_ratio(mouth):
#     A = distance.euclidean(mouth[2], mouth[10])
#     B = distance.euclidean(mouth[4], mouth[8])
#     C = distance.euclidean(mouth[0], mouth[6])
#     return (A + B) / (2.0 * C)

# # Set thresholds
# eye_thresh = 0.25
# frame_check = 20
# mar_thresh = 0.6
# yawn_limit = 5

# # Initialize Dlib face detector and landmark predictor
# detect = dlib.get_frontal_face_detector()
# predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
# (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# cap = cv2.VideoCapture(0)
# cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Frame", 1024, 768)

# flag = 0
# yawn_count = 0
# start_time = time.time()
# yawn_alert_time = None
# show_yawn_alert = False

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = imutils.resize(frame, width=800)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     subjects = detect(gray, 0)

#     # Object detection for distractions
#     results = model(frame)
#     alert_played = False  # Ensure only one sound plays at a time

#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             label = result.names[int(box.cls[0])]

#             if label == "cell phone" and not alert_played:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, "PHONE DETECTED", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 mixer.music.load(phone_sound)
#                 mixer.music.play()
#                 alert_played = True  # Prevent multiple sounds playing

#             elif label in ["bottle", "cup"] and not alert_played:
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
#                 cv2.putText(frame, "DRINKING DETECTED", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#                 mixer.music.load(bottle_sound)
#                 mixer.music.play()
#                 alert_played = True

#     for subject in subjects:
#         shape = predict(gray, subject)
#         shape = face_utils.shape_to_np(shape)

#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         mouth = shape[mStart:mEnd]
#         mar = mouth_aspect_ratio(mouth)

#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         mouthHull = cv2.convexHull(mouth)

#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 2)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 2)
#         cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 2)

#         if ear < eye_thresh:
#             flag += 1
#             if flag >= frame_check and not alert_played:
#                 cv2.putText(frame, "****************ARE YOU SLEEPY?!****************", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
#                 mixer.music.load(drowsiness_sound)
#                 mixer.music.play()
#                 alert_played = True
#         else:
#             flag = 0

#         if mar > mar_thresh:
#             yawn_count += 1

#         if yawn_count >= yawn_limit and not show_yawn_alert:
#             yawn_alert_time = time.time()
#             show_yawn_alert = True
#             mixer.music.load(drowsiness_sound)
#             mixer.music.play()
#             alert_played = True

#         if show_yawn_alert:
#             cv2.putText(frame, "********** ARE YOU SLEEPY? TAKE REST! **********", (50, 700),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
#             if time.time() - yawn_alert_time >= 3:
#                 show_yawn_alert = False
#                 mixer.music.stop()
#                 yawn_count = 0

#         if time.time() - start_time >= 60:
#             yawn_count = 0
#             start_time = time.time()

#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()
# cap.release()
import cv2
import dlib
import time
import imutils
import requests
import pyttsx3
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

# Load YOLOv8 for object detection (distraction detection)
print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")

# Load face detector and facial landmarks predictor
print("[INFO] Loading face detector and shape predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indexes for eyes & mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# EAR & MAR thresholds
EYE_THRESH = 0.25
MAR_THRESH = 0.6
YAWN_LIMIT = 5
FRAME_CHECK = 10

# Tracking variables
flag = 0
yawn_count = 0
start_time = time.time()
alert_time = None
show_alert = False
current_alert = None  # Track currently playing alert

# Hotels list
nearby_hotels = []

# Google Places API Key (Invalid here for safety, replace with real key)
API_KEY = "AIzaSyDKndPBkvqiyi1_vbfpe5nR-341ik_abeM"

# Function to calculate EAR
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate MAR
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# Function to get nearby hotels
def get_nearby_hotels(lat, lng):
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={lat},{lng}&radius=5000&type=lodging&key={API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        hotels = [place["name"] for place in data.get("results", [])]
        return hotels if hotels else ["No hotels found nearby"]
    except Exception as e:
        print(f"[ERROR] Hotel fetch failed: {e}")
        return ["Error fetching hotels"]

# Function to get current location
def get_current_location():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()
        return data['lat'], data['lon']
    except Exception as e:
        print(f"[ERROR] Location fetch failed: {e}")
        return None, None

# Function to speak hotels aloud
def speak_hotels(hotels):
    engine.say("Nearby Hotels are")
    for hotel in hotels[:5]:
        engine.say(hotel)
    engine.runAndWait()

# Function to play alert sound with delay to ensure complete playback
def play_alert_sound(sound_file):
    global current_alert
    if current_alert:
        return  # If an alert is already playing, skip this one
    current_alert = sound_file
    mixer.music.load(sound_file)
    mixer.music.play()

    # Wait until sound finishes playing
    while mixer.music.get_busy():
        time.sleep(0.1)

    current_alert = None

# Start video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Driver Monitoring System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Driver Monitoring System", 1024, 768)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame")
        break

    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    # Object detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])] 

            if label == "cell phone":
                play_alert_sound(PHONE_SOUND)

            elif label in ["bottle", "cup"]:
                play_alert_sound(BOTTLE_SOUND)

    # Drowsiness detection
    drowsy_triggered = False
    for subject in subjects:
        shape = predictor(gray, subject)
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

        # Fetch hotels if drowsy or yawning
        if (drowsy_triggered or yawn_count >= YAWN_LIMIT) and not show_alert:
            play_alert_sound(DROWSINESS_SOUND)
            alert_time = time.time()
            show_alert = True

            lat, lng = get_current_location()
            if lat and lng:
                nearby_hotels = get_nearby_hotels(lat, lng)
                speak_hotels(nearby_hotels)

        # Display alert on screen
        if show_alert:
            cv2.putText(frame, "ARE YOU SLEEPY! TAKE REST", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            y = 100
            for hotel in nearby_hotels[:5]:
                cv2.putText(frame, hotel, (50, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 30

            if time.time() - alert_time >= 5:
                show_alert = False
                yawn_count = 0
                flag = 0
                nearby_hotels = []

        if time.time() - start_time > 60:
            yawn_count = 0
            start_time = time.time()

    cv2.imshow("Driver Monitoring System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
