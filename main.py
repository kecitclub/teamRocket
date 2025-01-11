import cv2 as cv
import os
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Form, Request, WebSocket, HTTPException, status, Depends
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates  
import uvicorn
import pyaudio
import threading
import time
import io
import base64  
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from pydantic import BaseModel
from datetime import datetime
from ultralytics import YOLO 
from PIL import Image
from sqlalchemy.orm import Session
from typing import Annotated
import models
from models import User  # Import the User class, not the table name
from database import engine, SessionLocal

import google.generativeai as genai
import os
import pdfplumber
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

# Create tables in the database
models.Base.metadata.create_all(bind=engine)
class UserBase(BaseModel):
    username: str
    password: str

# Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# Set up templates directory
templates = Jinja2Templates(directory="templates")
# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables
current_user = None
eye_cheating = False
head_cheating = False
video_feed_active = False
video_cap = None
sound_detected = False
audio_detection_active = False
stream = None  # Initialize stream variable
total_time = 100
start_time = None
cheating_data = []
RECORDING_DIR = "cheating_recordings"
is_recording = False
out = None

# Add these new global variables
MINIMUM_CHEATING_DURATION = 3  # Minimum duration in seconds to save a clip
current_cheating_start = None
current_cheating_flags = set()  # Track which behaviors triggered the cheating detection

# Create the recordings directory if it doesn't exist
if not os.path.exists(RECORDING_DIR):
    os.makedirs(RECORDING_DIR)

# Face tracking setup
mp_face_mesh = mp.solutions.face_mesh
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# Yolo Model Setup
yolo_model = YOLO('yolov8m.pt')
class_names = ['person', 'book', 'cell phone']
class_indices = [i for i, name in yolo_model.names.items() if name in class_names]
multiple_persons_detected = False
book_detected = False
phone_detected = False

# Extracting data from database
def initialize_user_cheating_data():
    db: Session = SessionLocal()
    try:
        users = db.query(models.User.username).filter(models.User.username != "admin").distinct().all()
        user_cheating_data = {user.username: [] for user in users}
    finally:
        db.close()
    return user_cheating_data

# Initialize user_cheating_data
user_cheating_data = initialize_user_cheating_data()

def detect_objects(frame):
    global multiple_persons_detected, book_detected, phone_detected
    
    results = yolo_model(frame, classes=class_indices)
    
    person_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            class_name = result.names[cls]
            if class_name == 'person':
                person_count += 1
            elif class_name == 'book':
                book_detected = True
            elif class_name == 'cell phone':
                phone_detected = True
    
    multiple_persons_detected = person_count > 1

def eucidiean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def iris_position(iris_center, right_point, left_point):
    center_to_right = eucidiean_distance(iris_center, right_point)
    total_distance = eucidiean_distance(right_point, left_point)
    gaze_ratio = center_to_right / total_distance

    if gaze_ratio < 0.4:
        return "RIGHT", gaze_ratio
    elif 0.4 <= gaze_ratio <= 0.55:
        return "CENTER", gaze_ratio
    else:
        return "LEFT", gaze_ratio
    
def detect_sound():
    global sound_detected, stream
    # PyAudio setup for sound detection
    FORMAT = pyaudio.paInt16  # Audio format (16-bit resolution)
    CHANNELS = 1              # Mono channel
    RATE = 44100              # Sampling rate (44.1kHz)
    CHUNK = 1024              # Number of samples per chunk
    THRESHOLD = 500           # Adjust this value based on your environment

    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for sound...")

    try:
        while audio_detection_active:
            # Read audio data from the stream
            data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # Compute the average volume (amplitude)
            volume = np.linalg.norm(data)

            # Check if the volume exceeds the threshold
            if volume > THRESHOLD:
                sound_detected = True
            else:
                sound_detected = False

            time.sleep(0.1)  # Adjust frequency of sound detection

    except KeyboardInterrupt:
        print("Audio detection stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate PyAudio object
    p.terminate()

def start_audio_detection():
    global audio_detection_active
    audio_detection_active = True
    audio_thread = threading.Thread(target=detect_sound)
    audio_thread.daemon = True  # Ensure it closes with the main program
    audio_thread.start()

def stop_audio_detection():
    global audio_detection_active, stream
    audio_detection_active = False
    print("Audio detection stopped.")
    if stream is not None:
        stream.stop_stream()
        stream.close()

anti_spoofing_model = YOLO('best.pt')

def predict_anti_spoofing(image):
    results = anti_spoofing_model(image, verbose=False)
    for result in results:
        probs = result.probs
        if probs.top1 > 0.7:
            return f"{anti_spoofing_model.names[probs.top1]}", probs.top1conf.item()
        elif probs.top1 == 0:
            return f"{anti_spoofing_model.names[probs.top1]}", probs.top1conf.item()
    return "Unknown", 0.0

def start_recording(frame):
    global out, is_recording, current_cheating_start, current_cheating_flags
    if not is_recording:
        current_cheating_start = time.time()
        current_cheating_flags = set()  # Initialize empty set for new recording
        frame_size = (frame.shape[1], frame.shape[0])
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        # Don't create the file yet - we'll do that only if the cheating duration meets our threshold
        out = {
            'frames': [],
            'frame_size': frame_size,
            'fourcc': fourcc
        }
        is_recording = True
        
def update_recording(frame, detected_flags):
    global out, current_cheating_flags
    if is_recording and out is not None:
        out['frames'].append(frame.copy())
        current_cheating_flags.update(detected_flags)  # Add any new flags detected

def stop_recording():
    global out, is_recording, current_cheating_start, current_cheating_flags
    if is_recording and out is not None:
        cheating_duration = time.time() - current_cheating_start
        
        # Only save if cheating duration meets minimum threshold
        if cheating_duration >= MINIMUM_CHEATING_DURATION:
            # Create filename with timestamp and detected behaviors
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            duration_str = f"{int(cheating_duration)}s"
            flags_str = "_".join(sorted(current_cheating_flags))
            filename = os.path.join(
                RECORDING_DIR, 
                f"cheating_{timestamp}_duration{duration_str}_{flags_str}.mp4"
            )
            
            # Create the actual video file
            video_writer = cv.VideoWriter(
                filename,
                out['fourcc'],
                20.0,  # fps
                out['frame_size']
            )
            
            # Write all stored frames
            for frame in out['frames']:
                video_writer.write(frame)
            
            video_writer.release()
            print(f"Saved cheating clip: {filename}")
        
        # Reset recording state
        out = None
        current_cheating_start = None
        current_cheating_flags = set()
    
    is_recording = False
        
def generate_video_feed(username):
    global eye_cheating, head_cheating, video_feed_active, multiple_persons_detected, book_detected, phone_detected, start_time, video_cap, is_recording, out
    video_cap = cv.VideoCapture(0)

    # New variables for sampling and aggregation
    cheating_buffer = []
    buffer_duration = 30  # 1 minute buffer

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while video_feed_active:
            ret, frame = video_cap.read()
            if not ret:
                break
            frame = cv.flip(frame, 1)
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]

            detect_objects(frame)
            
            current_time = time.time()

            pil_image = Image.fromarray(frame_rgb)
            anti_spoofing_label, confidence = predict_anti_spoofing(pil_image)
            
            # Determine color based on the prediction
            color = (0, 255, 0) if anti_spoofing_label == "real" else (0, 0, 255)
            
            # Display anti-spoofing result on the frame
            cv.putText(frame, f"Anti-Spoofing: {anti_spoofing_label} ({confidence:.2f})", 
                       (20, 280), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if sound_detected:
                cv.circle(frame, (600, 50), 20, (0, 0, 255), -1)  # Red dot

            results = face_mesh.process(frame_rgb)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                iris_center_left = np.array([l_cx, l_cy], dtype=np.int32)
                iris_center_right = np.array([r_cx, r_cy], dtype=np.int32)

                # cv.circle(frame, iris_center_left, int(l_radius), (0, 255, 0), 1)
                # cv.circle(frame, iris_center_right, int(r_radius), (0, 255, 0), 1)
                # cv.circle(frame, iris_center_left, int(l_radius), (0, 255, 0), 1)
                # cv.circle(frame, iris_center_right, int(r_radius), (0, 255, 0), 1)
                # cv.circle(frame, mesh_points[L_H_LEFT][0], 1, (0, 255, 0), 1)
                # cv.circle(frame, mesh_points[L_H_RIGHT][0], 1, (0, 255, 0), 1)
                # cv.circle(frame, mesh_points[R_H_LEFT][0], 1, (0, 255, 0), 1)
                # cv.circle(frame, mesh_points[R_H_RIGHT][0], 1, (0, 255, 0), 1)

                iris_pos, gaze_ratio = iris_position(iris_center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
                eye_cheating = iris_pos != "CENTER"
                # cv.putText(frame, iris_pos, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
                
                face_2d = []
                face_3d = []
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            x, y = int(lm.x * img_w), int(lm.y * img_h)
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                ret, rotation_vec, translation_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)
                rmat, jac = cv.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                head_cheating = abs(y) > 10 or abs(x) > 10

                # if y < -10:
                #     text = "Looking Left"
                # elif y > 10:
                #     text = "Looking Right"
                # elif x < -7:
                #     text = "Looking Down"
                # elif x > 13:
                #     text = "Looking Up"
                # else:
                #     text = "Forward"

                # nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix, distortion_matrix)
                # p1 = (int(nose_2d[0]), int(nose_2d[1]))
                # p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                # cv.line(frame, p1, p2, (255, 0, 0), 3)
                # cv.putText(frame, text, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

# Create set of detected cheating behaviors
                detected_flags = set()
                if eye_cheating:
                    detected_flags.add("eye_movement")
                if head_cheating:
                    detected_flags.add("head_movement")
                if sound_detected:
                    detected_flags.add("sound")
                if multiple_persons_detected:
                    detected_flags.add("multiple_persons")
                if book_detected:
                    detected_flags.add("book")
                if phone_detected:
                    detected_flags.add("phone")
                if anti_spoofing_label != "real":
                    detected_flags.add("spoofing")
                
                cheating_votes = len(detected_flags)
                is_cheating = cheating_votes >= 2
                
                if is_cheating:
                    if not is_recording:
                        start_recording(frame)
                    update_recording(frame, detected_flags)
                elif is_recording:
                    stop_recording()

                cheating_buffer.append((current_time, is_cheating))

                    # Remove old entries from the buffer
                cheating_buffer = [entry for entry in cheating_buffer if current_time - entry[0] <= buffer_duration]

                    # Calculate the majority cheating status for the last minute
                cheating_count = sum(1 for _, cheating in cheating_buffer if cheating)
                majority_cheating = cheating_count > len(cheating_buffer) / 2

                elapsed_time = current_time - start_time
                user_cheating_data[username].append((elapsed_time, majority_cheating))

                # elapsed_time = time.time() - start_time
                # user_cheating_data[username].append((elapsed_time, is_cheating))

                color = (0, 0, 255) if is_cheating else (0, 255, 0)
                text = "CHEATING DETECTED" if is_cheating else "NO CHEATING DETECTED"
                cv.putText(frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display individual module results
                cv.putText(frame, f"Eye: {'Cheating' if eye_cheating else 'OK'}", (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Head: {'Cheating' if head_cheating else 'OK'}", (20, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Sound: {'Detected' if sound_detected else 'Not Detected'}", (20, 160), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Multiple Persons: {'Detected' if multiple_persons_detected else 'Not Detected'}", (20, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Book: {'Detected' if book_detected else 'Not Detected'}", (20, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv.putText(frame, f"Phone: {'Detected' if phone_detected else 'Not Detected'}", (20, 250), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                time_remaining = max(0, total_time - elapsed_time)
                # cv.putText(frame, f"Time remaining: {int(time_remaining)}s", (20, 310), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            _, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            multiple_persons_detected = False
            book_detected = False
            phone_detected = False

    if is_recording:
        stop_recording()
    video_cap.release()

def generate_cheating_graph(username):
    if not user_cheating_data[username]:
        return None
    
    times, cheating = zip(*user_cheating_data[username])
    plt.figure(figsize=(10, 5))
    plt.plot(times, cheating, 'r-')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cheating Detected')
    plt.title(f'Cheating Instances Over Time for {username}')
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], ['No', 'Yes'])
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close('all')
    
    return image_base64

genai.configure(api_key="AIzaSyAU9duCSva2f3XfdJKCEr80VeE3ZjxvW6g")
model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("expain about the highest mountain in the world")
# print(response.text)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text
    
# text = extract_text_from_pdf("attention is all you need.pdf"
# print(text)
# print("-------------------------------------------------------------------------------")

def summarize_text(text):
    try:
        response = model.generate_content([f"please summarize the following text:\n\n{text}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

# summary = summarize_text(text)
# print(summary)
# print("-------------------------------------------------------------------------------")

def question_text(text, question):
    try:
        response = model.generate_content([f"Please answer the following question based on the text:\n\nText: {text}\n\nQuestion: {question}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

# answer = question_text(text,'who are the author of the given paper?')
# print(answer)
# print("-------------------------------------------------------------------------------")
def generate_questions(text):
    try:    
        response = model.generate_content([f"Please generate questions based on the text:\n\nText: {text}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

# question = question_text(text)
# print(question)


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to render aboutUs.html
@app.get("/aboutUs", response_class=HTMLResponse)
async def get_about_us(request: Request):
    return templates.TemplateResponse("aboutUs.html", {"request": request})

# Route to render howItWorks.html
@app.get("/howItWorks", response_class=HTMLResponse)
async def get_how_it_works(request: Request):
    return templates.TemplateResponse("howItWorks.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/teachersMain", response_class=HTMLResponse)
async def teachers_main_page(request: Request):
    return templates.TemplateResponse("teachersMain.html", {"request": request})

@app.get("/scheduleExam", response_class=HTMLResponse)
async def schedule_exam_page(request: Request):
    return templates.TemplateResponse("scheduleExam.html", {"request": request})

@app.get("/studentsList", response_class=HTMLResponse)
async def student_list_page(request: Request):
    return templates.TemplateResponse("studentsList.html", {"request": request})

# Login route (POST)
@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    global current_user
    user = db.query(models.User).filter(
        models.User.username == username,
        models.User.password == password
    ).first()
    
    if user:
        current_user = username  # Store the user's username in the global variable
        if username == "admin" and password == "admin":
            return RedirectResponse(url="/teachersMain", status_code=302)  # Redirect to admin page
        elif username == "teacher" and password == "teacher":
            return RedirectResponse(url="/tdash", status_code=302)
        return RedirectResponse(url="/dash_main", status_code=302)
    return templates.TemplateResponse("login.html", {"request": Request, "msg": "Invalid credentials"})

# Dashboard route (GET)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if current_user:
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})
    return RedirectResponse(url="/")

# Admin route (GET)
@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    if current_user == "admin":
        db: Session = SessionLocal()
        try:
            users = db.query(models.User.username).filter(models.User.username != "admin").distinct().all()
            users = [user.username for user in users]
        finally:
            db.close()
        return templates.TemplateResponse("admin.html", {"request": request, "users": users})
    return RedirectResponse(url="/")

# @app.get("/dashboard", response_class=HTMLResponse)
# async def dashboard(request: Request):
#     if current_user:
#         return templates.TemplateResponse("dashboard.html", {"request": request, "user": current_user})
#     return RedirectResponse(url="/")

@app.get("/admin/user/{username}")
async def admin_user_dashboard(username: str):
    if current_user == "admin":
        # graph = generate_cheating_graph(username)
        graphs = generate_cheating_graph(username)
        if graphs:
            return HTMLResponse(f'<img src="data:image/png;base64,{graphs}" alt="Cheating Graph for {username}">')
        return HTMLResponse('<p>No data available for this user.</p>')
    # raise HTTPException(status_code=403, detail="Not authorized")

@app.get("/video_feed/{username}")
async def video_feed(username: str):
    if current_user and video_feed_active:
        return StreamingResponse(generate_video_feed(username), media_type="multipart/x-mixed-replace; boundary=frame")
    return RedirectResponse(url="/")

@app.get("/video_feed_status")
async def video_feed_status():
    global video_feed_active, start_time, total_time
    
    if video_feed_active and start_time:
        elapsed_time = time.time() - start_time
        time_remaining = max(0, total_time - elapsed_time)

        # If time_remaining is 0, deactivate the video feed
        if time_remaining <= 0:
            video_feed_active = False
            stop_audio_detection()  # Stop audio detection
            if video_cap is not None:
                video_cap.release()  # Release the camera
                video_cap = None  # Reset the capture object
        
        return {"active": video_feed_active, "time_remaining": int(time_remaining)}
    
    return {"active": video_feed_active, "time_remaining": total_time}

@app.post("/alt-tab")
async def handle_alt_tab(request: Request):
    data = await request.json()
    type = data.get('type')
    start_time = data.get('start_time')
    end_time = data.get('end_time')
    time_elapsed = data.get('time_elapsed')

    # Convert timestamps to readable format
    start_time = datetime.fromtimestamp(start_time / 1000) if start_time else None
    end_time = datetime.fromtimestamp(end_time / 1000) if end_time else None

    # Log the received data
    print(f"Type: {type}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Time Elapsed: {time_elapsed}")

    return {"message": "Data received"}

@app.post("/toggle_video_feed")
async def toggle_video_feed():
    global video_feed_active, start_time, cheating_data, video_cap
    video_feed_active = not video_feed_active
    if video_feed_active:
        start_time = time.time()
        cheating_data = []
        start_audio_detection()  # Start audio detection along with video feed
    else:
        stop_audio_detection()  # Stop audio detection when video feed stops
        if video_cap is not None:
            video_cap.release()  # Release the camera
    return {"status": "active" if video_feed_active else "inactive"}

# Route to create a user with error handling
@app.post("/users/", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserBase, db: db_dependency):
    existing_user = db.query(models.User).filter(
        (models.User.username == user.username) | (models.User.password == user.password)
    ).first()

    if existing_user:
        raise HTTPException(status_code=400, detail="User with this username or password already exists")

    # Create a new user entry in the database
    new_user = models.User(username=user.username, password=user.password)
    db.add(new_user)
    db.commit()
    return new_user

@app.post("/add_student")
async def add_student(request:Request,username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    # Check if the user already exists
    existing_user = db.query(User).filter(User.username == username).first()  # Use 'User' class here
    if existing_user:
        return {"error": "User already exists"}
    
    # Add the new user
    new_user = User(username=username, password=password)  # Use 'User' class here
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return templates.TemplateResponse("studentsList.html", {"request": request})


# Route to fetch user information based on username and password
@app.get("/users/{username}", status_code=status.HTTP_200_OK)
async def read_user(username: str, password: str, db: db_dependency):
    user = db.query(models.User).filter(
        models.User.username == username,
        models.User.password == password
    ).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        os.remove(file.filename)  # Cleanup uploaded file
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        
        os.remove(file.filename)  # Cleanup uploaded file
        summary = summarize_text(text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/question/")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        # print(text)
        os.remove(file.filename)  # Cleanup uploaded file
        # question = "Who is the author of this story?"
        print(f"Extracted text: {text}")
        print(f"Received question: {question}")
        answer = question_text(text, question)
        print(question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/generate-questions/")
async def generate(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        os.remove(file.filename)  # Cleanup uploaded file
        questions = generate_questions(text)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    
@app.get("/student_ai/")
async def index_page(request: Request):
    return templates.TemplateResponse("student_ai.html", {"request": request})

@app.get("/dash_main/")
async def index_page(request: Request):
    return templates.TemplateResponse("dash_main.html", {"request": request})

@app.get("/courses/")
async def index_page(request: Request):
    return templates.TemplateResponse("courses.html", {"request": request})

@app.get("/tdash/")
async def index_page(request: Request):
    return templates.TemplateResponse("tdash.html", {"request": request})

@app.get("/add-course/")
async def index_page(request: Request):
    return templates.TemplateResponse("add-course.html", {"request": request})

@app.get("/schexam_qb/")
async def index_page(request: Request):
    return templates.TemplateResponse("schexam_qb.html", {"request": request})

@app.get("/stats/")
async def index_page(request: Request):
    return templates.TemplateResponse("stats.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)