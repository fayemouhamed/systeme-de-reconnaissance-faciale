import cv2
import os
import sqlite3
from datetime import datetime

# === PARAMÈTRES ===
haar_file = "haarcascade_frontalface_alt.xml"
video_input_dir = "VideosAnalyser"
base_photo_dir = "BDPhoto3"
base_video_dir = "BDVideo3"
os.makedirs(base_photo_dir, exist_ok=True)
os.makedirs(base_video_dir, exist_ok=True)

# === CHARGEMENT DU CLASSIFICATEUR HAAR ===
if not os.path.exists(haar_file):
    print(f"Erreur : Le fichier {haar_file} est introuvable.")
    exit()
face_cascade = cv2.CascadeClassifier(haar_file)

# === BASES DE DONNÉES ===
photo_db_path = os.path.join(base_photo_dir, "photos.db")
video_db_path = os.path.join(base_video_dir, "videos.db")
photo_conn = sqlite3.connect(photo_db_path)
photo_cursor = photo_conn.cursor()
video_conn = sqlite3.connect(video_db_path)
video_cursor = video_conn.cursor()

# Création des tables
photo_cursor.execute('''
    CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        file_path TEXT,
        timestamp TEXT
    )
''')
photo_conn.commit()

video_cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        file_path TEXT,
        timestamp TEXT
    )
''')
video_conn.commit()

# === TRAITEMENT DE CHAQUE VIDÉO ===
for filename in os.listdir(video_input_dir):
    if not filename.endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(video_input_dir, filename)
    print(f"▶️ Traitement de : {filename}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur d’ouverture de : {video_path}")
        continue

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frames_to_capture = fps * 360  # 240 secondes par vidéo
    frame_count = 0
    video_writers = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Fin de la vidéo.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(100, 100))

        for i, (x, y, w, h) in enumerate(faces):
            person_id = f"{os.path.splitext(filename)[0]}_person_{i}"
            head_y = max(y - int(0.3 * h), 0)
            head_h = int(h * 1.4)
            head_x = max(x - int(0.2 * w), 0)
            head_w = int(w * 1.3)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = os.path.join(base_photo_dir, f"{person_id}_{timestamp}.jpg")
            cv2.imwrite(img_filename, frame[head_y:head_y + head_h, head_x:head_x + head_w])
            photo_cursor.execute("INSERT INTO photos (person_id, file_path, timestamp) VALUES (?, ?, ?)",
                                 (person_id, img_filename, timestamp))
            photo_conn.commit()

            if person_id not in video_writers:
                video_filename = os.path.join(base_video_dir, f"{person_id}_{timestamp}.avi")
                video_writers[person_id] = cv2.VideoWriter(video_filename, fourcc, fps, (head_w, head_h))
                video_cursor.execute("INSERT INTO videos (person_id, file_path, timestamp) VALUES (?, ?, ?)",
                                     (person_id, video_filename, timestamp))
                video_conn.commit()

            cropped_face = frame[head_y:head_y + head_h, head_x:head_x + head_w]
            video_writers[person_id].write(cropped_face)

        frame_count += 1
        if frame_count >= frames_to_capture:
            break

    cap.release()
    for writer in video_writers.values():
        writer.release()

# Fermer les connexions
photo_conn.close()
video_conn.close()
print("✅ Traitement de toutes les vidéos terminé !")
