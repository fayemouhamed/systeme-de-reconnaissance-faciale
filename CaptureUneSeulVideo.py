import cv2
import os
import sqlite3
from datetime import datetime

# === PARAMÈTRES ===
haar_file = "haarcascade_frontalface_alt2.xml"
video_source = 0  # Mettre ici le chemin d'une vidéo OU 0 pour la webcam
#video_source = "video3.mp4"
# === VÉRIFICATION DU CLASSIFICATEUR HAAR ===
if not os.path.exists(haar_file):
    print(f"Erreur : Le fichier {haar_file} est introuvable.")
    exit()
face_cascade = cv2.CascadeClassifier(haar_file)

# === OUVERTURE DE LA SOURCE VIDÉO ===
if isinstance(video_source, str) and not os.path.exists(video_source):
    print(f"Erreur : Le fichier vidéo {video_source} est introuvable.")
    exit()
cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la source vidéo.")
    exit()

# === CRÉATION DES DOSSIERS DE SORTIE ===
base_photo_dir = "BDPhoto1"
base_video_dir = "BDVideo1"
os.makedirs(base_photo_dir, exist_ok=True)
os.makedirs(base_video_dir, exist_ok=True)

# === BASES DE DONNÉES SQLITE ===
photo_conn = sqlite3.connect(os.path.join(base_photo_dir, "photos.db"))
video_conn = sqlite3.connect(os.path.join(base_video_dir, "videos.db"))
photo_cursor = photo_conn.cursor()
video_cursor = video_conn.cursor()

# Tables
photo_cursor.execute('''
    CREATE TABLE IF NOT EXISTS photos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        file_path TEXT,
        timestamp TEXT
    )
''')
video_cursor.execute('''
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        file_path TEXT,
        timestamp TEXT
    )
''')
photo_conn.commit()
video_conn.commit()

# === CAPTURE ET TRAITEMENT ===
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
max_duration_seconds = 240
frames_to_capture = fps * max_duration_seconds
frame_count = 0
video_writers = {}

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Fin de la vidéo ou arrêt du flux.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(100, 100))

    for i, (x, y, w, h) in enumerate(faces):
        person_id = f"person_{i}"
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

        # Affichage en direct avec cadre vert
        cv2.rectangle(frame, (head_x, head_y), (head_x + head_w, head_y + head_h), (0, 255, 0), 2)

    frame_count += 1
    if isinstance(video_source, str) and frame_count >= frames_to_capture:
        break

    cv2.imshow("Détection de visages", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === FERMETURE ===
cap.release()
for writer in video_writers.values():
    writer.release()
photo_conn.close()
video_conn.close()
cv2.destroyAllWindows()
print("✅ Traitement terminé.")
