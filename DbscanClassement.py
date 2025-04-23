import os
import shutil
import sqlite3
import numpy as np
import time
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ⏱ Début du chronomètre
t0 = time.time()

# 📁 Répertoires
base_photo_dir = "BDPhoto3"
classified_dir = "ClassifiedPhoto2"
photo_classified_dir = os.path.join(classified_dir, "Photos")
os.makedirs(photo_classified_dir, exist_ok=True)

# 🔌 Connexion à la base SQLite
photo_db_path = os.path.join(base_photo_dir, "photos.db")
photo_conn = sqlite3.connect(photo_db_path)
photo_cursor = photo_conn.cursor()

# 📥 Chargement des chemins de fichiers
photo_cursor.execute("SELECT file_path FROM photos")
photo_data = [row[0] for row in photo_cursor.fetchall()]

# 🧹 Nettoyage des chemins invalides
valid_photo_data = []
invalid_paths = []
for path in photo_data:
    if os.path.exists(path):
        valid_photo_data.append(path)
    else:
        photo_cursor.execute("DELETE FROM photos WHERE file_path = ?", (path,))
        invalid_paths.append(path)
        print(f"❌ Chemin invalide supprimé : {path}")
photo_conn.commit()

# 🤖 Détection faciale avec OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face = cv2.resize(gray[y:y+h, x:x+w], (100, 100)).flatten()
    return face / 255.0  # Normalisation

# 🎯 Extraction des vecteurs faciaux
embeddings = []
valid_paths = []

for path in valid_photo_data:
    embedding = extract_face_embedding(path)
    if embedding is not None:
        embeddings.append(embedding)
        valid_paths.append(path)

# 📦 Clustering avec DBSCAN
if embeddings:
    X = np.array(embeddings)
    dbscan = DBSCAN(eps=10.0, min_samples=2, metric='euclidean')
    labels = dbscan.fit_predict(X)

    clusters = {}
    for label, path in zip(labels, valid_paths):
        if label == -1:
            continue  # Ignorer les non-classifiés
        clusters.setdefault(f"person_{label}", []).append(path)

    # 🗂 Organisation des fichiers par cluster
    for label, paths in clusters.items():
        person_dir = os.path.join(photo_classified_dir, label)
        os.makedirs(person_dir, exist_ok=True)
        for file_path in paths:
            dest = os.path.join(person_dir, os.path.basename(file_path))
            shutil.copy(file_path, dest)

    # 📊 Statistiques
    total_images = len(valid_photo_data) + len(invalid_paths)
    classified = sum(len(paths) for paths in clusters.values())
    rate = (classified / total_images) * 100 if total_images > 0 else 0

    print(f"📈 Pourcentage d'images classées : {rate:.2f}% ({classified}/{total_images})")

    # 👥 Nombre de sous-dossiers contenant plusieurs visages différents
    seuil_similarite = 0.7
    multi_face_count = 0
    for label, paths in clusters.items():
        if len(paths) < 2:
            continue  # besoin d'au moins 2 images pour comparaison

        embeddings_cluster = []
        for path in paths:
            emb = extract_face_embedding(path)
            if emb is not None:
                embeddings_cluster.append(emb)

        if len(embeddings_cluster) >= 2:
            sims = cosine_similarity(embeddings_cluster)
            np.fill_diagonal(sims, 0)
            moyenne_sim = np.mean(sims)
            if moyenne_sim < seuil_similarite:
                multi_face_count += 1

    print(f"🔍 Nombre de sous-dossiers contenant plusieurs visages différents : {multi_face_count} / {len(clusters)}")

else:
    print("🚫 Aucun visage détecté ou embedding généré.")

# 🔚 Fermeture
photo_conn.close()

# ⏲ Fin
t1 = time.time()
print("✅ Classification terminée avec OpenCV + DBSCAN uniquement !")
print(f"⏱ Temps d'exécution : {(t1 - t0) / 60:.2f} minutes")
