import os
import shutil
import sqlite3
import numpy as np
import time
from deepface import DeepFace

# Début du chronomètre
t0 = time.time()

# Définition des répertoires
base_photo_dir = "BDPhoto3"
classified_dir = "ClassifiedPhoto1"
os.makedirs(classified_dir, exist_ok=True)

# Connexion à la base de données SQLite
photo_db_path = os.path.join(base_photo_dir, "photos.db")
photo_conn = sqlite3.connect(photo_db_path)
photo_cursor = photo_conn.cursor()

# Création du répertoire de classification
photo_classified_dir = os.path.join(classified_dir, "Photos")
os.makedirs(photo_classified_dir, exist_ok=True)

# Récupération des chemins depuis la base
photo_cursor.execute("SELECT file_path FROM photos")
photo_data = [row[0] for row in photo_cursor.fetchall()]

# Nettoyage des fichiers inexistants
valid_photo_data = []
invalid_paths = []
for file_path in photo_data:
    if os.path.exists(file_path):
        valid_photo_data.append(file_path)
    else:
        photo_cursor.execute("DELETE FROM photos WHERE file_path = ?", (file_path,))
        invalid_paths.append(file_path)
        print(f"❌ Chemin invalide supprimé : {file_path}")

photo_conn.commit()

# Initialisation
clusters = []
cluster_index = 0
processed = set()

for i, img1_path in enumerate(valid_photo_data):
    if img1_path in processed:
        continue

    current_cluster = [img1_path]
    processed.add(img1_path)

    for j in range(i + 1, len(valid_photo_data)):
        img2_path = valid_photo_data[j]
        if img2_path in processed:
            continue

        try:
            result = DeepFace.verify(img1_path, img2_path, model_name="Facenet", enforce_detection=False)
            if result["verified"]:
                current_cluster.append(img2_path)
                processed.add(img2_path)
        except Exception as e:
            print(f"Erreur lors de la comparaison : {img1_path} vs {img2_path} → {e}")

    clusters.append((f"person_{cluster_index}", current_cluster))
    cluster_index += 1

# Copie des images dans les dossiers correspondants
for label, paths in clusters:
    person_dir = os.path.join(photo_classified_dir, label)
    os.makedirs(person_dir, exist_ok=True)
    for file_path in paths:
        dest_path = os.path.join(person_dir, os.path.basename(file_path))
        if not os.path.exists(dest_path):
            shutil.copy(file_path, dest_path)

# 📊 Métriques de performance
total_images = len(valid_photo_data) + len(invalid_paths)
classified_images = sum(len(paths) for _, paths in clusters)
classification_rate = (classified_images / total_images) * 100 if total_images > 0 else 0
print(f"📈 Pourcentage d'images classées : {classification_rate:.2f}% ({classified_images}/{total_images})")

# 👥 Pourcentage de clusters avec plusieurs visages
multi_face_clusters = [paths for _, paths in clusters if len(paths) > 1]
percentage_multi_face_clusters = (len(multi_face_clusters) / len(clusters)) * 100 if clusters else 0
print(f"👥 Pourcentage de clusters contenant plusieurs visages : {percentage_multi_face_clusters:.2f}% ({len(multi_face_clusters)}/{len(clusters)})")

# Fermeture
photo_conn.close()

# Fin du chronomètre
t1 = time.time()
print("✅ Classification terminée avec OpenCV + DeepFace Uniquement !")
print(f"⏱ Temps d'exécution : {(t1 - t0) / 60:.2f} minutes")
