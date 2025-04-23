import os
import shutil
import sqlite3
import numpy as np
import time
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# ⏱ Début du chronomètre
t0 = time.time()

# 📁 Répertoires
base_photo_dir = "BDPhoto3"
classified_dir = "ClassifiedPhoto3"
photo_classified_dir = os.path.join(classified_dir, "Photos")
os.makedirs(photo_classified_dir, exist_ok=True)

# 🔌 Connexion à la base SQLite
photo_db_path = os.path.join(base_photo_dir, "photos.db")
photo_conn = sqlite3.connect(photo_db_path)
photo_cursor = photo_conn.cursor()

# 📥 Chargement des chemins de fichiers
photo_cursor.execute("SELECT file_path FROM photos")
photo_data = [row[0] for row in photo_cursor.fetchall()]

# 🧠 Fonction d'extraction des embeddings
def extract_embedding(image_path):
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False
        )
        return np.array(embedding[0]["embedding"])
    except Exception as e:
        print(f"⚠️ Erreur d'extraction pour {image_path} : {e}")
        return None

# 🔍 Traitement des images
embeddings = []
valid_paths = []
invalid_paths = []

for file_path in photo_data:
    if os.path.exists(file_path):
        embedding = extract_embedding(file_path)
        if embedding is not None:
            embeddings.append(embedding)
            valid_paths.append(file_path)
    else:
        invalid_paths.append(file_path)

# 🧹 Nettoyage de la base
if invalid_paths:
    for path in invalid_paths:
        photo_cursor.execute("DELETE FROM photos WHERE file_path = ?", (path,))
    photo_conn.commit()
    print(f"🧹 {len(invalid_paths)} chemins invalides supprimés de la base de données.")

# 📦 Clustering avec DBSCAN
if embeddings:
    embeddings = normalize(np.array(embeddings))
    clustering = DBSCAN(metric="euclidean", eps=0.7, min_samples=2).fit(embeddings)
    labels = clustering.labels_

    # 📁 Organisation par clusters
    clusters = {}
    for label, path in zip(labels, valid_paths):
        if label == -1:
            continue  # Ignorer les outliers
        clusters.setdefault(f"person_{label}", []).append(path)

    for label, paths in clusters.items():
        person_dir = os.path.join(photo_classified_dir, label)
        os.makedirs(person_dir, exist_ok=True)
        for file_path in paths:
            dest = os.path.join(person_dir, os.path.basename(file_path))
            if not os.path.exists(dest):
                shutil.copy(file_path, dest)

    # 📈 Statistiques
    total_images = len(valid_paths) + len(invalid_paths)
    classified = sum(len(paths) for paths in clusters.values())
    rate = (classified / total_images) * 100 if total_images > 0 else 0

    print(f"📈 Pourcentage d'images classées : {rate:.2f}% ({classified}/{total_images})")

else:
    print("🚫 Aucun embedding généré ou image valide détectée.")

# 🔚 Fermeture
photo_conn.close()

# 🕒 Fin
t1 = time.time()
print("✅ Classification des images terminée avec une combinaison de DeepFace + DBSCAN !")
print("⏱ Temps d'exécution : {:.2f} minutes".format((t1 - t0) / 60))
