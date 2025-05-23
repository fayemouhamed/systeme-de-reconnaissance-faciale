🧰 Fonctionnement
-----------------
- Utilisation du classificateur Haar pour la détection de visages (`haarcascade_frontalface_alt.xml` requis).
- Lecture des vidéos depuis le dossier `VideosAnalyser`.
- Pour chaque visage détecté :
  - Une image est enregistrée dans le dossier `BDPhoto3/`.
  - Un extrait vidéo associé au visage est enregistré dans `BDVideo3/`.
  - Les informations sont stockées dans les bases SQLite correspondantes.

📁 Structure des bases de données
---------------------------------
- `photos.db` (dans BDPhoto3/) contient : id, person_id, file_path, timestamp
- `videos.db` (dans BDVideo3/) contient : id, person_id, file_path, timestamp

🛠️ Pré-requis supplémentaires
-----------------------------
- OpenCV ---------- pip install opencv-python
- Fichier Haar Cascade (`haarcascade_frontalface_alt.xml`), disponible sur : https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml

🧪 Exemple de commande
----------------------
python extract_faces_from_videos.py

Assure-toi d'avoir placé plusieurs vidéos dans le dossier `VideosAnalyser/`.


📸 Système de Classification d'Images Faciales avec DeepFace
===========================================================

Ce projet Python permet de classer automatiquement des images faciales selon leur similarité, en s'appuyant uniquement sur la bibliothèque DeepFace et une base de données SQLite.

🚀 Fonctionnalités
------------------
- Lecture des chemins d'images depuis une base de données SQLite.
- Suppression automatique des fichiers inexistants.
- Comparaison des visages à l’aide du modèle Facenet de DeepFace.
- Regroupement des visages similaires en clusters.
- Copie des images classées dans des dossiers distincts (person_0, person_1, ...).
- Affichage des métriques de performance et du temps d'exécution.

📁 Structure attendue
---------------------
BDPhoto3/
├── photos.db           # Base de données contenant les chemins des images
├── image1.jpg
├── image2.jpg
...

ClassifiedPhoto3/       # Généré automatiquement
└── Photos/
    ├── person_0/
    ├── person_1/
    ...

🛠️ Pré-requis
-------------
- Python ≥ 3.7
- DeepFace       ------- pip install deepface
- SQLite3        ------- 
- Pillow, NumPy  ------- pip install pillow numpy matplotlib opencv-python scikit-learn facenet-pytorch

Installation rapide :
---------------------
pip install deepface numpy

🧠 Utilisation
--------------
1. Place tes images dans le dossier BDPhoto3/ et assure-toi que la base photos.db contient la table photos(file_path TEXT) avec les bons chemins.
2. Lance le script Python :
   python classify_faces.py
3. Les images classées seront disponibles dans le dossier ClassifiedPhoto3/Photos/.

📊 Exemple de sortie
--------------------
📈 Pourcentage d'images classées : 95.65% (22/23)
✅ Classification terminée avec DeepFace uniquement !
⏱ Temps d'exécution : 1.73 minutes

🧪 Modèle utilisé
-----------------
- Facenet (modèle par défaut dans DeepFace.verify)
- Pas de détection forcée (enforce_detection=False)

📝 Auteur
---------
Projet développé par [Mouhamed FAYE]
Licence : MIT

🎥 Extraction de Visages à partir de Vidéos
==========================

Ce module extrait automatiquement les visages détectés dans des vidéos présentes dans un dossier, enregistre les images extraites dans une base de données `photos.db` et les vidéos segmentées dans `videos.db`.

📦 Classification des visages par clustering (DBSCAN)
===========================

Ce script permet de regrouper automatiquement les images extraites (visages) à partir de vidéos, en clusters représentant des personnes distinctes, via la méthode de clustering DBSCAN.

🔧 Détails techniques :
- Chargement des chemins d'accès depuis la base de données SQLite "photos.db"
- Vérification des fichiers images valides (suppression des entrées invalides)
- Détection faciale avec OpenCV (Haar Cascade)
- Extraction de vecteurs faciaux par mise à l'échelle et flattening de la région du visage
- Clustering avec DBSCAN (paramètres : eps=5.0, min_samples=2)
- Classement des images par personne dans un dossier `ClassifiedPhoto4/Photos/person_X/`

📊 Statistiques finales affichées :
- Nombre d'images traitées
- Nombre d'images classées
- Pourcentage d'images classées
- Temps d'exécution total

💡 Objectif : Générer un classement automatique des visages pour mieux organiser les photos selon les personnes identifiées.
