# debruitage_demucs_gradio

# Traitement Audio : Détection et Débruitage
## Description du Projet
Ce projet est une application de traitement audio développée avec Gradio, permettant aux utilisateurs de télécharger des fichiers audio et d'appliquer diverses techniques de traitement, notamment la détection de bruit et le débruitage. L'application utilise le modèle pré-entraîné Demucs, qui est conçu pour séparer les sources audio, permettant ainsi d'isoler les voix et de réduire le bruit de fond dans les enregistrements.
Fonctionnalités
L'application comprend plusieurs fonctionnalités accessibles via une interface conviviale :
Débruitage : Cette fonctionnalité permet aux utilisateurs de télécharger un fichier audio (au format MP3 ou WAV) et d'isoler la voix en réduisant le bruit de fond. L'utilisateur peut écouter la voix isolée et la télécharger.
Détectage de Bruit : Les utilisateurs peuvent télécharger un fichier audio pour détecter les segments où le bruit est présent. L'application analyse le fichier audio et génère un graphique montrant les intervalles de bruit détectés, ainsi qu'un message indiquant le nombre de segments détectés.
Correction Rapide : Cette fonctionnalité permet d'appliquer un filtre passe-bas à la voix isolée pour lisser le signal audio. Les utilisateurs peuvent écouter la voix filtrée et la télécharger.
Technologies Utilisées
Gradio : Pour créer l'interface utilisateur interactive.
PyTorch : Pour l'utilisation du modèle Demucs.
Librosa : Pour l'analyse et le traitement des signaux audio.
Soundfile : Pour lire et écrire des fichiers audio.
Matplotlib : Pour générer des graphiques visuels des données audio.
Installation
Pour exécuter ce projet, assurez-vous d'avoir Python 3.x installé sur votre machine. Ensuite, installez les dépendances nécessaires en utilisant pip :
bash
# pip install gradio torch torchaudio librosa matplotlib soundfile

# Utilisation
Clonez ce dépôt sur votre machine locale.
Placez l'image image_audio.jpeg dans le même répertoire que le script Python.
Exécutez l'application avec la commande suivante :
bash
# python votre_script.py

Ouvrez votre navigateur web à l'adresse indiquée dans la console pour accéder à l'application.
# Contribuer
Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet, n'hésitez pas à soumettre une demande de tirage (pull request). N'hésitez pas à personnaliser ce texte selon vos besoins spécifiques ou à ajouter d'autres sections si nécessaire, comme des informations sur les auteurs ou des notes sur la licence du projet.