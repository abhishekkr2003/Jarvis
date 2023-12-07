import cv2
import numpy as np
from PIL import Image
import os

path = 'samples'
trainer_dir = 'trainers'
trainer_path = os.path.join(trainer_dir, 'trainer.yml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def create_trainer_directory():
    try:
        # Check if 'trainers' directory exists, create if not
        os.makedirs(trainer_dir, exist_ok=True)
        print(f"Directory '{trainer_dir}' created successfully.")
    except Exception as e:
        print(f"Error creating directory: {e}")

def Images_And_Labels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        gray_img = Image.open(imagePath).convert('L')
        img_arr = np.array(gray_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_arr)

        for (x, y, w, h) in faces:
            faceSamples.append(img_arr[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

# Ensure 'trainers' directory is created
create_trainer_directory()

print("Training faces. It will take a few seconds. Wait ...")

faces, ids = Images_And_Labels(path)

if faces and ids:
    recognizer.train(faces, np.array(ids))
    recognizer.save(trainer_path)
    print(f"Model trained. Trainer saved at: {trainer_path}")
else:
    print("No training data found. Make sure there are samples in the 'samples' folder.")
