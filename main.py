import cv2
import subprocess
from Speak import Say

varified = False
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainers/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX
names = ['', 'archit', 'puskar', 'mayank', 'archit','archit']

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
i = 50
while i>0:
    ret, img = cam.read()
    i = i-1
    converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        converted_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, accuracy = recognizer.predict(converted_image[y:y+h, x:x+w])
        
        # Check if id is within the range of names list and accuracy is below a certain threshold
        threshold = 80  # Adjust as needed
        if 0 <= id < len(names) and accuracy > threshold:
            print(id,accuracy)
            print(names[id])
            accuracy = "  {0}%".format(round(100 - accuracy))
            varified = True
        else:
           
            id = "unknown"
            print(id,accuracy)
            accuracy = "  {0}%".format(round(100 - accuracy))

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(accuracy), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

        # Check if face is verified, and if so, start the main file
        if varified:
            print("Face verified. Starting the main file.")
            Say("Face verified. Welcome back sir, good to see you again")
            cam.release()
            cv2.destroyAllWindows()

            # Start the main file using subprocess
            subprocess.run(["python", "Jarvis.py"])
            exit()

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("Unable to verify,try again later")
cam.release()
cv2.destroyAllWindows()
