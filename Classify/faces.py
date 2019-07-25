import cv2
import sys
import pickle
import numpy as np


cascPath = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_fullbody.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
bodyCascade = cv2.CascadeClassifier(cascPath2)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}




video_capture = cv2.VideoCapture(0)
i=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ####### FACE
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # loop on every faces per slice of image on the cam
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame, 'FACE', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cropped = gray[y:y + w, x:x + w]
        gx, gy = np.gradient(cropped)  # take the gradiant to vectorize the image into two values
        cropped = np.sqrt(np.square(gx) + np.square(gy))  # get magnitude to normalize it

        # capture faces and save for training
        #saved = "images/face" + str(i) + ".jpg"
        # print("saved")
        #cv2.imwrite(saved, cropped)
        i=i+1

        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(cropped)
        if conf >= 15 and conf <= 95:
            # print(5: #id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] + str(round(conf))+ "%"
            color = (255, 255, 255)
            stroke = 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)


    # Display the resulting frame
    #cv2.imshow('Video', cropped) # this is what the gradient vector image looks like
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
