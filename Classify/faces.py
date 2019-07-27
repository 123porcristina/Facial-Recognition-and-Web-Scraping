import cv2
import sys
import pickle
import numpy as np
from web_scraping import Insta_Info_Scraper as scraper

# scraper
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.4

# instance class web scraper to get info from instagram
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)

# cascades
cascPathrightpalm = "cascades/haarcascade_rightpalm.xml"
cascPathrightfist = "cascades/haarcascade_rightfist.xml"
cascPath = "cascades/haarcascade_frontalface_default.xml"
cascPathbody = "cascades/haarcascade_fullbody.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
bodyCascade = cv2.CascadeClassifier(cascPathbody)
rightpalmCascade = cv2.CascadeClassifier(cascPathrightpalm)
rightfistCascade = cv2.CascadeClassifier(cascPathrightfist)

# recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# labels
labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

# camera actions
video_capture = cv2.VideoCapture(0)
i = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FACE
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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'FACE', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        cropped = gray[y:y + w, x:x + w]
        gx, gy = np.gradient(cropped)  # take the gradiant to vectorize the image into two values
        cropped = np.sqrt(np.square(gx) + np.square(gy))  # get magnitude to normalize it

        ########capture faces and save for training#######
        # saved = "images/face" + str(i) + ".jpg"
        # print("saved")
        # cv2.imwrite(saved, cropped)
        i = i + 1

        # recognizer
        id_, conf = recognizer.predict(cropped)
        # confidence levels
        if conf >= 1 and conf <= 90:
            name = labels[id_]

            # if username is recognized  from the camera, save the url in a text file
            # to be pulled out later by a scraper
            open('users.txt', 'w').close()  # clear it first
            file1 = open("users.txt", "a")  # append mode
            file1.write("https://www.instagram.com/" + name + "/")
            file1.close()
            obj.main(frame, x, h, y, conf, w, name)

    # HAND GESTURES
    # right palm
    rightpalms = rightpalmCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in rightpalms:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'rightpalm', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)
    # right fists
    rightfists = rightfistCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in rightfists:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'rightfist', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame###
    # cv2.imshow('Video', cropped) # this is what the gradient vector image looks like
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
