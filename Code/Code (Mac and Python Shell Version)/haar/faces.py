import cv2
import sys
import pickle
import numpy as np
#import Insta_Info_Scraper as scraper
from haar import Insta_Info_Scraper as scraper


# scraper
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.4

# instance class web scraper to get info from instagram
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)

# cascades
cascPath = "haar/cascades/haarcascade_frontalface_default.xml"


faceCascade = cv2.CascadeClassifier(cascPath)



# camera actions

class VideoCamera2(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        print("[INFO] starting video stream......")

    def __del__(self):
        print("DEL fue ejecutado")
        self.video.release()

    def get_frame(self):
        # recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("haar/trainer.yml")

        # labels
        labels = {"person_name": 1}
        with open("haar/labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            labels = {v: k for k, v in og_labels.items()}
        
        while True:
            # Capture frame-by-frame
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # FACE
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # loop on every faces per slice of images on the cam
            for (x, y, w, h) in faces:
                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(frame, 'FACE', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

                cropped = gray[y:y + w, x:x + w]
                gx, gy = np.gradient(cropped)  # take the gradiant to vectorize the images into two values
                cropped = np.sqrt(np.square(gx) + np.square(gy))  # get magnitude to normalize it

                ########capture faces and save for training#######
                # saved = "images/face" + str(i) + ".jpg"
                # print("saved")
                # cv2.imwrite(saved, cropped)


                # recognizer
                id_, conf = recognizer.predict(cropped)
                # confidence levels
                if conf >= 1 and conf <= 90:
                    name = labels[id_]

                    # if username is recognized  from the camera, save the url in a text file
                    # to be pulled out later by a scraper
                    open('haar/users.txt', 'w').close()  # clear it first
                    file1 = open("haar/users.txt", "a")  # append mode
                    file1.write("https://www.instagram.com/" + name + "/")
                    file1.close()
                    obj.main(frame, x, h, y, conf, w, name)
                    
            ###############################
            #frame = cropped ##gradient vector mode shows on cam
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()


