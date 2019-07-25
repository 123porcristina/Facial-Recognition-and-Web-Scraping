import cv2
import sys
import pickle
import numpy as np

import requests
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json
import time


cascPathrightpalm = "haarcascade_rightpalm.xml"
cascPathrightfist = "haarcascade_rightfist.xml"
cascPath = "haarcascade_frontalface_default.xml"
cascPathbody = "haarcascade_fullbody.xml"

faceCascade = cv2.CascadeClassifier(cascPath)
bodyCascade = cv2.CascadeClassifier(cascPathbody)
rightpalmCascade = cv2.CascadeClassifier(cascPathrightpalm)
rightfistCascade = cv2.CascadeClassifier(cascPathrightfist)


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
        if conf >= 1 and conf <= 45:
            # print(5: #id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 1
            size = 0.4



            ####if username is recognized  from the camera, save the url in a text file to be pulled out later by a scraper
            open('users.txt', 'w').close()#clear it first
            file1 = open("users.txt", "a")  # append mode
            file1.write("https://www.instagram.com/" + name +"/")
            file1.close()


            ########SCRAPER#################
            ##scraper pulls data and shows the details on the screen
            class Insta_Info_Scraper:

                def getinfo(self, url):
                    html = urllib.request.urlopen(url, context=self.ctx).read()
                    soup = BeautifulSoup(html, 'html.parser')
                    data = soup.find_all('meta', attrs={'property': 'og:description'})



                    text = data[0].get('content').split()
                    user = '%s %s %s' % (text[-3], text[-2], text[-1])
                    followers = text[0]
                    following = text[2]
                    posts = text[4]
                    print('User:', user)
                    print('Followers:', followers)
                    print('Following:', following)
                    print('Posts:', posts)
                    print('---------------------------')
                    cv2.putText(frame, 'User:' + user, (x, y+h+15), font, size, color, stroke, cv2.LINE_AA)
                    cv2.putText(frame, 'Followers:'+ followers, (x, y+h+25), font, size, color, stroke, cv2.LINE_AA)
                    cv2.putText(frame, 'Following:'+ following, (x, y+h+35), font, size, color, stroke, cv2.LINE_AA)
                    cv2.putText(frame, 'Posts:'+ posts, (x, y+h+45), font, size, color, stroke, cv2.LINE_AA)
                    cv2.putText(frame, "Confidence" + str(round(conf)) + "%", (x, y+h+55), font, size, color, stroke, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                def main(self):
                    self.ctx = ssl.create_default_context()
                    self.ctx.check_hostname = False
                    self.ctx.verify_mode = ssl.CERT_NONE

                    with open('users.txt') as f:
                        self.content = f.readlines()
                    self.content = [x.strip() for x in self.content]
                    for url in self.content:
                        print("before 125")
                        self.getinfo(url)


            if __name__ == '__main__':
                print("before 130")
                obj = Insta_Info_Scraper()
                print("before 132")
                obj.main()



    ####### HAND

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




   ## Display the resulting frame###
    #cv2.imshow('Video', cropped) # this is what the gradient vector image looks like
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


