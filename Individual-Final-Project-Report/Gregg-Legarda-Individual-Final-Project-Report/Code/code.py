"""File: Faces.py
from Code.web_scraper import Insta_Info_Scraper as scraper
def haar(video):
    """'allows to turn of the light of the cam'"""
    os.getenv("OPENCV_VIDEOIO_PRIORITY_MSMF", None)
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    # scraper
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    stroke = 1
    size = 0.4
    # instance class web scraper to get info from instagram
    obj = scraper.Insta_Info_Scraper(font, color, stroke, size)
    # cascades
    cascPath = "./haar/cascades/haarcascade_frontalface_default.xml"
    recognizer.read("./haar/trainer.yml")
    with open("./haar/labels.pickle", 'rb') as f:
    # Draw a rectangle around the face
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = gray[y:y + w, x:x + w]
    gx, gy = np.gradient(cropped)  # take the gradiant to vectorize the images into two values
    cropped = np.sqrt(np.square(gx) + np.square(gy))  # get magnitude to normalize it
    # capture faces and save for training
    # recognizer
    id_, conf = recognizer.predict(cropped)
    if conf >= 1 and conf <= 90:
    open('users.txt', 'w').close()  # clear5 it first
    file1 = open("users.txt", "a")  # append mode
    file1.write("https://www.instagram.com/" + name + "/")
    file1.close()
    # obj.main(frame, x, h, y, conf, w, name, conf)
    obj.main(frame, y, x, 100, 100, name, conf)
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

Total: 59
Modify: 19
Created:14
Ours: 33

File: Gradient.py
while True:
# Capture frame-by-frame
ret, frame = self.video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = frame.astype('uint8')
gx, gy = np.gradient(gray)
cropped = np.sqrt(np.square(gx) + np.square(gy))
frame = cropped ##gradient vector mode
ret, jpeg = cv2.imencode('.jpg', frame)
# cv2.imshow('frame', frame)
# if cv2.waitKey(20) & 0xFF == ord('q'):
#     break
return jpeg.tobytes()
def main():

Total : 11
modify: 1
create: 10

File: Capture_image.py
def save_img(self):
# capture faces and save for training
# print("button CAPTURE was pressed")
video_capture = cv2.VideoCapture(0)
print("[INFO] Taking picture...")
time.sleep(1)
ret, frame = video_capture.read()
saved = self.path + str(self.image_count) + ".jpg"
cv2.imwrite(saved, frame)
self.image_count = self.image_count+1
video_capture.release()
cv2.destroyAllWindows()
del video_capture
print("cerrar camara")
return "[INFO] Image saved"

File: encode_faces.py
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,
help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=False,
help="path to serialized db of facial encodings")


File: Insta_Info_Scraper.py

print('Followers:', followers)
print('Following:', following)
print('Posts:', posts)
print('---------------------------')
# Set info about the user on the screen according to the face on it
def setTextScreen(self, frame, x, h, y, w, name, conf):
# retrieves info from the dictionary according to the user
dict_text = self.getinfo_dict(name)
position = x - 15 if x - 15 > 15 else x + 15
if not conf:
cv2.rectangle(frame, (h, x), (y, w), self.color, 2)
# cv2.putText(frame, name, (h, position), cv2.FONT_HERSHEY_SIMPLEX,0.75,self.color, 2)
cv2.putText(frame, 'User:' + dict_text['User'], (h, position - 45), self.font, self.size, self.color,
self.stroke)
cv2.putText(frame, 'Followers:' + dict_text['Followers'], (h, position - 30), self.font, self.size,
self.color,
self.stroke)
cv2.putText(frame, 'Following:' + dict_text['Following'], (h, position - 15), self.font, self.size,
self.color,
self.stroke)
cv2.putText(frame, 'Posts:' + dict_text['Posts'], (h, position), self.font, self.size, self.color,
self.stroke)
if conf:
cv2.putText(frame, "Confidence" + str(round(conf)) + "%", (100, x), self.font, self.size, self.color,
self.stroke,
cv2.LINE_AA)
def main(self, frame, x, h, y, w, name, conf):
# it verifies if the info to get is new
val = self.check_info(name)
print("this is VAL" + str(val))
if val is False:  # if the face is new get the info by doing a request
    self.setTextScreen(frame, x, h, y, w, name, conf)
    else:  # when the face is not new just get the info from the dictionary
    self.setTextScreen(frame, x, h, y, w, name, conf)
if __name__ == '__main__':
obj = Insta_Info_Scraper()
obj.main()

File: App.py
import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
import math
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import base64
from dash.dependencies import Input, Output
from flask import Flask, Response
from Code.web_scraper import Insta_Info_Scraper as scraper
from Code.capture_image import capture_image as cp
from Code.haar import faces as haar
from Code.hog import encode_faces as ef
from Code.haar import faces_train as train
"""'construct the argument parser and parse the arguments'"""
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=False,
help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
"""'scraper'"""
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.5
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)
writer = None
"""'allows to turn of the light of the cam'"""
os.getenv("OPENCV_VIDEOIO_PRIORITY_MSMF", None)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
class VideoCamera(object):
def __init__(self, vd_type):
print("[INFO] starting video stream...")
self.video = cv2.VideoCapture(0)
self.vd_type = vd_type
def __del__(self):
print("DEL fue ejecutado")
self.video.release()
def gen(camera):
while True:
frame = camera.get_frame()
if args["display"] > 0:
yield (b'--frame\r\n'
b'Content-Type: images/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
server = Flask(__name__)
app = dash.Dash(__name__, server=server)
@server.route('/video_feed')
def video_feed():
return Response(gen(VideoCamera(vd)),
mimetype='multipart/x-mixed-replace; boundary=frame')
# App Layout
app.layout = html.Div(
children=[
# Top Banner Facial recognition
html.Div(
className="study-browser-banner row",
children=[
html.H2(className="h2-title", children="FACIAL RECOGNITION & WEB SCRAPING"),
],
),
html.P('Given all these fascinating applications, we are interested '
'to understand how this technology works by using facial '
'recognition in real time and implementing web scraping '
'to obtain basic information about Instagram users accounts '
'and to present such information on the facial recognition.'),
html.P('In the "Data" tab, you can opt to use capture an image; '
'Train the algorithm or initiate the facial recognition.'
'It is worth to clarify that if you want to perform '
'another task after the facial recogonition is started, it'
'would be necessary to stop the video by pressing "STOP VIDEO"'
'button.'),
# show video
html.Div(
className="eight columns card-left",
children=[
html.H5("Recognition"),
html.Div(
className="bg-white",
children=[
# dcc.Store(id='memory-output'),
html.Div(id='output-video'),
dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="default"),
],
)
],
),
dcc.Store(id="error", storage_type="memory"),
]
)
image_count = 1
@app.callback(Output('loading-output-1', 'children'),
[Input('btn-1', 'n_clicks_timestamp'),
Input('btn-2', 'n_clicks_timestamp'),
Input('btn-3', 'n_clicks_timestamp'),
Input('btn-4', 'n_clicks_timestamp'),
Input('btn-5', 'n_clicks_timestamp'),
Input('btn-6', 'n_clicks_timestamp')])
def displayLoadTrain(btn1, btn2, btn3, btn4, btn5, btn6):
if int(btn2) > int(btn1) and int(btn2) > int(btn3) and int(btn2) > int(btn4) and int(btn2) > int(btn5) and int(btn2) > int(btn6):  # button 2 train
# print('Button 2 was most recently clicked')
time.sleep(1)
ef.encoding()
train.faces_train()
msg = 'Training has finished!'
image_filename = 'saved_images/training_image.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
return html.Div(
[html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())))])"""