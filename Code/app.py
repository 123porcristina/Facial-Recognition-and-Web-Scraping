# import the necessary packages
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

"""construct the argument parser and parse the arguments"""
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

"""scraper"""
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.5
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)

writer = None

"""allows to turn of the light of the cam"""
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

    def get_frame(self):

        if self.vd_type == 1: #HOG

            print("[INFO] loading encodings...")
            data = pickle.loads(open("./encodings.pickle", "rb").read())  ##
            success, frame = self.video.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            """convert the input frame from BGR to RGB then resize it to have
            a width of 750px (to speedup processing)"""
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            """detect the (x, y)-coordinates of the bounding boxes
            corresponding to each face in the input frame, then compute
            the facial embeddings for each face"""
            boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            """loop over the facial embeddings for face detection"""
            for encoding in encodings:
                # attempt to match each face in the input images to our known
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
                name = "Unknown"

                #  get distances and confidence levels
                face_distances = face_recognition.face_distance(encoding, data["encodings"])
                accuracy = self.get_accuracy(face_distances)
                print("[INFO] Confidence Level: "+str(accuracy))


                """check to see if we have found a match"""
                if True in matches:
                    # find the indexes of all matched faces then initialize a
                    # dictionary to count the total number of times each face
                    # was matched
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1

                    # determine the recognized face with the largest number
                    # of votes (note: in the event of an unlikely tie Python
                    # will select first entry in the dictionary)
                    name = max(counts, key=counts.get)
                # update the list of names
                names.append(name)

            # Show the results
            for ((top, right, bottom, left), name) in zip(boxes, names):
                # rescale the face coordinates
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)

                # draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Get and draw info from instagram and user
                open('users.txt', 'w').close()  # clear5 it first
                file1 = open("users.txt", "a")  # append mode
                file1.write("https://www.instagram.com/" + name + "/")
                file1.close()
                obj.main(frame, top, left, right, bottom, name, "")

            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        elif self.vd_type == 2: #gradient
            while True:
                ret, frame = self.video.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = frame.astype('uint8')
                gx, gy = np.gradient(gray)
                cropped = np.sqrt(np.square(gx) + np.square(gy))
                frame = cropped
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()

        elif self.vd_type == 3: #haar
            return haar.haar(self.video)



    def get_accuracy(self, face_distances, face_match_threshold=0.6):
        # print("distances: "+str(face_distances))
        for i, face_distance in enumerate(face_distances):
            if face_distance > face_match_threshold:
                interval = (1.0 - face_match_threshold)
                linear_val = (1.0 - face_distance) / (interval * 2.0)
                return linear_val
            else:
                interval = face_match_threshold
                linear_val = 1.0 - (face_distance / (interval * 2.0))
                return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


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
        # Tabs
        html.Div(id='circos-control-tabs', className='control-tabs', children=[
            dcc.Tabs(id='circos-tabs', value='what-is', children=[
                dcc.Tab(
                    label='About',
                    value='what-is', className='control-tab',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='content', children=[


                            html.H4(className='what-is', children="What is Facial Recognition?"),

                            html.P('At a basic level, facial recognition works by obtaining '
                                   'geometry by scanning the face and recognizing patterns  '
                                   'such as the distance between eyes, the size of the nose '
                                   'and mouth, and so on. With that information, ', ),
                            html.P('the computer can create a virtual map of the face and is '
                                   'then able to perform a match against other faces to identify '
                                   'the appearance of the person who is being captured through a '
                                   'a camera or a digital image (Bala & Watney, 2019). '),
                            html.P('Nowadays, facial recognition has become a subject of great '
                                   'importance in various fields.  For example, it is used in '
                                   'law enforcement to identify criminals, in social media to '
                                   'to recognize friends, in technology to drive cars without a '
                                   'driver, and in smart devices to unlock the device with just '
                                   'the look of the face.'),
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

                            html.Div([
                                'Reference: ',
                                html.A('TechTank paper',
                                       href='https://www.brookings.edu/blog/techtank/2019/06/20/what-are-the-proper-limits-on-police-use-of-facial-recognition/)')
                            ]),
                            html.Div([
                                'For a look into facial recognition and web scraping, please visit the '
                                'original repository ',
                                html.A('here', href='https://github.com/123porcristina/Final-Project-Group)'),
                                '.'
                            ]),

                            html.Br()


                        ]),
                    ], )
                ),

                dcc.Tab(
                    label='Data',
                    value='data',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Actions'),
                            html.Hr(),
                            html.Div(className="'app-controls-block'", children=[
                                html.Label("Directory name *"),
                                dcc.Input(id='input-box', placeholder='Instagram user...', type='text'),
                                html.Br(),
                                html.Br(),
                                html.Button('Capture Picture', id='btn-1', className="control-download",
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Train  Algorithm', id='btn-2',
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Facial Recognition', id='btn-3',
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Stop video', id='btn-4',  n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Haar', id='btn-5',  n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Gradient', id='btn-6',  n_clicks_timestamp=0),

                                # html.Div(id='container-button-timestamp')

                            ]),
                        ]),
                        html.Hr(),
                    ])
                ),

            ])
        ]),
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


@app.callback(Output('output-video', 'children'),
              [Input('btn-1', 'n_clicks_timestamp'),
               Input('btn-2', 'n_clicks_timestamp'),
               Input('btn-3', 'n_clicks_timestamp'),
               Input('btn-4', 'n_clicks_timestamp'),
               Input('btn-5', 'n_clicks_timestamp'),
               Input('btn-6', 'n_clicks_timestamp'),
               Input('input-box', 'value')])
def displayClick(btn1, btn2, btn3, btn4, btn5, btn6, value):

    global image_count
    global vd

    if int(btn1) > int(btn2) and int(btn1) > int(btn3) and int(btn1) > int(btn4) and int(btn1) > int(btn5) and int(btn1) > int(btn6):
        # capture faces and save for training
        print("button CAPTURE was pressed")
        img = cp.CaptureImage(value, image_count)
        print(img.create_dir())
        msg = img.save_img()
        image_filename = 'saved_images/image_captured.png'  # replace with your own image
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return html.Div(
            [html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())))])

    # elif int(btn2) > int(btn1) and int(btn2) > int(btn3) and int(btn2) > int(btn4): #button 2 train
    #     msg = 'Button 2 was most recently clicked'
    #     return None
        # return html.Div([html.Div(msg),])

    elif int(btn3) > int(btn1) and int(btn3) > int(btn2) and int(btn3) > int(btn4) and int(btn3) > int(btn5) and int(btn3) > int(btn6):  # button 3 - Recognition
        vd = 1
        return html.Div([html.Div(html.Img(src="/video_feed"))])

    elif int(btn4) > int(btn1) and int(btn4) > int(btn2) and int(btn4) > int(btn3) and int(btn4) > int(btn5) and int(btn4) > int(btn6):
        print("[INFO] Facial recognition has been STOPPED (button4: stop video pressed)")
        cv2.destroyAllWindows()
        image_filename = 'saved_images/stop_image.png'  # replace with your own image
        encoded_image = base64.b64encode(open(image_filename, 'rb').read())
        return html.Div(
            [html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())))])
        # return html.Div([html.Div(html.Img(src=" "),)])


    elif int(btn5) > int(btn1) and int(btn5) > int(btn3) and int(btn5) > int(btn4) and int(btn5) > int(btn6):  # btn 5 - HAAR
        print("[INFO] HAAR")
        vd = 3
        gen(camera=vd)
        return html.Div([html.Div(html.Img(src="/video_feed"))])

    elif int(btn6) > int(btn1) and int(btn6) > int(btn3) and int(btn6) > int(btn4) and int(btn6) > int(btn5):  # btn 5 - HAAR
        print("[INFO] GRADIENT")
        vd = 2
        gen(camera=vd)
        return html.Div([html.Div(html.Img(src="/video_feed"))])

    else:
        msg = 'None of the buttons have been clicked yet'
        return html.Div([])


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
            [html.Div(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode())))])


if __name__ == '__main__':
    app.run_server(debug=True)