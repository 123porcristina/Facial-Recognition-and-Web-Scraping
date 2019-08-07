# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
# import Insta_Info_Scraper as scraper
from Prueba import Insta_Info_Scraper as scraper
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from flask import Flask, Response
import cv2

import datetime
import json



# construct the argument parser and parse the arguments
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

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())

# scraper
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.5
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()

writer = None


# time.sleep(2.0)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #####################################################
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
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

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # saved = "images/face" + str(i) + ".jpg"
            # cv2.imwrite(saved, frame)

            # draw the predicted face name and instagram status on the image
            # if username is recognized  from the camera, save the url in a text file
            # to be pulled out later by a scraper
            open('users.txt', 'w').close()  # clear5 it first
            file1 = open("users.txt", "a")  # append mode
            file1.write("https://www.instagram.com/" + name + "/")
            file1.close()
            obj.main(frame, top, left, right, bottom, name)

        ###############################
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        if args["display"] > 0:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



server = Flask(__name__)
app = dash.Dash(__name__, server=server,  external_stylesheets=[dbc.themes.BOOTSTRAP])


@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
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

        # Body of the App
        html.Div(
            className="row app-body",
            children=[
                # User Controls
                html.Div(
                    className="four columns card",
                    children=[
                        html.Div(
                            className="bg-white user-control",
                            children=[
                                html.Div(id='vp-control-tabs', className='control-tabs', children=[
                                    dcc.Tabs(id='vp-tabs', value='what-is', children=[
                                        dcc.Tab(
                                            label='About',
                                            value='what-is',
                                            children=html.Div(className='control-tab', children=[
                                                html.Br(),
                                                html.H4(className='what-is', children='What is face recognition?'),
                                                html.P(

                                                    'This project was created by Cristina Giraldo '
                                                    'and Gregg Legarda. We created this project because'
                                                    'we have curiosity about how computer vision works '
                                                    'in the facial recognition area. '

                                                ),
                                                html.P(

                                                    'In the "Data" tab, you can take a picture '
                                                    'train the algorithm, and of course start the video '
                                                    'to see how facial recognition works.'

                                                ),
                                                html.P(

                                                    'The video takes some time to initialize so please '
                                                    'be patient. '
                                                )
                                            ])
                                        ),
                                        dcc.Tab(
                                            label='Data',
                                            value='data',
                                            children=html.Div(className='control-tab', children=[
                                                html.Div(className='app-controls-block', children=[
                                                    html.Div(className='app-controls-name',
                                                             children='Actions: '
                                                             ),

                                                    html.Div([
                                                        html.Button('Capture Picture', id='btn-1',
                                                                    n_clicks_timestamp=0),
                                                        html.Button('Train  Algorithm', id='btn-2',
                                                                    n_clicks_timestamp=0),
                                                        html.Button('Facial Recognition', id='btn-3',
                                                                    n_clicks_timestamp=0),
                                                        html.Button('Button 4', id='btn-4', n_clicks_timestamp=0),

                                                        dbc.Button("Primary", outline=True, color="primary",
                                                                   className="mr-1", id='btn-5'),
                                                        dbc.Button(
                                                            "Secondary", outline=True, color="secondary",
                                                            className="mr-1"
                                                        ),
                                                        dbc.Button("Success", outline=True, color="success",
                                                                   className="mr-1"),
                                                        dbc.Button("Warning", outline=True, color="warning",
                                                                   className="mr-1"),
                                                        dbc.Button("Danger", outline=True, color="danger",
                                                                   className="mr-1"),
                                                        dbc.Button("Info", outline=True, color="info",
                                                                   className="mr-1"),
                                                        dbc.Button("Light", outline=True, color="light",
                                                                   className="mr-1"),
                                                        dbc.Button("Dark", outline=True, color="dark"),





                                                        html.Div(id='container-button-timestamp')



                                                    ])

                                                ])
                                            ])
                                        ),
                                    ]),
                                ]),
                            ],
                        )
                    ],
                ),
                # Graph
                html.Div(
                    className="eight columns card-left",
                    children=[
                        html.Div(id="video1",
                            className="bg-white",
                            children=[
                                html.H5("Recognition"),
                                html.Img(id="video"),
                                dcc.Loading(id="loading-1", children=[html.Div(id="output-1")], type="default"),
                                # html.Img(src="/video_feed")
                                # dcc.Graph(id="plot"),
                            ],
                        ),

                        ###loading...
                        # html.Div(
                        #     children=[
                        #         dcc.Loading(id="loading-1", children=[html.Div(id="output-1")], type="default"),
                        #         # dcc.Input(id="input-1", value='Input triggers local spinner'),
                        #     ],
                        # ),#endloading...


                    ],
                ),
                # dcc.Store(id="error", storage_type="memory"),
            ],
        ),
    ]
)


@app.callback(Output('video1', 'children'),
              [Input('btn-1', 'n_clicks_timestamp'),
               # Input('btn-2', 'n_clicks_timestamp'),
               Input('btn-3', 'n_clicks_timestamp')])
# def displayClick(btn1, btn2, btn3):
def displayClick(btn1, btn3):
    # if int(btn1) > int(btn2) and int(btn1) > int(btn3):
    if int(btn1) > int(btn3):
        msg = 'Button 1 was most recently clicked'
        print('Button 1 was most recently clicked')
        prueba = "botoncito1"
    # elif int(btn2) > int(btn1) and int(btn2) > int(btn3): #button 2 train
        # msg = 'Button 2 was most recently clicked'
        # print('Button 2 was most recently clicked')
        # prueba = "botoncito2"
        # import encode_faces # calls the encoding when button train is pressed
    # elif int(btn3) > int(btn1) and int(btn3) > int(btn2):  # button 3 - Recognition
    elif int(btn3) > int(btn1) :#button 3 - Recognition
        msg = 'Button 3 was most recently clicked'
        print('Button 3 was most recently clicked')
        # # html.Img(src="/video_feed")
        return html.Div([
            # html.Img(src="/video_feed")
            # html.Div('btn1: {}'.format(btn1)),
            # html.Div('btn2: {}'.format(btn2)),
            # html.Div('btn3: {}'.format(btn3)),
            # html.Div(msg),
            html.Div(html.Img(src="/video_feed")),
        ])
    else:
        msg = 'None of the buttons have been clicked yet'
        return html.Div([])
    # return html.Div([
    #     # html.Img(src="/video_feed")
    #     html.Div('btn1: {}'.format(btn1)),
    #     html.Div('btn2: {}'.format(btn2)),
    #     html.Div('btn3: {}'.format(btn3)),
    #     html.Div(msg),
    #     html.Div(html.Img(src="/video_feed")),
    # ])


@app.callback(Output('output-1', 'children'),
              [Input('btn-2', 'n_clicks_timestamp')])
def displayLoadTrain(btn2):
    if int(btn2):  # button 2 train
        # msg = 'Button 2 was most recently clicked'
        print('Button 2 was most recently clicked')
        # prueba = "botoncito2"
        time.sleep(1)
        from Prueba import encode_faces # calls the encoding when button train is pressed
        msg = 'Training has finished!'
        return html.Div([
            # html.Img(src="/video_feed")
            # html.Div('btn1: {}'.format(btn1)),
            # html.Div('btn2: {}'.format(btn2)),
            # html.Div('btn3: {}'.format(btn3)),
            html.Div(msg),
            # html.Div(html.Img(src="/video_feed")),
        ])
    # else:
    #     msg = 'None of the buttons have been clicked yet'
    #     return html.Div([])
    # return html.Div([
    #     # html.Img(src="/video_feed")
    #     html.Div('btn1: {}'.format(btn1)),
    #     html.Div('btn2: {}'.format(btn2)),
    #     html.Div('btn3: {}'.format(btn3)),
    #     html.Div(msg),
    #     html.Div(html.Img(src="/video_feed")),
    # ])

if __name__ == '__main__':
    app.run_server(debug=True)
