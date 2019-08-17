# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
#from Code import Insta_Info_Scraper as scraper
import Insta_Info_Scraper as scraper
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from haar.faces import VideoCamera2
from gradient.gradient import VideoCamera3
from haar import faces

from flask import Flask, Response
import os
from imutils.video import videostream

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

"""load the known faces and embeddings"""
print("[INFO] loading encodings...")
data = pickle.loads(open("encodings.pickle", "rb").read())

"""scraper"""
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)
stroke = 1
size = 0.4
obj = scraper.Insta_Info_Scraper(font, color, stroke, size)

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
writer = None

"""allows to turn of the light of the cam"""
os.getenv("OPENCV_VIDEOIO_PRIORITY_MSMF", None)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        print("[INFO] starting video stream...")

    def __del__(self):
        print("DEL fue ejecutado")
        self.video.release()

    # def release(self):
    #     print("DEL fue ejecutado")
    #     self.video.release()
    #     cv2.destroyAllWindows()


    def get_frame(self):
        success, frame = self.video.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #####################################################
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
            # encodings
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

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

            # draw the predicted face name and instagram status on the images
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
                   b'Content-Type: images/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
algorithm = VideoCamera()

def algotype():
    return algorithm 
    


server = Flask(__name__)
# app = dash.Dash(__name__, server=server,  external_stylesheets=[dbc.themes.BOOTSTRAP])
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(algotype()),
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

        ############################################################

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
                                html.Button('Capture Image', id='btn-1', className="control-download",
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Train  Algorithm', id='btn-2',
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('HOG Webcam', id='btn-3',
                                            n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Stop Webcam', id='btn-4',  n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('HAAR Webcam', id='btn-5',  n_clicks_timestamp=0),
                                html.Br(),
                                html.Button('Gradient', id='btn-6',  n_clicks_timestamp=0),

                                # html.Div(id='container-button-timestamp')

                            ]),
                        ]),
                        html.Hr(),
                    ])
                ),

                dcc.Tab(
                    label='Graph',
                    value='graph',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Graph type'),
                            dcc.Dropdown(
                                id='circos-graph-type',
                                options=[
                                    {'label': graph_type.title(),
                                     'value': graph_type} for graph_type in [
                                        'heatmap',
                                        'chords',
                                        'highlight',
                                        'histogram',
                                        'line',
                                        'scatter',
                                        'stack',
                                        'text',
                                        'parser_data'
                                    ]
                                ],
                                value='chords'
                            ),
                            html.Div(className='app-controls-desc', id='chords-text'),
                        ]),
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name', children='Graph size'),
                            dcc.Slider(
                                id='circos-size',
                                min=500,
                                max=800,
                                step=10,
                                value=650
                            ),
                        ]),
                        html.Hr(),
                        html.H5('Hover data'),
                        html.Div(
                            id='event-data-select'
                        ),

                    ]),
                ),
            ])
        ]),
        ############################################################

                # Graph
                html.Div(
                    className="eight columns card-left",
                    children=[
                        html.H5("Recognition"),
                        html.Div(
                            className="bg-white",
                            children=[
                                # html.H5("Recognition"),
                                # dcc.Store(id='memory-output'),
                                html.Div(id='output-video'),
                                dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="default"),
                                # dcc.Graph(id="plot"),
                            ],
                        )
                    ],
                ),
                dcc.Store(id="error", storage_type="memory"),
            # ],
        # ),
    ]
)

# @app.callback(Output('output-video', 'children'),
#               [Input('btn-1', 'n_clicks_timestamp'),
#                Input('btn-2', 'n_clicks_timestamp'),
#                Input('btn-3', 'n_clicks_timestamp')])
# def displayClick(btn1, btn2, btn3):
#     if int(btn1) > int(btn2) and int(btn1) > int(btn3):
#         msg = 'Button 1 was most recently clicked'
#     elif int(btn2) > int(btn1) and int(btn2) > int(btn3):
#         # msg = 'Button 2 was most recently clicked'
#         time.sleep(1)
#         msg = 'Training has finished!'
#         return html.Div([html.Div(msg)])
#     elif int(btn3) > int(btn1) and int(btn3) > int(btn2):
#         msg = 'Button 3 was most recently clicked'
#         return html.Div([html.Div(html.Img(src="/video_feed"))])
#     else:
#         msg = 'None of the buttons have been clicked yet'
#         return html.Div([])
#

# if __name__ == '__main__':
#     app.run_server(debug=True)

image_count = 1


@app.callback(Output('output-video', 'children'),
              [Input('btn-1', 'n_clicks_timestamp'),
               Input('btn-3', 'n_clicks_timestamp'),
               Input('btn-4', 'n_clicks_timestamp'),
               Input('btn-5', 'n_clicks_timestamp'),
               Input('btn-6', 'n_clicks_timestamp')])
def displayClick(btn1, btn3, btn4, btn5, btn6):
    global image_count

    if int(btn1) > int(btn3) and int(btn1) > int(btn4) and int(btn1) > int(btn5) and int(btn1) > int(btn6):   #btn 1 Capture 1mage
        # capture faces and save for training
        print("button CAPTURE was pressed")
        video_capture = cv2.VideoCapture(0)
        time.sleep(2)
        ret, frame = video_capture.read()
        saved = "images/face" + str(image_count) + ".jpg"
        cv2.imwrite(saved, frame)
        image_count = image_count+1
        video_capture.release()
        cv2.destroyAllWindows()
        del video_capture
        return None


    elif int(btn3) > int(btn1) and int(btn3) > int(btn4) and int(btn3) > int(btn5) and int(btn3) > int(btn6):  # button 3 - Recognition HOG
        cv2.destroyAllWindows()
        global algorithm
        algorithm = VideoCamera()#for hog
        return html.Div([ html.Div(html.Img(src="/video_feed"))])

    elif int(btn4) > int(btn1) and int(btn4) > int(btn3) and int(btn4) > int(btn5) and int(btn4) > int(btn6): #btn 4 - stop video
        print("[INFO] Facial recognition has been STOPPED (button4: stop video pressed)")
        cv2.destroyAllWindows()
        return html.Div([html.Div(html.Img(src=" "))])

    elif int(btn5) > int(btn1)  and int(btn5) > int(btn3) and int(btn5) > int(btn4) and int(btn5) > int(btn6):     #btn 5 - HAAR
        cv2.destroyAllWindows()
        #global algorithm
        algorithm = VideoCamera2()#for haar
        return html.Div([ html.Div(html.Img(src="/video_feed"))])
    elif int(btn6) > int(btn1)  and int(btn6) > int(btn3) and int(btn6) > int(btn4) and int(btn6) > int(btn5):   #btn 6 - gradient
        cv2.destroyAllWindows()
        #global algorithm
        algorithm = VideoCamera3()#for haar
        return html.Div([ html.Div(html.Img(src="/video_feed"))])
    else:
        msg = 'None of the buttons have been clicked yet'
        return html.Div([])



@app.callback(Output('loading-output-1', 'children'),
              [Input('btn-2', 'n_clicks_timestamp'),
               Input('btn-6', 'n_clicks_timestamp')])
def displayLoadTrain(btn2,btn6):                                                         
    if  int(btn2) > int(btn6):         # button 2 train
        #from Code import encode_faces
        #from Code.haar import faces_train
        from haar import faces_train
        import encode_faces # calls the encoding on both algorithms when button train is pressed
        msg = 'Training has finished!'
        return html.Div([
            html.Div(msg),
        ])


    else:
        msg = 'None of the buttons have been clicked yet'
        return html.Div([])  



if __name__ == '__main__':
    app.run_server(debug=True)
