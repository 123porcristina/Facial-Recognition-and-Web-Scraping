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

from flask import Flask, Response
import cv2


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
#vs = VideoStream(src=0).start()

writer = None
# time.sleep(2.0)







class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

#####################################################
        # convert the input frame from BGR to RGB then resize it to have
	# a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])



        
        
        # detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        
        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
	    # encodings
            matches = face_recognition.compare_faces(data["encodings"],encoding)
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
            #update the list of names
            names.append(name)
            
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):        
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            #draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
            #saved = "images/face" + str(i) + ".jpg"
            #cv2.imwrite(saved, frame)
            
            # draw the predicted face name and instagram status on the image
            # if username is recognized  from the camera, save the url in a text file
            # to be pulled out later by a scraper
            open('users.txt', 'w').close() #clear5 it first
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

def header_colors():
    return {
        'bg_color': '#e7625f',
        'font_color': 'white'
    }



server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# app.layout = html.Div([
#     html.H1("Webcam Test"),
#     html.Img(src="/video_feed")
# ])

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



app.layout = html.Div([
    html.Div([
        html.Div(children=[
            html.H1(children='Facial Recognition and web scrapping',
                    style={'textAlign': 'center', 'color': 'white', 'backgroundColor':'#C50063'},
                    className= "twelve columns"), #title occupies 9 cols

            html.Div(children=''' 
                        Dash: Webcam Test - prueba
                        ''',
                     className="nine columns")#this subtitle occupies 9 columns
        ], className = "row"),

        html.Div([
            # html.Div(dcc.Input(id='input-box', type='text')),
            html.Button("Capture", id='button', className="button_instruction"),
            html.Button("Train", className="demo_button", id="demo"),
            # html.Div(id='output-container-button',
            #          children='Enter a value and press submit')
        ]),

        html.Div([
            # dcc.Upload(
            #     id='upload-data',
            #     children=html.Div([
            #         'Drag and Drop or ',
            #         html.A('Select Files')
            #     ]),
            #     style={
            #         'width': '100%',
            #         'height': '60px',
            #         'lineHeight': '60px',
            #         'borderWidth': '1px',
            #         'borderStyle': 'dashed',
            #         'borderRadius': '5px',
            #         'textAlign': 'center',
            #         'margin': '10px'
            #     },
            #     # Allow multiple files to be uploaded
            #     multiple=True
            # ),
            # html.Div([
            #     html.Div(id='output-data-upload'),
            #     # html.Div(id='output-data-info'),
            # ], className="row")
            html.Img(src="/video_feed")
        ], className='row'),
    ])
])


if __name__ == '__main__':
    app.run_server(debug=False)
