"""File: faces_train.py
    def faces_train():
    gx, gy = np.gradient(roi) #take the gradiant to vectorize the images into two values
    roi = np.sqrt(np.square(gx) + np.square(gy)) # get magnitude to normalize it
    x_train.append(roi)
    y_labels.append(id_)



File: Gradient.py
import cv2
import numpy as np
class VideoCamera3(object):
def __init__(self):
self.video = cv2.VideoCapture(0)
print("[INFO] starting video stream......")
def __del__(self):
print("DEL fue ejecutado")
self.video.release()
def get_frame(self):
print("Get frame gradiant")

gr = VideoCamera3()
gr.get_frame()



File: Capture_image.py
import time
import cv2
import os
class CaptureImage(object):
def __init__(self, name, image_count):
self.name = name
self.image_count = image_count
self.path = 'dataset/'+str(self.name)+'/'
def create_dir(self):
if os.path.exists(self.path):
return "[INFO] Already the path exists"
else:
os.mkdir(self.path)
return "[INFO] Directory ", self.name, " Created"


File: encode_faces.py
ap.add_argument("-d", "--detection-method", type=str, default="hog",
help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())
def encoding():
imagePaths = list(paths.list_images("dataset"))

Total: 40
modify:10
Created:0
Ours:10

File: Insta_Info_Scraper.py
class Insta_Info_Scraper:
def __init__(self, font, color, stroke, size):
self.font = font
self.color = color
self.stroke = stroke
self.size = size
self.insta_dict = {}
# Verify if the info already exists in the dictionary
def check_info(self, user):
return user in self.insta_dict
# Get info from the dictionary according to the user
def getinfo_dict(self, name):
# if the user exists returns the dictonary correspondent to it
if name not in self.insta_dict.values():
return self.insta_dict[name]
else:
print("user not found")
return "User not found"
# Get info by doing a request. In this case we are using Instagram.
# This is called everytime is a new face on the screen
def getinfo(self, url, name):
print("getinfo error 1 = url is", url, "name is", name)
if name == "Unknown":
# add new info to the dictionary
info_instagram = {name: {'User': name,
'Followers': name,
'Following': name,
'Posts': name}}
self.insta_dict.update(info_instagram)
print('User: Unknown')
print('---------------------------')
else:
# add new info to the dictionary
info_instagram = {name: {'User': user,
'Followers': followers,
'Following': following,
'Posts': posts}}
self.insta_dict.update(info_instagram)
print('User:', user)

File app.py

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

"""



