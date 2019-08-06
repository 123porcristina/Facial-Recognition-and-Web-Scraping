# import dash
# import dash_core_components as dcc
# import dash_html_components as html
#
# from flask import Flask, Response
# import cv2
#
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
#
#
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
# server = Flask(__name__)
# app = dash.Dash(__name__, server=server)
#
# @server.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
# app.layout = html.Div([
#     html.H1("Webcam Test"),
#     html.Img(src="/video_feed")
# ])
#
# if __name__ == '__main__':
#     app.run_server(debug=True)
#
# # # App Layout
# # app.layout = html.Div(
# #     children=[
# #         # Top Banner
# #         html.Div(
# #             className="study-browser-banner row",
# #             children=[
# #                 html.H2(className="h2-title", children="FACE RECOGNITION & WEB SCRAPPING"),
# #             ],
# #         ),
# #
# #         # Body of the App
# #         html.Div(
# #             className="row app-body",
# #             children=[
# #                 # User Controls
# #                 html.Div(
# #                     className="four columns card",
# #                     children=[
# #                         html.Div(
# #                             className="bg-white user-control",
# #                             children=[
# #                                 html.Div(
# #                                     className="padding-top-bot",
# #                                     children=[
# #                                         html.H6("Test Articles"),
# #                                         dcc.Dropdown(id="study-dropdown"),
# #                                     ],
# #                                 ),
# #                                 html.Div( #type of plot
# #                                     className="padding-top-bot",
# #                                     children=[
# #                                         html.H6("Choose the type of plot"),
# #                                         dcc.RadioItems(
# #                                             id="chart-type",
# #                                             options=[
# #                                                 {"label": "Box Plot", "value": "box"},
# #                                                 {
# #                                                     "label": "Violin Plot",
# #                                                     "value": "violin",
# #                                                 },
# #                                             ],
# #                                             value="violin",
# #                                             labelStyle={
# #                                                 "display": "inline-block",
# #                                                 "padding": "12px 12px 12px 0px",
# #                                             },
# #                                         ),
# #                                     ],
# #                                 ),
# #                                 html.Div( #select csv
# #                                     className="padding-top-bot",
# #                                     children=[
# #                                         html.H6("CSV File"),
# #                                         dcc.Upload(
# #                                             id="upload-data",
# #                                             className="upload",
# #                                             children=html.Div(
# #                                                 children=[
# #                                                     html.P("Drag and Drop or "),
# #                                                     html.A("Select Files"),
# #                                                 ]
# #                                             ),
# #                                             accept=".csv",
# #                                         ),
# #                                     ],
# #                                 ),
# #                             ],
# #                         )
# #                     ],
# #                 ),
# #                 # Graph
# #                 html.Div(
# #                     className="eight columns card-left",
# #                     children=[
# #                         html.Div(
# #                             className="bg-white",
# #                             children=[
# #                                 html.H5("Recognition"),
# #                                 html.Img(src="/video_feed")
# #                                 # dcc.Graph(id="plot"),
# #                             ],
# #                         )
# #                     ],
# #                 ),
# #                 # dcc.Store(id="error", storage_type="memory"),
# #             ],
# #         ),
# #     ]
# # )
# #
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ##el otro codigo por si laca
#
#
# # import the necessary packages
# from imutils.video import VideoStream
# import face_recognition
# import argparse
# import imutils
# import pickle
# import time
# import cv2
# # import Insta_Info_Scraper as scraper
# from Prueba import Insta_Info_Scraper as scraper
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
#
# from flask import Flask, Response
# import cv2
#
# import pandas as pd
# import os
# import dash_daq as daq
# import numpy as np
#
#
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-e", "--encodings", required=False,
#                 help="path to serialized db of facial encodings")
# ap.add_argument("-o", "--output", type=str,
#                 help="path to output video")
# ap.add_argument("-y", "--display", type=int, default=1,
#                 help="whether or not to display output frame to screen")
# ap.add_argument("-d", "--detection-method", type=str, default="hog",
#                 help="face detection model to use: either `hog` or `cnn`")
# args = vars(ap.parse_args())
#
# # load the known faces and embeddings
# print("[INFO] loading encodings...")
# data = pickle.loads(open("encodings.pickle", "rb").read())
#
# # scraper
# font = cv2.FONT_HERSHEY_SIMPLEX
# color = (255, 255, 255)
# stroke = 1
# size = 0.5
# obj = scraper.Insta_Info_Scraper(font, color, stroke, size)
#
# # initialize the video stream and pointer to output video file, then
# # allow the camera sensor to warm up
# print("[INFO] starting video stream...")
# # vs = VideoStream(src=0).start()
#
# writer = None
#
#
# # time.sleep(2.0)
#
#
# class VideoCamera(object):
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, frame = self.video.read()
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         #####################################################
#         # convert the input frame from BGR to RGB then resize it to have
#         # a width of 750px (to speedup processing)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         rgb = imutils.resize(frame, width=750)
#         r = frame.shape[1] / float(rgb.shape[1])
#
#         # detect the (x, y)-coordinates of the bounding boxes
#         # corresponding to each face in the input frame, then compute
#         # the facial embeddings for each face
#         boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
#         encodings = face_recognition.face_encodings(rgb, boxes)
#         names = []
#
#         # loop over the facial embeddings
#         for encoding in encodings:
#             # attempt to match each face in the input image to our known
#             # encodings
#             matches = face_recognition.compare_faces(data["encodings"], encoding)
#             name = "Unknown"
#
#             # check to see if we have found a match
#             if True in matches:
#                 # find the indexes of all matched faces then initialize a
#                 # dictionary to count the total number of times each face
#                 # was matched
#                 matchedIdxs = [i for (i, b) in enumerate(matches) if b]
#                 counts = {}
#
#                 # loop over the matched indexes and maintain a count for
#                 # each recognized face face
#                 for i in matchedIdxs:
#                     name = data["names"][i]
#                     counts[name] = counts.get(name, 0) + 1
#
#                 # determine the recognized face with the largest number
#                 # of votes (note: in the event of an unlikely tie Python
#                 # will select first entry in the dictionary)
#                 name = max(counts, key=counts.get)
#             # update the list of names
#             names.append(name)
#
#         # loop over the recognized faces
#         for ((top, right, bottom, left), name) in zip(boxes, names):
#             # rescale the face coordinates
#             top = int(top * r)
#             right = int(right * r)
#             bottom = int(bottom * r)
#             left = int(left * r)
#
#             # draw rectangle
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             # saved = "images/face" + str(i) + ".jpg"
#             # cv2.imwrite(saved, frame)
#
#             # draw the predicted face name and instagram status on the image
#             # if username is recognized  from the camera, save the url in a text file
#             # to be pulled out later by a scraper
#             open('users.txt', 'w').close()  # clear5 it first
#             file1 = open("users.txt", "a")  # append mode
#             file1.write("https://www.instagram.com/" + name + "/")
#             file1.close()
#             obj.main(frame, top, left, right, bottom, name)
#
#         ###############################
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()
#
#
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if args["display"] > 0:
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#
# def header_colors():
#     return {
#         'bg_color': '#e7625f',
#         'font_color': 'white',
#         'textAlign': 'center',
#         'backgroundColor': '#C50063',
#         'color': 'white',
#     }
#
#
# DATAPATH = os.path.join('.', 'tests', 'dashbio_demos', 'sample_data', 'volcano_')
#
# DATASETS = {
#     'SET1': {
#         'label': 'Set1',
#         'dataframe': None,
#         'datafile': '{}data1.csv'.format(DATAPATH),
#         'datasource': 'ftp://ftp.ncbi.nlm.nih.gov/hapmap/genotypes/'
#                       '2009-01_phaseIII/plink_format/',
#         'dataprops': {}
#     },
#     'SET2': {
#         'label': 'Set2',
#         'dataframe': None,
#         'datafile': '{}data2.csv'.format(DATAPATH),
#         'datasource': 'https://doi.org/10.1371/journal.pntd.0001039.s001',
#         'dataprops': dict(
#             effect_size='log2_(L3i/L1)_signal_ratio',
#             p='p-value',
#             snp=None,
#             gene='PFAM_database_id',
#             annotation='annotation'
#         )
#     }
# }
#
# for dataset in DATASETS:
#     DATASETS[dataset]['dataframe'] = ""#pd.read_csv(
#         # DATASETS[dataset]['datafile'], comment='#')
#
#
# server = Flask(__name__)
# app = dash.Dash(__name__, server=server)
#
#
#
# @server.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# # app.layout = html.Div([
# #     html.H1("Webcam Test"),
# #     html.Img(src="/video_feed")
# # ])
#
# # external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# # # external_stylesheets = ['https://github.com/plotly/dash-bio/blob/master/assets/volcanoplot-style.css']
# # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
#
#
# # app.layout = html.Div([
# #     html.Div([
# #         html.Div(children=[
# #             html.H1(children='Facial Recognition and web scrapping',
# #                     style={'textAlign': 'center', 'color': 'white', 'backgroundColor': '#C50063'},
# #                     className="twelve columns"),  # title occupies 9 cols
# #
# #             html.Div(children='''
# #                         Dash: Webcam Test - prueba
# #                         ''',
# #                      className="nine columns")  # this subtitle occupies 9 columns
# #         ], className="row"),
#
# # app.layout =  html.Div([
# #     html.Div([
# #         html.Div(children=[
# #             html.H1(children='Facial Recognition and web scrapping',
# #                     style=header_colors(),#{'textAlign': 'center', 'color': 'white', 'backgroundColor': '#C50063'},
# #                     className="app-page-header"),  # title occupies 9 cols
# #
# #             # html.Div(children='''
# #             #             Dash: Webcam Test - prueba
# #             #             ''',
# #             #          className="nine columns")  # this subtitle occupies 9 columns
# #         ], className="app-page-header"),
# #
# #         html.Div(id='vp-page-content', className='app-body', children=[
# #             dcc.Loading(className='dashbio-loading', children=html.Div(
# #                 id='vp-graph-div',
# #                 children=dcc.Graph(
# #                     id='vp-graph'
# #                 ),
# #             )),
# #
# #
# #         html.Div(id='vp-control-tabs', className='control-tabs', children=[
# #             dcc.Tabs(id='vp-tabs', value='what-is', children=[
# #                 dcc.Tab(
# #                     label='About',
# #                     value='what-is',
# #                     children=html.Div(className='control-tab', children=[
# #                         html.H4(className='what-is', children='What is Volcano Plot?'),
# #                         html.P(
# #                             'You can use Volcano Plot to interactively '
# #                             'identify clinically meaningful markers in '
# #                             'genomic experiments, i.e., markers that are '
# #                             'statistically significant and have an effect '
# #                             'size greater than some threshold. '
# #                             'Specifically, volcano plots depict the negative '
# #                             'log-base-10 p-values plotted against their '
# #                             'effect size.'
# #                         ),
# #                         html.P(
# #                             'In the "Data" tab, you can select a dataset '
# #                             'to view on the plot. In the "View" tab, you '
# #                             'can control the color of the highlighted '
# #                             'points, as well as the threshold lines that '
# #                             'define which values are significant. You can '
# #                             'also access metadata from hovering and '
# #                             'clicking on the graph.'
# #                         )
# #                     ])
# #                 ),
# #                 dcc.Tab(
# #                     label='Data',
# #                     value='data',
# #                     children=html.Div(className='control-tab', children=[
# #                         html.Div(className='app-controls-block', children=[
# #                             html.Div(
# #                                 className='app-controls-name',
# #                                 children='Dataset: '
# #                             ),
# #                             dcc.Dropdown(
# #                                 id='vp-dataset-dropdown',
# #                                 options=[
# #                                     {
# #                                         'label': DATASETS[dset]['label'],
# #                                         'value': dset
# #                                     }
# #                                     for dset in DATASETS
# #                                 ],
# #                                 value='SET2'
# #                             )
# #                         ])
# #                     ])
# #                 ),
# #                 dcc.Tab(
# #                     label='View',
# #                     value='view', children=html.Div(className='control-tab', children=[
# #                         html.Div(className='app-controls-block', children=[
# #                             html.Div(
# #                                 className='app-controls-name',
# #                                 children='Effect size bounds'
# #                             ),
# #                             dcc.RangeSlider(
# #                                 id='vp-bound-val',
# #                                 min=-4,
# #                                 max=4,
# #                                 value=[-1, 1],
# #                                 step=0.01,
# #                                 marks={str(num): str(num) for num in range(-4, 5)}
# #                             )
# #                         ]),
# #                         html.Div(className='app-controls-block', children=[
# #                             html.Div(
# #                                 className='app-controls-name',
# #                                 children='Threshold',
# #                             ),
# #                             dcc.Slider(
# #                                 id='vp-genomic-line-val',
# #                                 value=4,
# #                                 max=10,
# #                                 min=0,
# #                                 step=0.01,
# #                                 marks={str(num): str(num) for num in range(0, 11, 2)}
# #                             ),
# #                         ]),
# #                         html.Div(className='app-controls-block', children=[
# #                             daq.ColorPicker(
# #                                 id='vp-color-picker',
# #                                 value=dict(hex="#0000FF"),
# #                                 size=150,
# #                             ),
# #                             html.Div(
# #                                 id='vp-num-points-display',
# #                                 children=[
# #                                     html.Div(
# #                                         title='Number of points in the upper left',
# #                                         children=[
# #                                             daq.LEDDisplay(
# #                                                 className='vp-input-like',
# #                                                 label='Upper left points',
# #                                                 id='vp-upper-left',
# #                                                 size=25,
# #                                                 color='#19D3F3'
# #                                             ),
# #                                             html.Div(
# #                                                 className='vp-test-util-div',
# #                                                 id='vp-upper-left-val'
# #                                             )
# #                                         ]
# #                                     ),
# #                                     html.Br(),
# #                                     html.Div(
# #                                         className='vp-vertical-style',
# #                                         title='Number of points in the upper right',
# #                                         children=[
# #                                             daq.LEDDisplay(
# #                                                 className='vp-input-like',
# #                                                 label='Upper right points',
# #                                                 id='vp-upper-right',
# #                                                 size=25,
# #                                                 color='#19D3F3'
# #                                             ),
# #                                             html.Div(
# #                                                 className='vp-test-util-div',
# #                                                 id='vp-upper-right-val'
# #                                             )
# #                                         ]
# #                                     ),
# #                                 ],
# #                             )
# #                         ]),
# #                         html.Hr(),
# #                         html.Div(id='vp-event-data')
# #                     ])
# #                 )
# #             ])
# #         ])
# #     ])
# #
# #
# #
# #
# #
# #
# # ])
# # ])
# #
# #
# #
# #
# #
# # #         html.Div([
# # #             # html.Div(dcc.Input(id='input-box', type='text')),
# # #             html.Button("Capture", id='button', className="button_instruction"),
# # #             html.Button("Train", className="demo_button", id="demo"),
# # #             # html.Div(id='output-container-button',
# # #             #          children='Enter a value and press submit')
# # #         ]),
# # #
# # #         html.Div([
# # #             # dcc.Upload(
# # #             #     id='upload-data',
# # #             #     children=html.Div([
# # #             #         'Drag and Drop or ',
# # #             #         html.A('Select Files')
# # #             #     ]),
# # #             #     style={
# # #             #         'width': '100%',
# # #             #         'height': '60px',
# # #             #         'lineHeight': '60px',
# # #             #         'borderWidth': '1px',
# # #             #         'borderStyle': 'dashed',
# # #             #         'borderRadius': '5px',
# # #             #         'textAlign': 'center',
# # #             #         'margin': '10px'
# # #             #     },
# # #             #     # Allow multiple files to be uploaded
# # #             #     multiple=True
# # #             # ),
# # #             # html.Div([
# # #             #     html.Div(id='output-data-upload'),
# # #             #     # html.Div(id='output-data-info'),
# # #             # ], className="row")
# # #             html.Img(src="/video_feed")
# # #         ], className='row'),
# # #     ])
# # # ])
# #
# # def callbacks(app):  # pylint: disable=redefined-outer-name
# #     @app.callback(
# #         Output('vp-event-data', 'children'),
# #         [Input('vp-graph', 'hoverData'),
# #          Input('vp-graph', 'clickData')]
# #     )
# #     def get_event_data(hover, click):
# #         hover_data_div = [
# #             html.Div(className='app-controls-name', children='Hover data')
# #         ]
# #         hover_data = 'Hover over a data point to see it here.'
# #
# #         if hover is not None:
# #             hovered_point = hover['points'][0]
# #             hovered_text = hovered_point['text'].strip('<br>').split('<br>')
# #
# #             hover_data = [
# #                 'x: {}'.format('{0:.3f}'.format(hovered_point['x'])),
# #                 html.Br(),
# #                 'y: {}'.format('{0:.3f}'.format(hovered_point['y'])),
# #                 html.Br(),
# #                 '{} ({})'.format(hovered_text[0], hovered_text[1])
# #             ]
# #
# #         hover_data_div.append(
# #             html.Div(className='vp-event-data-display', children=hover_data)
# #         )
# #
# #         click_data_div = [
# #             html.Div(className='app-controls-name', children='Click data')
# #         ]
# #         click_data = 'Click on a data point to see it here.'
# #
# #         if click is not None:
# #             clicked_point = click['points'][0]
# #             clicked_text = clicked_point['text'].strip('<br>').split('<br>')
# #
# #             click_data = [
# #                 'x: {}'.format('{0:.3f}'.format(clicked_point['x'])),
# #                 html.Br(),
# #                 'y: {}'.format('{0:.3f}'.format(clicked_point['y'])),
# #                 html.Br(),
# #                 '{} ({})'.format(clicked_text[0], clicked_text[1])
# #             ]
# #
# #         click_data_div.append(
# #             html.Div(className='vp-event-data-display', children=click_data)
# #         )
# #
# #         return html.Div([
# #             html.Div(hover_data_div),
# #             html.Div(click_data_div)
# #         ])
# #
# #     @app.callback(
# #         Output('vp-graph', 'figure'),
# #         [
# #             Input('vp-bound-val', 'value'),
# #             Input('vp-genomic-line-val', 'value'),
# #             Input('vp-dataset-dropdown', 'value'),
# #             Input('vp-color-picker', 'value')
# #         ]
# #     )
# #     def update_graph(effect_lims, genomic_line, datadset_id, color):
# #         """Update rendering of data points upon changing x-value of vertical dashed lines."""
# #         l_lim = effect_lims[0]
# #         u_lim = effect_lims[1]
# #         if 'hex' in color:
# #             color = color.get('hex', 'red')
# #         # return dash_bio.VolcanoPlot(
# #         #     DATASETS[datadset_id]['dataframe'],
# #         #     genomewideline_value=float(genomic_line),
# #         #     effect_size_line=[float(l_lim), float(u_lim)],
# #         #     highlight_color=color,
# #         #     **DATASETS[datadset_id]['dataprops']
# #         # )
# #
# #     @app.callback(
# #         Output('vp-dataset-div', 'title'),
# #         [
# #             Input('vp-dataset-dropdown', 'value')
# #         ]
# #     )
# #     def update_vp_dataset_div_hover(dataset_id):
# #         """Update the dataset of interest."""
# #         return DATASETS[dataset_id]['datasource']
# #
# #     @app.callback(
# #         Output('vp-upper-right', 'value'),
# #         [Input('vp-graph', 'figure')],
# #         [State('vp-bound-val', 'value')]
# #     )
# #     def update_upper_right_number(fig, bounds):
# #         """Update the number of data points in the upper right corner."""
# #         u_lim = bounds[1]
# #         number = 0
# #         if len(fig['data']) > 1:
# #             x = np.array(fig['data'][0]['x'])
# #             idx = x > float(u_lim)
# #             number = len(x[idx])
# #         return number
# #
# #     @app.callback(
# #         Output('vp-upper-left', 'value'),
# #         [Input('vp-graph', 'figure')],
# #         [State('vp-bound-val', 'value')]
# #     )
# #     def update_upper_left_number(fig, bounds):
# #         """Update the number of data points in the upper left corner."""
# #         l_lim = bounds[0]
# #         number = 0
# #         if len(fig['data']) > 1:
# #             x = np.array(fig['data'][0]['x'])
# #             idx = x < float(l_lim)
# #             number = len(x[idx])
# #         return number
# #
# #     # Callbacks for integration test purposes
# #     @app.callback(
# #         Output('vp-upper-left-val', 'children'),
# #         [Input('vp-graph', 'figure')],
# #         [State('vp-bound-val', 'value')]
# #     )
# #     def update_upper_left_number_val(fig, bounds):
# #         """Update the number of data points in the upper left corner
# #         for testing purposes.
# #         """
# #         l_lim = bounds[0]
# #         number = 0
# #         if len(fig['data']) > 1:
# #             x = np.array(fig['data'][0]['x'])
# #             idx = x < float(l_lim)
# #             number = len(x[idx])
# #         return str(number)
# #
# #     @app.callback(
# #         Output('vp-upper-bound-val', 'value'),
# #         [Input('vp-bound-val', 'value')],
# #     )
# #     def update_upper_bound_val(bounds):
# #         """For selenium tests."""
# #         return bounds[1]
# #
# #     @app.callback(
# #         Output('vp-upper-right-val', 'children'),
# #         [Input('vp-graph', 'figure')],
# #         [State('vp-bound-val', 'value')]
# #     )
# #     def update_upper_right_number_val(fig, bounds):
# #         """Update the number of data points in the upper right corner
# #         for testing purposes.
# #         """
# #         u_lim = bounds[1]
# #         number = 0
# #         if len(fig['data']) > 1:
# #             x = np.array(fig['data'][0]['x'])
# #             idx = x > float(u_lim)
# #             number = len(x[idx])
# #         return str(number)
# #
# #     @app.callback(
# #         Output('vp-lower-bound-val', 'value'),
# #         [Input('vp-bound-val', 'value')],
# #     )
# #     def update_lower_bound_val(bounds):
# #         """For selenium tests."""
# #         l_lim = bounds[0]
# #         return l_lim
# #
# #     @app.callback(
# #         Output('vp-genomic-line-val', 'value'),
# #         [Input('vp-genomic-line', 'value')],
# #     )
# #     def update_genomic_line_val(val):
# #         """For selenium tests purpose."""
# #
# #         # A value of 0 is forbidden by the component
# #         if val == 0:
# #             val = 1e-9
# #         return val
#
#
#
# #
# # ######################
# # # App Layout
# # app.layout = html.Div(
# #     children=[
# #         # Top Banner
# #         html.Div(
# #             className="study-browser-banner row",
# #             children=[
# #                 html.H2(className="h2-title", children="FACE RECOGNITION & WEB SCRAPPING"),
# #             ],
# #         ),
# #
# #         # Body of the App
# #         html.Div(
# #             className="row app-body",
# #             children=[
# #                 # User Controls
# #                 html.Div(
# #                     className="four columns card",
# #                     children=[
# #                         html.Div(
# #                             className="bg-white user-control",
# #                             children=[
# #                                 html.Div(id='vp-control-tabs', className='control-tabs', children=[
# #                                                     dcc.Tabs(id='vp-tabs', value='what-is', children=[
# #                                                         dcc.Tab(
# #                                                             label='About',
# #                                                             value='what-is',
# #                                                             children=html.Div(className='control-tab', children=[
# #                                                                 html.Br(),
# #                                                                 html.H4(className='what-is', children='What is face recognition?'),
# #                                                                 html.P(
# #
# #                                                                     'This project was created by Cristina Giraldo '
# #                                                                     'and Gregg Legarda. We created this project because'
# #                                                                     'we have curiosity about how computer vision works '
# #                                                                     'especially in the facial recognition area. '
# #
# #                                                                 ),
# #                                                                 html.P(
# #
# #                                                                     'In the "Data" tab, you can take a picture '
# #                                                                     'train the algorithm, and of course start the video '
# #                                                                     'to see how facial recognition works.'
# #
# #                                                                 ),
# #                                                                 html.P(
# #
# #                                                                     'The video takes some time to initialize so please '
# #                                                                     'be patient. '
# #                                                                 )
# #                                                             ])
# #                                                         ),
# #                                                         dcc.Tab(
# #                                                             label='Data',
# #                                                             value='data',
# #                                                             children=html.Div(className='control-tab', children=[
# #                                                                 html.Div(className='app-controls-block', children=[
# #                                                                     html.Div( className='app-controls-name',
# #                                                                               children='Dataset: '
# #                                                                               ),
# #
# #
# #                                                                     # html.Div(
# #                                                                     #     [
# #                                                                     #         html.Button(
# #                                                                     #             "CAPTURE",
# #                                                                     #             className="button_instruction",
# #                                                                     #             id="learn-more-button",
# #                                                                     #         ),
# #                                                                     #         html.Button(
# #                                                                     #             "TRAIN ALGORITHM",
# #                                                                     #             className="demo_button", id="demo"
# #                                                                     #         ),
# #                                                                     #         html.Br(),
# #                                                                     #         html.Br(),
# #                                                                     #         html.Button(
# #                                                                     #             "START RECOGNITION",
# #                                                                     #             className="button_submit", id="start"
# #                                                                     #         ),
# #                                                                     #
# #                                                                     #
# #                                                                     #
# #                                                                     #     ],
# #                                                                     #     className="mobile_buttons",
# #                                                                     # ),
# #
# #                                                                     html.Div([
# #                                                                         html.Button('Capture Picture', id='btn-1', n_clicks_timestamp=0),
# #                                                                         html.Button('Train  Algorithm', id='btn-2', n_clicks_timestamp=0),
# #                                                                         html.Button('Facial Recognition', id='btn-3', n_clicks_timestamp=0),
# #                                                                         html.Button('Button 4', id='btn-4', n_clicks_timestamp=0 ),
# #                                                                         html.Div(id='container-button-timestamp')
# #                                                                     ],className="mobile_buttons",)
# #
# #
# #                                                                     # html.Br(),
# #                                                                     # html.Br(),
# #                                                                     # html.Div(
# #                                                                     #     [
# #                                                                     #         # html.Button(
# #                                                                     #         #     "CAPTURE",
# #                                                                     #         #     className="button_instruction",
# #                                                                     #         #     id="learn-more-button",
# #                                                                     #         # ),
# #                                                                     #         # html.Button(
# #                                                                     #         #     "TRAIN ALGORITHM",
# #                                                                     #         #     className="demo_button", id="demo"
# #                                                                     #         # ),
# #                                                                     #         html.Button(
# #                                                                     #             "START RECOGNITION",
# #                                                                     #             className="button_submit", id="start"
# #                                                                     #         ),
# #                                                                     #
# #                                                                     #     ],
# #                                                                     #     className="mobile_buttons",
# #                                                                     # ),
# #
# #
# #
# #                                                                     # html.Button("Capture", id='button', className="button_instruction"),
# #                                                                     # html.Button("Train", className="demo_button", id="demo"),
# #                                                                     # html.Button("Start", id='button2', className="start"),
# #
# #                                                                     # html.A(
# #                                                                     #     html.Button(
# #                                                                     #         "Download sample structure",
# #                                                                     #         id="mol3d-download-sample-data",
# #                                                                     #         className='control-download'
# #                                                                     #     ),
# #                                                                     #     # href=os.path.join('assets', 'sample_data',
# #                                                                     #     #                   'molecule3d_2mru.pdb'),
# #                                                                     #     # download='2mru.pdb'
# #                                                                     # )
# #                                                                 ])
# #                                                             ])
# #                                                         ),
# #                                                         ]),
# #                                                     ]),
# #
# #
# #
# #
# #
# #                                 # html.Div(
# #                                 #     className="padding-top-bot",
# #                                 #     children=[
# #                                 #         html.H6("Test Articles"),
# #                                 #         dcc.Dropdown(id="study-dropdown"),
# #                                 #     ],
# #                                 # ),
# #                                 # html.Div( #type of plot
# #                                 #     className="padding-top-bot",
# #                                 #     children=[
# #                                 #         html.H6("Choose the type of plot"),
# #                                 #         dcc.RadioItems(
# #                                 #             id="chart-type",
# #                                 #             options=[
# #                                 #                 {"label": "Box Plot", "value": "box"},
# #                                 #                 {
# #                                 #                     "label": "Violin Plot",
# #                                 #                     "value": "violin",
# #                                 #                 },
# #                                 #             ],
# #                                 #             value="violin",
# #                                 #             labelStyle={
# #                                 #                 "display": "inline-block",
# #                                 #                 "padding": "12px 12px 12px 0px",
# #                                 #             },
# #                                 #         ),
# #                                 #     ],
# #                                 # ),
# #                                 # html.Div( #select csv
# #                                 #     className="padding-top-bot",
# #                                 #     children=[
# #                                 #         html.H6("CSV File"),
# #                                 #         dcc.Upload(
# #                                 #             id="upload-data",
# #                                 #             className="upload",
# #                                 #             children=html.Div(
# #                                 #                 children=[
# #                                 #                     html.P("Drag and Drop or "),
# #                                 #                     html.A("Select Files"),
# #                                 #                 ]
# #                                 #             ),
# #                                 #             accept=".csv",
# #                                 #         ),
# #                                 #     ],
# #                                 # ),
# #                             ],
# #                         )
# #                     ],
# #                 ),
# #                 # Graph
# #                 html.Div(
# #                     className="eight columns card-left",
# #                     children=[
# #                         html.Div(
# #                             className="bg-white",
# #                             children=[
# #                                 html.H5("Recognition"),
# #                                 html.Img(src="/video_feed")
# #                                 # dcc.Graph(id="plot"),
# #                             ],
# #                         )
# #                     ],
# #                 ),
# #                 # dcc.Store(id="error", storage_type="memory"),
# #             ],
# #         ),
# #     ]
# # )
# #
# #
# #
# # if __name__ == '__main__':
# #     # header_colors()
# #     app.run_server(debug=False)
#
#


# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import datetime
import json

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dcc.Tab(
                    label='Data',
                    value='data',
                    children=html.Div(className='control-tab', children=[
                        html.Div(className='app-controls-block', children=[
                            html.Div(className='app-controls-name',
                                     children='Dataset: '
                                     ),

                            html.Div([
                                html.Button('Capture Picture', id='btn-1',
                                            n_clicks_timestamp=0),
                                html.Button('Train  Algorithm', id='btn-2',
                                            n_clicks_timestamp=0),
                                html.Button('Facial Recognition', id='btn-3',
                                            n_clicks_timestamp=0),
                                html.Button('Button 4', id='btn-4', n_clicks_timestamp=0),
                                html.Div(id='container-button-timestamp')
                            ],
                                className="mobile_buttons", )

                        ])
                    ])
                )

# app.layout = html.Div([
#     html.Button('Button 1', id='btn-1', n_clicks_timestamp=0),
#     html.Button('Button 2', id='btn-2', n_clicks_timestamp=0),
#     html.Button('Button 3', id='btn-3', n_clicks_timestamp=0),
#     html.Div(id='container-button-timestamp')
# ])

@app.callback(Output('container-button-timestamp', 'children'),
              [Input('btn-1', 'n_clicks_timestamp'),
               Input('btn-2', 'n_clicks_timestamp'),
               Input('btn-3', 'n_clicks_timestamp')])
def displayClick(btn1, btn2, btn3):
    if int(btn1) > int(btn2) and int(btn1) > int(btn3):
        msg = 'Button 1 was most recently clicked'
    elif int(btn2) > int(btn1) and int(btn2) > int(btn3):
        msg = 'Button 2 was most recently clicked'
    elif int(btn3) > int(btn1) and int(btn3) > int(btn2):
        msg = 'Button 3 was most recently clicked'
    else:
        msg = 'None of the buttons have been clicked yet'
    return html.Div([
        html.Div('btn1: {}'.format(btn1)),
        html.Div('btn2: {}'.format(btn2)),
        html.Div('btn3: {}'.format(btn3)),
        html.Div(msg)
    ])

if __name__ == '__main__':
    app.run_server(debug=True)