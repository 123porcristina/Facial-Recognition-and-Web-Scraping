# Facial Recognition and Web Scraping
Packages to import:
•    Requirements for imports are OpenCV, Pillow, Pickle, Os, numpy glib, requests


Before to execute this project it is necessary to install:

- install cmake
- pip install dlib
- pip install face_recognition
- pip install opencv-python

To visualize the application is necessary install core dash backend and DAQ components

- pip install dash==1.1.1  
- pip install dash-daq==0.1.0 


STEPS TO RUN THE MODEL

1. Encode_faces.py:
    - Run this file to perform training and encoding of the images in the dataset folder
 
2. Main.py: 
   - Run this file to get the https server. Copy and paste this address to a Chrome browser.

3. Once in the screen, it will present the following information:

 - Tab About: overview of the project
 - Tab Data: contains buttons to perform an action
    - Capture picture: allows to capture a new picture an store it in the image folder
    - Train algorithm: allows to encode faces
    - Facial recognition: allows to open the web camera to do the facial recognition and present basic information 
      obtained from Instagram if the user exists.
    - Stop video: allows to stop the video. 
    
    Note: After initialize facial recognition if you want to push another button please press "STOP VIDEO".
         
 Folders Outline:
 - data set
 - names of insta account, and photos 
 
 - encode_faces.py
 - encodingss.pickle
 - Insta_Info_scraper.py
 - users.txt
 - main.py
 
 
 
