# Facial Recognition and Web Scraping

Before to execute this project it is necessary to install:

- pip install dlib
- pip install face_recognition
- pip install opencv

To visualize the application is necessary install core dash backend and DAQ components

- pip install dash==1.1.1  
- pip install dash-daq==0.1.0 


STEPS TO RUN THE MODEL

1. Encode_faces.py:
    - When this file is ran, it will perform the facial encoding
 
2. Main.py: 
   - When this file is ran, it will run the main screen. The code in this screen contains the web information.

3. Once in the screen, it will present the following information:

 - Tab About: overview of the project
 - Tab Data: contains buttons to perform an action
    - Capture picture: allows to capture a new picture an store it in the image folder
    - Train algorithm: allows to encode faces
    - Facial recognition: allows to open the web camera to do the facial recognition and present basic information 
      obtained from Instagram if the user exists.
    - Stop video: allows to stop the video. 
    
    Note: After initialize facial recognition if you want to push another button please press "STOP VIDEO".
         
 