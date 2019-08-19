# Facial Recognition and Web Scraping


PACKAGE INSTALLATION REQUIREMENTS:
- dlib
- face_recognition
- opencv
- pillow
- pickle
- os
- glib
- requests
- imutils
- dash
- base64
- beautifulsoup or bs4
- json
- urllib

To visualize the application is necessary install core dash backend and DAQ components
- pip install dash==1.1.1  
- pip install dash-daq==0.1.0 


FOLDERS OUTLINE:

- Main.py (Main file to run)
- README.md
- assets
- capture_image
        - capture_image.py
- dataset (Contains folders of instagram usernames. These folders contains images)
- gradient
        - gradient.py
- haar
        - cascades
        - face_train.py
        - faces.py
        - insta_Info_Scraper.py
        - labels.pickle
        - trainer.yml
        - users.txt
- hog
        - encode_faces.py
        - encodings.pickle
        - hog.py
        - Insta_Info_Scraper.py
        - users.txt
- saved_images
        - image_captured
        - stop_image.png
        - training_image.png



STEPS TO RUN THE MODEL

1. Download the packages required from the Package Installation Requirements section of this file.
2. Run Main.py on the python shell or any desired text editor: 
   - Run this file to get the https server (output shown below). 
   - Copy and paste the http address to a Chrome browser.
   - in this case ("http://127.0.0.1:8050/")
   
   Output:
   Running on http://127.0.0.1:8050/
   Debugger PIN: 004-066-703
   * Serving Flask app "Main" (lazy loading)
   * Environment: production
   [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
   [2m   Use a production WSGI server instead.[0m
   * Debug mode: on

2. On the web browser:
    - About tab: Reveals the overview of the project
    - Data tab: Contains buttons to perform an action
    - Capture Image: allows the user to capture a new image an store it in the image folder for later training
            - to capture an image for training, first type in the instagram username in the text box.
            - then click Capture Image.
            - this image is stored in a folder with the text input used.
    - Train Algorithm: allows the userto train and encode faces into the algorithms.
            - current images in the folder is used to train the models
    - HOG Webcam: Uses HOG (Histogram of Gradients) facial recognition and presents instagram users information.
    - Haar Webcam: Uses Haar (Haar-like features) facial recognition and presents instagram users information.
    - Stop Webcam: Stops the Haar and HOG outputs.
    - Gradient Demo: Shows the user a gradient view of how the algorithm sees the live feed.
    
    Note: Before clicking any other action after starting any of the webcams, it is necessary to click "Stop Webcam".
         

 
 
