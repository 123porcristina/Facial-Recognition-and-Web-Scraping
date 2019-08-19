import cv2
import sys
import numpy as np




# camera actions

class VideoCamera3(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        print("[INFO] starting video stream......")

    def __del__(self):
        print("DEL fue ejecutado")
        self.video.release()

    def get_frame(self):
        
        while True:
            # Capture frame-by-frame
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gx, gy = np.gradient(gray)
            cropped = np.sqrt(np.square(gx) + np.square(gy))
            frame = cropped ##gradient vector mode
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()


def main():
    gr = VideoCamera3()
    gr.get_frame()

if __name__ == "__main__": # "Executed when invoked directly"
    main()
