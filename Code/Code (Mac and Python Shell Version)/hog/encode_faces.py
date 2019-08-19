# USAGE
# python encode_faces.py --dataset dataset --encodings encodings.pickle

# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=False,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=False,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())


def encode():
        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR,"..", "dataset") #sister folder path

        imagePaths = list(paths.list_images(image_dir))

        # initialize the list of known encodings and known names
        knownEncodings = []
        knownNames = []

        # loop over the images paths
        for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the images path
                print("[INFO] processing images {}/{}".format(i + 1,
                        len(imagePaths)))
                name = imagePath.split(os.path.sep)[-2]

                # load the input images and convert it from BGR (OpenCV ordering)
                # to dlib ordering (RGB)
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input images
                boxes = face_recognition.face_locations(rgb,
                        model=args["detection_method"])

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)

                # loop over the encodings
                for encoding in encodings:
                        # add each encoding + name to our set of known names and
                        # encodings
                        knownEncodings.append(encoding)
                        knownNames.append(name)

        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open("hog/encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("[INFO] Ready!")

