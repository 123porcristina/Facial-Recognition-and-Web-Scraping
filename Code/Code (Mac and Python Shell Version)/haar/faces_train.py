import cv2
import os
import numpy as np
from PIL import Image
import pickle



cascPath = "haar/cascades/haarcascade_frontalface_default.xml"


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"..", "dataset")#sister folder path


def train():
    faceCascade = cv2.CascadeClassifier(cascPath)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0 #to create labels
    label_ids = {} #to create labels
    y_labels=[] # numbers related to the labels
    x_train = [] #contains pixel values

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
                #print(path)
                if not label in label_ids: #create labels
                    label_ids[label] = current_id
                    current_id +=1
                id_= label_ids[label]

                # print(label_ids)
                # y_labels.append(label) #number for the label
                # x_train.append(path)
                pil_image = Image.open(path).convert("L") #gray scale
                size = (550,550)

                image_array = np.array(pil_image, "uint8") #convert the gray scale in an array
                # print(image_array)
                faces = faceCascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    gx, gy = np.gradient(roi) #take the gradiant to vectorize the images into two values
                    roi = np.sqrt(np.square(gx) + np.square(gy)) # get magnitude to normalize it
                    x_train.append(roi)
                    y_labels.append(id_)

    #print(y_labels)
    #print(x_train)

    with open("haar/labels.pickle","wb") as f:
        pickle.dump(label_ids,f) #save the label ids in a file

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("haar/trainer.yml")
    print("\n \"face-train.py\" successful")


