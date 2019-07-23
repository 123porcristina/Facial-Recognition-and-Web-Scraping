import cv2
import sys


cascPath = "haarcascade_frontalface_default.xml"
cascPath2 = "haarcascade_fullbody.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
bodyCascade = cv2.CascadeClassifier(cascPath2)

video_capture = cv2.VideoCapture(0)
i=0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()


    ####### FACE
    faces = faceCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'FACE', (x + w, y + h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # capture faces and save for training
        cropped = frame[y:y+w, x:x+w]
        saved = "faces/face" + str(i) + ".jpg"
        cv2.imwrite(saved, cropped)
        i=i+1

    #######BODY
    body = bodyCascade.detectMultiScale(
        frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the body
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'BODY', (x+w, y+h), font, 2, (255, 255, 255), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
