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

    def save_img(self):
        # capture faces and save for training
        # print("button CAPTURE was pressed")
        video_capture = cv2.VideoCapture(0)
        print("[INFO] Taking picture...")
        time.sleep(1)
        ret, frame = video_capture.read()
        saved = self.path + str(self.image_count) + ".jpg"
        cv2.imwrite(saved, frame)
        self.image_count = self.image_count+1
        video_capture.release()
        cv2.destroyAllWindows()
        del video_capture
        print("cerrar camara")
        return "[INFO] Image saved"
        # time.sleep(1)
        # return 0


# def main():
#     name = input("Please enter the name o the subfolder:")
#     img = CaptureImage(name, 1)
#     test = img.create_dir()
#     print(test)
#     print(img.save_img())
#
#
# if __name__ == "__main__": # "Executed when invoked directly"
#     main()

