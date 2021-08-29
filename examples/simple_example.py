import cv2
from face_recognizer import FaceRecognizer

LIBRARY_FOLDER_PATH = "#PATH TO THE LIBRARY FOLDER#"
IMAGE_PATH = "#PATH TO THE IMAGE THAT NEEDS TO BE ANALYZED#"

faces_names, image = FaceRecognizer(LIBRARY_FOLDER_PATH).classify(IMAGE_PATH)

cv2.imshow('image', image)
cv2.waitKey(0)
