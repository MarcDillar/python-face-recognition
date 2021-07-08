import os
import face_recognition as fr
import cv2
import numpy as np

class FaceRecognizer:

    def __init__(self, library_folder="faces"):
        self.library_names, self.library_images = [], []

        for folder in os.listdir(library_folder):
            for file in os.listdir(os.path.join(library_folder, folder)):
                if file.endswith(".jpg") or file.endswith(".png"):
                    face = fr.load_image_file(os.path.join(library_folder, folder, file))
                    encoding = fr.face_encodings(face)[0]
                    self.library_names.append(folder)
                    self.library_images.append(encoding)

    def get_face_name(self, face_image):
        best_match_index = np.argmin(fr.face_distance(self.library_images, face_image))

        matches = fr.compare_faces(self.library_images, face_image)

        if matches[best_match_index]:
            return self.library_names[best_match_index]

        return "Unknown" 

    def draw_faces_rectangle(self, original_image, faces_locations, faces_names):
        for (top, right, bottom, left), name in zip(faces_locations, faces_names):
                # Draw a box around the face
                cv2.rectangle(original_image, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(original_image, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(original_image, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    def show_image(self, img):
        # Display the resulting image
        while True:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True

    def classify_face(self, image_path):
        """
        will find all of the faces in a given image and label
        them if it knows what they are

        :param image_path: str of file path
        :return: list of face names
        """
        img = cv2.imread(image_path, 1)
        #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        #img = img[:,:,::-1]

        faces_locations = fr.face_locations(img)
        unknown_faces_images = fr.face_encodings(img, faces_locations)

        faces_names = []
        for unknown_face in unknown_faces_images:
            name = self.get_face_name(unknown_face)

            faces_names.append(name)

            self.draw_faces_rectangle(img, faces_locations, faces_names)

        self.show_image(img)

        return faces_names

if __name__ == '__main__':
    FaceRecognizer().classify_face("test.jpg")
