import os
import face_recognition as fr
import cv2
import numpy as np

class FaceRecognizer:

    def __init__(self, faces_folder="./faces"):
        self.encoded = {}

        for dirpath, dnames, fnames in os.walk(faces_folder):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file(faces_folder + "/" + f)
                    encoding = fr.face_encodings(face)[0]
                    self.encoded[f.split(".")[0]] = encoding

    def classify_face(self, image_path):
        """
        will find all of the faces in a given image and label
        them if it knows what they are

        :param image_path: str of file path
        :return: list of face names
        """
        faces_library = self.encoded
        library_images = list(faces_library.values())
        library_names = list(faces_library.keys())

        img = cv2.imread(image_path, 1)
        #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        #img = img[:,:,::-1]

        faces_locations = fr.face_locations(img)
        unknown_faces_images = fr.face_encodings(img, faces_locations)

        face_names = []
        for unknown_face in unknown_faces_images:
            # See if the face is a match for the known face(s)
            matches = fr.compare_faces(library_images, unknown_face)
            name = "Unknown"

            # use the known face with the smallest distance to the new face
            face_distances = fr.face_distance(library_images, unknown_face)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = library_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(faces_locations, face_names):
                # Draw a box around the face
                cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

                # Draw a label with a name below the face
                cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

        # Display the resulting image
        while True:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return face_names

if __name__ == '__main__':
    FaceRecognizer().classify_face("test.jpg")
