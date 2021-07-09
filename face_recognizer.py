"""Simple Face Recognizer class

Classes:
    FaceRecognizer
"""

import os
import face_recognition as fr
import cv2
import numpy as np

class FaceRecognizer:
    '''
    Class providing basic face recognition methods

    ...

    Attributes
    ----------
    library_names : list
        List of the names associated to each picture of the library
    library_images:
        List of the images of the library

    Methods
    -------
    classify(image_path):
        Finds faces in the image provided
    '''

    def __init__(self, library_folder="faces"):
        '''
        Create a FaceRecognizer instance

        Parameters:
            library_folder (str, optionnal): path to the folder containing the images of the library
        '''

        self.library_names, self.library_images = [], []

        for folder in os.listdir(library_folder):
            for file in os.listdir(os.path.join(library_folder, folder)):
                if file.endswith(".jpg") or file.endswith(".png"):
                    face = fr.load_image_file(os.path.join(library_folder, folder, file))
                    encoding = fr.face_encodings(face)[0]
                    self.library_names.append(folder)
                    self.library_images.append(encoding)

    def __get_face_name(self, face_image):
        '''
        Private method. Get the name associated to a face passed as parameter.

        Parameters:
            face_image (numpy.ndarray): image

        Returns:
            The face's name if the face was identified. Else, Unknown.
        '''

        best_match_index = np.argmin(fr.face_distance(self.library_images, face_image))

        matches = fr.compare_faces(self.library_images, face_image)

        if matches[best_match_index]:
            return self.library_names[best_match_index]

        return "Unknown"

    def __draw_faces_rectangle(self, image, faces_locations, faces_names, font=cv2.FONT_HERSHEY_DUPLEX):
        '''
        Private method. Draws rectangles around the faces present in an image

        Parameters:
            image (numpy.ndarray): original image
            faces_locations (list): location of all faces identified in the image
            faces_names (list): list of the names associated to each face
            font (int, optionnal): cv2 font used to write the faces names on the picture
        '''
        for (top, right, bottom, left), name in zip(faces_locations, faces_names):
            # Draw a box around the face
            cv2.rectangle(image, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(image, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            cv2.putText(image, name, (left -20, bottom + 15), font, 1.0, (255, 255, 255), 2)

    def classify(self, image_path):
        '''
        Finds the faces in a picture, finds their names and draws a rectangle around them

        Parameters:
            image_path (str): path to the image that needs to be analyzed

        Returns:
            faces_names (list): list of the names of all the identified faces
            img (numpy.ndarray): image with all faces highlighted by rectangles
        
        Raises:
            ValueError: if the image path passed as an argument isn't correct
        '''
        img = cv2.imread(image_path, 1)
        if img is None:
            raise ValueError

        faces_locations = fr.face_locations(img)
        unknown_faces_images = fr.face_encodings(img, faces_locations)

        faces_names = []
        for unknown_face in unknown_faces_images:
            name = self.__get_face_name(unknown_face)

            faces_names.append(name)

            self.__draw_faces_rectangle(img, faces_locations, faces_names)

        return faces_names, img   
