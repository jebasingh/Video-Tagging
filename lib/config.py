import inspect
import os
import platform
import cv2

(CV_MAJOR_VER, CV_MINOR_VER, mv1) = cv2.__version__.split(".")

_platform = platform.system().lower()
path_to_file = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

RECOGNITION_ALGORITHM = 2
POSITIVE_THRESHOLD = 1000
users = ["blue shirt", "Daniel", "Santhanam","A R Rahman",  "John", "nayanthara",
             "Mani", "User9", "User10"]

TRAINING_FILE = 'training.xml'
TRAINING_DIR = './training_data/'
FACE_WIDTH = 92
FACE_HEIGHT = 112


#HAAR_FACES = 'lib/haarcascade_frontalface_alt.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_alt2.xml'
#HAAR_FACES = 'lib/haarcascade_frontalface_default.xml'
HAAR_FACES = 'lib/haarcascade_frontalface.xml'
HAAR_SCALE_FACTOR = 1.05
HAAR_MIN_NEIGHBORS_FACE = 4    
HAAR_MIN_SIZE_FACE = (30, 30)


def is_cv2():
    if CV_MAJOR_VER == 2:
        return True
    else:
        return False


def is_cv3():
    if CV_MAJOR_VER == 3:
        return True
    else:
        return False


def model(algorithm, thresh):
    # set the choosen algorithm
    model = None
    if not is_cv3():
        # OpenCV version renamed the face module
        if algorithm == 1:
            model = cv2.face.createLBPHFaceRecognizer(threshold=thresh)
        elif algorithm == 2:
            model = cv2.face.createFisherFaceRecognizer(threshold=thresh)
        elif algorithm == 3:
            model = cv2.face.createEigenFaceRecognizer(threshold=thresh)
        else:
            print("WARNING: face algorithm must be in the range 1-3")
            os._exit(1)
    else:
        if algorithm == 1:
            model = cv2.createLBPHFaceRecognizer(threshold=thresh)
        elif algorithm == 2:
            model = cv2.createFisherFaceRecognizer(threshold=thresh)
        elif algorithm == 3:
            model = cv2.createEigenFaceRecognizer(threshold=thresh)
        else:
            print("WARNING: face algorithm must be in the range 1-3")
            os._exit(1)
    return model


def user_label(i):
    i = i - 1
    if i < 0 or i > len(users):
        return "User" + str(int(i))
    return users[i]



"""
 if RECOGNITION_ALGORITHM == 1:
        POSITIVE_THRESHOLD = 100
    elif RECOGNITION_ALGORITHM == 2:
        POSITIVE_THRESHOLD = 250
    else:
        POSITIVE_THRESHOLD = 5500
"""

