import cv2
import config

haar_faces = cv2.CascadeClassifier(config.HAAR_FACES)

def detect_faces(image):
    faces = haar_faces.detectMultiScale(image,scaleFactor=config.HAAR_SCALE_FACTOR,minNeighbors=config.HAAR_MIN_NEIGHBORS_FACE,
                                        minSize=config.HAAR_MIN_SIZE_FACE,flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def crop(image, x, y, w, h):
    crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
    midy = y + h / 2
    y1 = max(0, midy - crop_height / 2)
    y2 = min(image.shape[0] - 1, midy + crop_height / 2)
    return image[y1:y2, x:x + w]


def resize(image):
    return cv2.resize(image, (config.FACE_WIDTH, config.FACE_HEIGHT),interpolation=cv2.INTER_LANCZOS4)
