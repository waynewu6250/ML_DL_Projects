import cv2
import dlib
import sys
import os.path
from glob import glob

def detect_cascade(filename, cascade_file="lbpcascade_animeface.xml"):
    
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.2,
                                     minNeighbors=1,
                                     minSize=(16, 16))
    
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (96, 96))
        namelist = os.path.basename(filename).split('.')
        name = str(namelist[0]+namelist[-2])
        save_filename = 'faces/' + '%s-%d-cascade-2.jpg' % (name, i)
        if os.path.exists(save_filename):
            pass
        else:
            cv2.imwrite(save_filename, face)

def detect_dlib(filename):
    
    image = cv2.imread(filename)
    detector = dlib.get_frontal_face_detector()
    face_rects = detector(image,0)

    for i, d in enumerate(face_rects):
        x1 = d.left()
        y1 = d.top()
        x2 = d.right()
        y2 = d.bottom()
        try:
            face = image[y1:y2, x1:x2, :]
            face = cv2.resize(face, (96, 96))
            namelist = os.path.basename(filename).split('.')
            name = str(namelist[0]+namelist[-2])
            save_filename = 'faces/' + '%s-%d-dlib-2.jpg' % (name, i)
            if os.path.exists(save_filename):
                pass
            else:
                cv2.imwrite(save_filename, face)
        except:
            continue
    

if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('imgs2/*.jpg')
    for filename in file_list:
        detect_cascade(filename)
        print("Image detection done: %s"%filename)