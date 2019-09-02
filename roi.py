import cv2
import numpy as np

def draw_boundary(img, classifier, scalefactor, minneighbors, color, text):
    gray_image=cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    features= classifier.detectMultiScale(gray_image, scalefactor, minneighbors)

    coords = []

    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h), color,2)
        cv2.putText(img, text, y-4, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8,color,1,cv2.LINE_AA)
        coords =[x,y,w,h]

    return coords, img

def detect(img, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords, img  = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")

    return img

'''Initialisng the camera'''
cap=cv2.VideoCapture(0)
faceCascade =  cv2.CascadeClassifier('haarcascade_frontalface_default.XML')


while(1):

    ret, frame1 = cap.read()
    img = detect(frame1, faceCascade)

    cv2.imshow('Frame',img)


    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()