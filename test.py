import cv2 as cv

'''Initialisng the camera'''
cap=cv.VideoCapture(0)

#face classifier
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye classifier
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
#mouth classifier
mouth_cascade = cv.CascadeClassifier('Mouth.xml')

def draw_boundary(img):
#function for drawing
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        # mo= mouth_cascade.detectMultiScale(roi_gray)
        # for (ex, ey, ew, eh) in mo:
        #     cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img



# img = cv.imread('sachin.jpg')
while 1:
    ret,img=cap.read()
    cv.imshow('frame',draw_boundary(img))

    if (cv.waitKey(1) & 0xFF == ord('Q')):
        break

cap.release()
cv.imshow('img',draw_boundary(img))
cv.waitKey(0)
cv.destroyAllWindows()