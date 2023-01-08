import cv2 as cv
import sys

detector = cv.CascadeClassifier()
detector.load(f'../out/iter_3/{sys.argv[1].lower()}/cascade.xml')

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    
    faces = detector.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    cv.imshow('Capture - Face detection', frame)
    
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    detectAndDisplay(frame)
    if cv.waitKey(10) == 27:
        break