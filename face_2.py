import cv2 as cv


#load camera
cap = cv.VideoCapture(0)

while True:
    ret,frame = cap.read()
    cv.imshow('frame',frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
