import cv2

vid = cv2.VideoCapture("face-detection/faces.mp4")
face_cascade = cv2.CascadeClassifier("cascade/frontalface.xml")

while True:
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
    
    for x,y,w,h in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0,0,255), 2)
    cv2.imshow("frame", frame)
    cv2.waitKey(10)
    
vid.release()
cv2.destroyAllWindows()