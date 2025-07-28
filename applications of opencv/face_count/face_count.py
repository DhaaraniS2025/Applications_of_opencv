import cv2
import requests
import numpy as np
import imutils

face= cv2.CascadeClassifier(r"C:\Users\Ramkumar b\Downloads\haarcascade_frontalface_default.xml")
url = "http://192.0.0.4:8080/shot.jpg"

frame_count=0
while True:
    img_resp= requests.get(url)
    img_arr= np.array(bytearray(img_resp.content),dtype= np.uint8)
    frame= cv2.imdecode(img_arr,cv2.IMREAD_COLOR)
    frame_count+=1
    if frame_count%2 !=0:
        continue
    #frame= cv2.flip(frame,1)
    frames = imutils.resize(frame,width =600, height=600, inter=cv2.INTER_AREA)
    gray= cv2.cvtColor(frames,cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)
    cv2.putText(frames, f"Faces: {face_count}", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    print(f"Faces detected: {face_count}")

    for(x,y,w,h) in faces:
        cv2.rectangle(frames,(x,y),(x+w,y+h),(0,255,0),2)
        roi_gray= gray[y:y+h,x:x+h]
        roi_color= frame[y:y+h,x:x+h]
    
    cv2.imshow("video",frames)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()