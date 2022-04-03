import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yaml')

cap = cv2.VideoCapture(0)

while True:
              ret,frame = cap.read()
              gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
              cv2.putText(frame,str(cap.get(cv2.CAP_PROP_FPS)),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255))

              faces = face_cascade.detectMultiScale(gray,scaleFactor=2.5,minNeighbors=5)


              for (x,y,w,h) in faces:
                            roi_gray = gray[y:y+h,x:x+w]
                            roi_color = frame[y:y+h,x:x+w]
                            
                            id_, conf = recognizer.predict(roi_gray)

                            color = (255,0,0)
                            stroke = 2
                            cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke)

                            if conf >= 45 and conf <= 85:
                                          print(id_,conf)
              cv2.imshow('frame',frame)

              if cv2.waitKey(20) & 0xFF == ord('q'):
                            break
cap.release()
cap.destroyAllWindows()