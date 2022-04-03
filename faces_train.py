import os
import cv2
import numpy as np
import pickle
from PIL import Image

# You can experiment with different classifiers (the same goes for face.py)
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'images')


x_train = []
y_labels = []
ids = {}
current_id = 0

err = []
for root,dirs,files in os.walk(image_dir):
              for file in files:
                            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg') or file.endswith('webp'):
                                          path = os.path.join(root,file)
                                          label = os.path.basename(root).replace(' ','_')

                                          #get the image ready
                                          img = cv2.imread(path)
                                          img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                                          img = Image.open(path).convert('L')
                                          img = np.array(img,'uint8')
                        
                                          #get a label ready
                                          if not label in ids:
                                                        ids[label] = current_id
                                                        id_ = current_id
                                                        current_id += 1

                                         #detect face
                                          face = face_cascade.detectMultiScale(img,scaleFactor=2.5,minNeighbors=5)

                                          if not list(face):
                                                        err.append(file)
                                          #find ROI
                                          for (x,y,w,h) in face:
                                                        roi = img[y:y+h,x:x+w]
                                                        x_train.append(roi)
                                                        y_labels.append(id_)



x_train = np.array(x_train,dtype=object)
y_labels = np.array(y_labels)

with open('labels.pickle','wb') as f:
              pickle.dump(ids,f)

recognizer.train(x_train,y_labels)
recognizer.save('trainer.yaml')
cv2.imshow('Some',x_train[8])
cv2.waitKey(0)
