import cv2
import numpy as np


from keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input

mdl = load_model('MaskNet1.hdf5')


faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}

while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.4,5)
    no_of_faces=len(faces)
    if no_of_faces>=1:
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
            roi_color = frame[y:y+h,x:x+w]
            img = roi_color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224,224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis = 0)
            value =np.argmax(mdl.predict(img))
            mask_status=labels_dict[value]
            arr = np.nanmax(mdl.predict(img))
            label = "{}: {:.2f}%".format(mask_status,arr* 100)
            cv2.putText(frame, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.95, color_dict[value], 3)
            cv2.rectangle(frame, (x, y), (x+w, y+w), color_dict[value], 5)
    #else:
        #print("face_not_found")
    cv2.imshow('EagleEye',frame)

    k = cv2.waitKey(30)#esc
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()
