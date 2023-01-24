import cvlib as cv
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import utils

#- load saved model
model = load_model('./wall_climbing_detection_model.h5')

#- load video file
vedio_path = './sample/sample_video.mp4'

#- activate camera
webcam = cv2.VideoCapture(vedio_path) # if you want to use webcam, vedop_path -> 0

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():

    status, frame = webcam.read()
    
    if not status:
        print("Could not read frame")
        exit()
 
    #- Detecting only person
    saram, label, confidence = utils.detect_only_person(frame)               

    #- Boxing object area
    for idx, f in enumerate(saram):
        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        if 0 <= startX <= frame.shape[1] and 0 <= endX <= frame.shape[1] and 0 <= startY <= frame.shape[0] and 0 <= endY <= frame.shape[0]:
            
            saram_region = frame[startY:endY, startX:endX]
            
            saram_region1 = cv2.resize(saram_region, (224, 224), interpolation = cv2.INTER_AREA)
            
            x = img_to_array(saram_region1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            
            #- Predict Climbing or Walking
            prediction = model.predict(x)
 
            if prediction < 0.5:
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Walking ({:.2f}%)".format((1 - prediction[0][0])*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
            else:
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,0,255), 2)
                Y = startY - 10 if startY - 10 > 10 else startY + 10
                text = "Climbing ({:.2f}%)".format(prediction[0][0]*100)
                cv2.putText(frame, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
    #- Displaying output 
    cv2.imshow("vedio Climbing Detection", frame)
 
    #- If you want to stop, press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
webcam.release()
cv2.destroyAllWindows() 