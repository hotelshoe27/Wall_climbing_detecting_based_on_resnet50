import cv2
import cvlib as cv
import numpy as np
import utils
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


#- Load saved model
model = load_model('./wall_climbing_detection_model.h5')

#- Load sample image file
img_path = './sample/sample_1.png'

img = cv2.imread(img_path)
saram , label, confidence = utils.detect_only_person(img)

#- Preprocessing input image
for idx, f in enumerate(saram):
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]
    
    if 0 <= startX <= img.shape[1] and 0 <= endX <= img.shape[1] and 0 <= startY <= img.shape[0] and 0 <= endY <= img.shape[0]:
            
        saram_region = img[startY:endY, startX:endX]    
        saram_region1 = cv2.resize(saram_region, (224, 224), interpolation = cv2.INTER_AREA)
            
        x = img_to_array(saram_region1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
            
        #- Predict Climbing or Walking
        prediction = model.predict(x)
        pre_cat = np.argmax(prediction, axis=-1)
 
        if prediction < 0.5:
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,255,0), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "Walking ({:.2f}%)".format((1 - prediction[0][0])*100)
            cv2.putText(img, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
        else:
            cv2.rectangle(img, (startX,startY), (endX,endY), (0,0,255), 2)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            text = "Climbing ({:.2f}%)".format(prediction[0][0]*100)
            cv2.putText(img, text, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
#- Displaying output
cv2.imshow("Img Climbing detection", img)

cv2.waitKey()
cv2.destroyAllWindows()