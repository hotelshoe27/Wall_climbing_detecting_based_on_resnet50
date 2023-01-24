import cv2
import numpy as np
import os

#- Set data path
climb_data_path = './data/climb'
walking_data_path = './data/walk'
folderlist = os.listdir(walking_data_path)


classes = []
with open("./weight/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_image_saved(frame, size, score_threshold, nms_threshold, count):
    
    #- Using YOLOv4 net
    model_file = './weight/yolov4.weights'
    config_file = './weight/yolov4.cfg' 
    net = cv2.dnn.readNet(model_file, config_file)
    
    #-- Set GPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)  
    net.setInput(blob)

    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            #- Using only person class
            if class_id == 0 and confidence > 0.1:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

                img_crop = frame[y:(y+h), x:(x+w), :]

                #- Save detected person image data
                try:
                    #cv2.imwrite('./data/pre_climb/pre_climbing_image'+str(count)+'.png', img_crop)
                    cv2.imwrite('./data/pre_walk/pre_walking_image'+str(count)+'.png', img_crop)
                except Exception as e:
                    continue
                

    #- Output utility
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]
           
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            
            print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")

    return frame


cnt_num = 0
for i in range(1, len(folderlist)+1):
    #raw_img = './data/climb/climbing'+str(i)+'.png'
    raw_img = './data/walk/walking'+str(i)+'.png'

    data = cv2.imread(raw_img)
    if data is None:
        print('Image data is None')
        continue

    cnt_num = cnt_num + 1
    detect_image_saved(frame=data, size=416, score_threshold=0.4, nms_threshold=0.4, count=cnt_num)