import cv2
import os
import numpy as np
import requests
import progressbar as pb

dest_dir = os.path.expanduser('~') + os.path.sep + '.cvlib' + os.path.sep + 'object_detection' + os.path.sep + 'yolo' + os.path.sep + 'yolov3'
initialize = True

def download_file(url, file_name, dest_dir):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    full_path_to_file = dest_dir + os.path.sep + file_name

    if os.path.exists(dest_dir + os.path.sep + file_name):
        return full_path_to_file

    print("Downloading " + file_name + " from " + url)

    try:
        r = requests.get(url, allow_redirects=True, stream=True)
    except:
        print("Could not establish connection. Download failed")
        return None

    file_size = int(r.headers['Content-Length'])
    chunk_size = 1024
    num_bars = round(file_size / chunk_size)

    bar = pb.ProgressBar(maxval=num_bars).start()
    
    if r.status_code != requests.codes.ok:
        print("Error occurred while downloading file")
        return None

    count = 0
    
    with open(full_path_to_file, 'wb') as file:
        for chunk in  r.iter_content(chunk_size=chunk_size):
            file.write(chunk)
            bar.update(count)
            count +=1

    return full_path_to_file

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def detect_only_person(image, confidence=0.5, nms_thresh=0.3, model='yolov4', enable_gpu=True):
    """A method to detect common objects
    Args:
        image: A colour image in a numpy array
        confidence: A value to filter out objects recognised to a lower confidence score
        nms_thresh: An NMS value
        model: The detection model to be used, supported models are: yolov3, yolov3-tiny, yolov4, yolov4-tiny
        enable_gpu: A boolean to set whether the GPU will be used

    """

    Height, Width = image.shape[:2]
    scale = 0.00392

    global classes
    global dest_dir

    if model == 'yolov3-tiny':
        config_file_name = 'yolov3-tiny.cfg'
        cfg_url = "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
        weights_file_name = 'yolov3-tiny.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    elif model == 'yolov4':
        config_file_name = 'yolov4.cfg'
        cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
        weights_file_name = 'yolov4.weights'
        weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    elif model == 'yolov4-tiny':
        config_file_name = 'yolov4-tiny.cfg'
        cfg_url = 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg'
        weights_file_name = 'yolov4-tiny.weights'
        weights_url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)      

    else:
        config_file_name = 'yolov3.cfg'
        cfg_url = 'https://github.com/arunponnusamy/object-detection-opencv/raw/master/yolov3.cfg'
        weights_file_name = 'yolov3.weights'
        weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)    

    config_file_abs_path = dest_dir + os.path.sep + config_file_name
    weights_file_abs_path = dest_dir + os.path.sep + weights_file_name    
    
    if not os.path.exists(config_file_abs_path):
        download_file(url=cfg_url, file_name=config_file_name, dest_dir=dest_dir)

    if not os.path.exists(weights_file_abs_path):
        download_file(url=weights_url, file_name=weights_file_name, dest_dir=dest_dir)    

    global initialize
    global net

    if initialize:
        classes = ['person']
        net = cv2.dnn.readNet(weights_file_abs_path, config_file_abs_path)
        initialize = False

    # enables opencv dnn module to use CUDA on Nvidia card instead of cpu
    if enable_gpu:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            max_conf = scores[class_id]
            if class_id == 0 and max_conf > confidence:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - (w / 2)
                y = center_y - (h / 2)
                class_ids.append(class_id)
                confidences.append(float(max_conf))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms_thresh)

    bbox = []
    label = []
    conf = []

    for i in indices:
        box = boxes[i[0]]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        bbox.append([int(x), int(y), int(x+w), int(y+h)])
        label.append(str(classes[class_ids[i[0]]]))
        conf.append(confidences[i[0]])
        
    return bbox, label, conf