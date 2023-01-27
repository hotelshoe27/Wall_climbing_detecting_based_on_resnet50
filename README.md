# Wall_climbing_detecting_based_on_resnet50  
**This simple project detects a person climbing a wall using resnet50**  
![output (2) (1)](https://user-images.githubusercontent.com/120359150/214790541-b36a0394-6860-4891-812a-a7269bb42e44.gif)  
Sample vedio source: https://www.youtube.com/watch?v=OD4_NFJ9Da8  

---

## 0_Requirements  
**1) Download yolo4v weight file.**  
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights  
Download 'yolov4.weights' file to 'weight' folder through above URL.  

**2)(Option)Connect OpenCV with CUDA**  
Before starting this project, you have to build a connection OpenCV with CUDA. (It's possible without this connection, but it's very slow😥, Of course you have to have at least one GPU)  
If you want to know how to connect OpenCV with CUDA, See this URL(Korean): https://prlabhotelshoe.tistory.com/24  

---

## 1_Create a dataset
Collecting images about wall climbing and just walking(Google searching or etc.).  

![123](https://user-images.githubusercontent.com/120359150/214732065-91bf314b-1f90-4ff3-8fa2-4bbd286b302e.PNG)  
Numbering image files using '1_Numbering_file_name.py' for building up dataset.  

![1234](https://user-images.githubusercontent.com/120359150/214735000-17ee9ade-b665-4520-9a97-2da94a38a15d.PNG)  
And Preprocess each images about 'wall climbing', 'Walking' using '2_Create_dataset.py'. This process using YOLOv4 crops only 'person' area in images and restructure a dataset.

---

## 2_Model training  
Train a model based on resnet50 through '3_Model_build.py' file. See source code for details about model.  

---

## 3_Test(Images)  
Testing a built model using '4_Test_image.py' about sample images.  
![1](https://user-images.githubusercontent.com/120359150/214736014-e641b0ee-f4d3-415d-aa13-f674daba356c.PNG)  
![2](https://user-images.githubusercontent.com/120359150/214736018-4aeba363-df1e-41fc-bf6d-3d167ef1d37d.PNG)  

---

## 4_Test(Vedio)  
Testing a built model using '4_Test_video.py' about sample vedio or webcam.  
![output (2)](https://user-images.githubusercontent.com/120359150/214729247-86efd565-9d62-496e-bb4d-ab7c6d1cf13e.gif)  
