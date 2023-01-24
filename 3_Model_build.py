import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, GlobalMaxPooling2D, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
 
#- setting file path
walikg_data_dir = './data/pre_walk/'
climb_data_dir = './data/pre_climb/'
 
waling_file = os.listdir(walikg_data_dir)
climbing_file = os.listdir(climb_data_dir)
 
file_num = len(waling_file) + len(climbing_file) 
 
#- Preprocessing images
num = 0
all_img = np.float32(np.zeros((file_num, 224, 224, 3))) 
all_label = np.float64(np.zeros((file_num, 1)))

#- Walking
for img_name in waling_file:
    img_path = walikg_data_dir+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 0
    num = num + 1

#- Climbing
for img_name in climbing_file:
    img_path = climb_data_dir+img_name
    img = load_img(img_path, target_size=(224, 224))
    
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    all_img[num, :, :, :] = x
    
    all_label[num] = 1
    num = num + 1
 
 
#- Dataset shuffle
n_elem = all_label.shape[0]
indices = np.random.choice(n_elem, size=n_elem, replace=False)
 
all_label = all_label[indices]
all_img = all_img[indices]
 
 
#- Slpit dataset
num_train = int(np.round(all_label.shape[0]*0.8))
num_test = int(np.round(all_label.shape[0]*0.2))
 
train_img = all_img[0:num_train, :, :, :]
test_img = all_img[num_train:, :, :, :] 
 
train_label = all_label[0:num_train]
test_label = all_label[num_train:]

#- Model build
img_input = (224, 224, 3)
resmodel = ResNet50(input_shape=img_input, weights='imagenet', include_top=False)
resmodel.trainable = False


model = Sequential()
model.add(resmodel)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation= 'sigmoid'))


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('wall_climbing_detection_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#- Train model 
model.fit(train_img, train_label, epochs=30, callbacks=[es, mc], batch_size=16, validation_split = 0.2)
 
 
#- Save model
model.save("wall_climbing_detection_model.h5")