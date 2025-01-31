import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import pickle

# PARAMETERS
path = 'myData'
dimensions=(32,32,3)

# IMPORTING THE DATA 
images=[]
classNo=[]
for i in range (0,10):
    my_l=os.listdir(path+'/'+str(i))
    for j in my_l:
        img=cv2.imread(path+'/'+str(i)+'/'+j)
        img=cv2.resize(img,(32,32))
        images.append(img)
        classNo.append(i)
# CONVERT TO NUMPY ARRRAY
images=np.array(images)
classNo=np.array(classNo)

# SPLITTING THE DATA
x_train,x_test,y_train,y_test=train_test_split(images,classNo,test_size=0.2)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2)

# PREPROCESSING THE IMAGES
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

x_train=np.array(list(map(preProcess,x_train)))
x_test=np.array(list(map(preProcess,x_test)))
x_val=np.array(list(map(preProcess,x_val)))

# RESHAPE IMAGES
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

# IMAGE AUGMENTATION
datagen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.2,
                           shear_range=0.1,
                           rotation_range=10)
datagen.fit(X_train)

# ONE HOT ENCODING OF MATRICES
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)
y_validation=to_categorical(y_validation,10)

# CREATING THE MODEL
noOfFilters = 60
sizeOfFilter1 = (5,5)
sizeOfFilter2 = (3, 3)
sizeOfPool = (2,2)
noOfNodes= 500

model=Sequential()
model.add((Conv2D(no.filters,sizeOfFilter1,input_shape=(32,32,1),activation='relu')))
model.add((Conv2D(no.filters,sizeOfFilter1,activation='relu')))
model.add(MaxPooling2D(pool_size=sizeOfPool))
model.add((Conv2D(no.filters//2,sizeOfFilter2,activation='relu')))
mode.add((Conv2D(no.filters//2,sizeOfFilter2,activation='relu')))
model.add(MaxPooling2D(pool_size=sizeOfPool))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(noOfNodes,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))


model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

# TRAIN THE MODEL
history = model.fit(dataGen.flow(X_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)
# EVALUATE THE MODEL
# SAVE THE MODEL
