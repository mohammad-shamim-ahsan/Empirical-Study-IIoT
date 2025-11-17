import warnings
warnings.filterwarnings("ignore")
import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import concatenate, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
)
from tensorflow.keras.optimizers import SGD

import os
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Input,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.models import Model,load_model,Sequential
from tensorflow.keras.applications.xception import  Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2L
import tensorflow.keras.callbacks as kcallbacks
import keras
#from keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.utils import img_to_array
import math
import random
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import os
import math
import operator
import numpy as np
from PIL import Image
from collections import defaultdict
import tensorflow as tf  # Added for GPU support check    

gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus)

if not gpus:
    print("No GPUs detected.")
else:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")

#Generate Images from Test Set
TARGET_SIZE=(224,224)
INPUT_SIZE=(224,224,3)
BATCHSIZE=32

#Normalization
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './tmp_ack_not_Datasets/train_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        './tmp_ack_not_Datasets/test_A',
        target_size=TARGET_SIZE,
        batch_size=BATCHSIZE,
        class_mode='categorical')


'''Generic CNN'''

input_shape=INPUT_SIZE
num_class=15
epochs=5                #Training is executed for 25 epochs for the better visualization of results 
savepath='./models_ack/generic_cnn.h5'

#Define CNN Model
model = Sequential()
model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=input_shape,padding='same',activation='relu'))
model.add(Conv2D(32,(5,5),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128,(5,5),strides=(1,1),padding='same',activation='selu'))
model.add(Conv2D(128,(5,5),strides=(1,1),padding='same',activation='selu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(256,activation='selu'))
model.add(Dropout(rate=0.1))
model.add(Dense(num_class,activation='softmax'))
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])


'''Xception'''

savepath='./models_ack/xception.h5'

#Define Xception Model
model_fine_tune = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:128]:
    layer.trainable = False
for layer in model_fine_tune.layers[128:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=128,activation='selu')(model)
model=Dropout(0.2)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='xception')
opt = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

# '''VGG16'''

savepath='./models_ack/VGG16.h5'

#Define VGG16 Model
model_fine_tune = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:8]:	
    layer.trainable = False
for layer in model_fine_tune.layers[8:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output) 
model=Dense(units=128,activation='elu')(model)
model=Dropout(0.5)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='vgg')
opt = keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

'''VGG19'''

savepath='./models_ack/VGG19.h5'

#Define VGG19 Model
model_fine_tune = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:10]:	
    layer.trainable = False
for layer in model_fine_tune.layers[10:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model = Dense(units=128,activation='relu')(model)
model = Dropout(0.3)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='vgg')
opt = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
   
#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

# '''Inception'''

savepath='./models_ack/inception.h5'

#Define Inception Model
model_fine_tune = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:45]:
    layer.trainable = False
for layer in model_fine_tune.layers[45:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=256,activation='relu')(model)
model=Dropout(0.5)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='resnet')
opt = keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

# '''InceptionResnet V2'''
savepath='./models_ack/inceptionresnetv2.h5'

#Define Inception Resnet V2 Model
model_fine_tune = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:451]:
    layer.trainable = False
for layer in model_fine_tune.layers[451:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=128,activation='selu')(model)
model=Dropout(0.3)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='resnet')
opt = keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

# '''EfficientNetB7'''

savepath='./models_ack/efficientnetb7.h5'

#Define EfficientNetB7 Model
model_fine_tune = EfficientNetB7(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:288]:
    layer.trainable = False
for layer in model_fine_tune.layers[288:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=128,activation='selu')(model)
model=Dropout(0.4)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='efficientnetb7')
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

# '''EfficientNetV2L'''

savepath='./models_ack/efficientnetv2l.h5'

#Define EfficientNetB7 Model
model_fine_tune = EfficientNetV2L(include_top=False, weights='imagenet', input_shape=input_shape)
for layer in model_fine_tune.layers[:316]:
    layer.trainable = False
for layer in model_fine_tune.layers[316:]:
    layer.trainable = True
model = GlobalAveragePooling2D()(model_fine_tune.output)
model=Dense(units=128,activation='selu')(model)
model=Dropout(0.5)(model)
model = Dense(num_class, activation='softmax')(model)
model = Model(model_fine_tune.input, model, name='efficientnetv2-l')
opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)	
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Model Training
saveBestModel = kcallbacks.ModelCheckpoint(filepath=savepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
history=model.fit(train_generator,steps_per_epoch=len(train_generator),epochs=epochs,validation_data=validation_generator,
                  validation_steps=len(validation_generator), callbacks=[saveBestModel])

