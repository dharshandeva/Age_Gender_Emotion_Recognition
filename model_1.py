import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

train_dir = "../train/" #passing the path with training images
test_dir = "../test/"   #passing the path with testing images

classes=os.listdir("../train/")
classes

train_count = []
train_dict = {}
test_count = []
test_dict = {}


# avg_of_train_data=0
# avg_of_test_data=0

def test_train_distribution():
    print("Train Set :")

    for folder in os.listdir(train_dir):
        # print(folder, "folder contains:", len(os.listdir(train_dir+folder)), "image")
        train_count.append(len(os.listdir(train_dir + folder)))
        train_dict[folder] = len(os.listdir(train_dir + folder))
    avg_of_train_data = sum(train_count) / len(train_count)
    print(train_dict)
    print("Avgerage dataset len should be ~", round(avg_of_train_data))
    print()

    #####################test##################

    #####################test##################

    print("Test Set :")

    for folder in os.listdir(test_dir):
        # print(folder, "folder contains:", len(os.listdir(test_dir+folder)), "images")
        test_count.append(len(os.listdir(test_dir + folder)))
        test_dict[folder] = len(os.listdir(test_dir + folder))
    avg_of_test_data = sum(test_count) / len(test_count)
    print("Avgerage dataset len should be ~", round(avg_of_test_data))

    print(test_dict)


test_train_distribution()

def plot_data_dist(dictn):
  def addlabels(x,y):
      for i in range(len(x)):
          plt.text(i,y[i],y[i])


  names = list(dictn.keys())
  values = list(dictn.values())
  addlabels(names,values)

  plt.bar(range(len(dictn)), values, tick_label=names)
  plt.show()

plot_data_dist(train_dict)
plot_data_dist(test_dict)


#averages of train test sets
avg_of_train_data=round(sum(train_count)/len(train_count))
avg_of_test_data=round(sum(test_count)/len(test_count))
avg_of_train_data,avg_of_test_data
#average of test train

# Definnig a function to do so
def grayscale_RGB_and_upsizing(image, size=[224, 224]):
    image = tf.image.resize(tf.convert_to_tensor(image), size)

    return image


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

# train_datagen=ImageDataGenerator(rescale=1/255)
# test_datagen=ImageDataGenerator(rescale=1/255)

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=grayscale_RGB_and_upsizing)
test_datagen = ImageDataGenerator(rescale=1 / 255,
                                  preprocessing_function=grayscale_RGB_and_upsizing)
train_set = train_datagen.flow_from_directory(train_dir,
                                              target_size=(224, 224),
                                              batch_size=32,

                                              class_mode='categorical')
test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size=(224, 224),
                                            batch_size=32,

                                            class_mode='categorical')

# See the shape of any data
train_sample=next(train_set)
print(train_sample[0].shape)

train_set.class_indices
#Optional list of class subdirectories (e.g. ['dogs','cats']).
# Default: None. If not provided, the list of classes will be automatically
#inferred from the subdirectory names/structure under directory

model=tf.keras.applications.MobileNetV2()

# model.summary()
#Removing last layer
ip=model.layers[0].input
op=model.layers[-2].output
op

#adding last layers
final_output=keras.layers.Dense(128)(op)
final_output=keras.layers.Activation('relu')(final_output)
final_output=keras.layers.Dense(64)(final_output)
final_output=keras.layers.Activation('relu')(final_output)
final_output=keras.layers.Dense(7,activation='softmax')(final_output)

final_output

new_model=keras.Model(inputs =ip,outputs=final_output)
new_model.summary()

new_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#'categorical_crossentropy': to find loss in multiclass classification with OHE output
#optimizer !
new_model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early=EarlyStopping(monitor='val_accuracy',patience=3,restore_best_weights=True,verbose=1,min_delta=0.001)
history=new_model.fit(train_set,epochs=5,validation_data=test_set,batch_size=32)

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


plt.show()


y_pred = model.predict(test_set, batch_size=32)

new_model.save_weights('model_weights_1.h5')