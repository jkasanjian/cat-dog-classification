import numpy as np 
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K 
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import image
from keras.regularizers import l2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_TRAIN = 1000        # 20000
NUM_VALIDATION = 100    # 5000
BATCH_SIZE = 24
EPOCHS = 20
NUM_CLASSES = 2
IMG_SIZE = 227
TRAIN_DIR = 'data/train/'
VALIDATION_DIR = 'data/validation/'



if K.image_data_format() == 'channels_first':
    input_shape = (3, IMG_SIZE, IMG_SIZE)
else:
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_data = ImageDataGenerator(rescale=1. / 225)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size = (IMG_SIZE, IMG_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')
    
validation_generator = test_data.flow_from_directory(
        VALIDATION_DIR,
        target_size = (IMG_SIZE, IMG_SIZE),
        batch_size = BATCH_SIZE,
        class_mode = 'categorical')
    

print('Creating Model')


# model = Sequential()
# model.add(Conv2D(filters=96, kernel_size=(11,11), input_shape=(227, 227, 3), padding="valid", strides=(4,4)))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(4096, input_shape=(227*227*3,)))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Dense(4096))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Dense(1000))
# model.add(Activation("relu"))
# model.add(Dropout(0.4))
# model.add(BatchNormalization())
# model.add(Dense(2))
# model.add(Activation('softmax'))
# Initialize model
l2_reg=0.
alexnet = Sequential()

# Layer 1
alexnet.add(Conv2D(96, (11, 11), input_shape=(227,227,3),
    padding='same', kernel_regularizer=l2(l2_reg)))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2
alexnet.add(Conv2D(256, (5, 5), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(512, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 4
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))

# Layer 5
alexnet.add(ZeroPadding2D((1, 1)))
alexnet.add(Conv2D(1024, (3, 3), padding='same'))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
alexnet.add(Flatten())
alexnet.add(Dense(3072))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 7act
alexnet.add(Dense(4096))
alexnet.add(BatchNormalization())
alexnet.add(Activation('relu'))
alexnet.add(Dropout(0.5))

# Layer 8
alexnet.add(Dense(2))
alexnet.add(BatchNormalization())
alexnet.add(Activation('softmax'))

alexnet.summary()

print("Compiling model...")

alexnet.compile(
    loss= 'categorical_crossentropy',
    optimizer='rmsprop', 
    metrics=['accuracy'])

alexnet.fit_generator(
    train_generator,
    steps_per_epoch= NUM_TRAIN // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=NUM_VALIDATION // BATCH_SIZE
)

# model.save_weights('first_try.h5')

# image_pred = image.load_img







