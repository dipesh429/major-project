import numpy as np
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,MaxPooling2D
from keras.layers import Dense,Flatten,SpatialDropout2D
from keras.layers.merge import concatenate
from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import History
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from keras.optimizers import Adam

#from keras.applications.vgg16 import VGG16 as PTModel
#from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
#from keras.applications.inception_v3 import InceptionV3 as PTModel
#from keras.layers import Dropout, Conv2D, multiply, LocallyConnected2D, Lambda

#import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

image_size = tuple((224,224))

def plot_model_history(history):
    """
    Function to plot training and validation data of model
    
    Parameters
    ----------
    history: dictionary
             history of training and validation of model
    Returns
    -------
    None
    
    """
    print(history.history.keys())
    plt.margins(x=0)
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
#def buildNet(num_classes):
    """
    Function to build 4 layer NN with 2 Conv layers, 1 MaxPool layer,
    
    1 GlobalMaxPool layer and 2 Dense layers
    Parameters
    ----------
    num_classes: int
                 Number of classes in training data
    Returns
    -------
    Neural Network created
    """
#    model1=Sequential()
#    model1.add(Convolution2D(32, (3,3),input_shape=(224,224,3),activation='relu'))
#    model1.add(MaxPooling2D(pool_size=(2,2)))
#    model1.add(Convolution2D(64,(3,3),activation='relu'))
#    model1.add(GlobalAveragePooling2D())
#
#    model1.add(Dense(128, activation='relu'))
#    model1.add(Dense(2, activation='softmax'))
#    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#    print(model1.summary())
#    return model1

INIT_LR = 1e-5
EPOCHS = 15

def buildNet():
    in_layer = layers.Input(shape=(224,224,3))
    conv1 = layers.Conv2D(96, 11, strides=4, activation='relu')(in_layer)
   
    pool1 = layers.MaxPool2D(3, 2)(conv1)
    conv2 = layers.Conv2D(256, 5, strides=1, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D(3, 2)(conv2)
    conv3 = layers.Conv2D(384, 3, strides=1, padding='same', activation='relu')(pool2)
    conv4 = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPool2D(3, 2)(conv4)
    flattened = layers.Flatten()(pool3) 
    dense1 = layers.Dense(4096, activation='relu')(flattened)
    drop1 = layers.Dropout(0.2)(dense1)
    dense2 = layers.Dense(4096, activation='relu')(drop1)
    drop2 = layers.Dropout(0.2)(dense2)
    preds = layers.Dense(2, activation='softmax')(drop2)

    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS),metrics=["accuracy"])
   
    
    
    
#    in_lay = Input(shape=(256,256,3))
#    base_pretrained_model = PTModel(input_shape =  (256,256,3), include_top = False, weights = 'imagenet')
##    dont update weights
#    base_pretrained_model.trainable = False 
#    pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
#    pt_features = base_pretrained_model(in_lay)
#    from keras.layers import BatchNormalization
#    bn_features = BatchNormalization()(pt_features)
#    
#    # here we do an attention mechanism to turn pixels in the GAP on an off
#    
#    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(Dropout(0.5)(bn_features))
#    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
#    attn_layer = Conv2D(8, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
#    attn_layer = Conv2D(1, 
#                        kernel_size = (1,1), 
#                        padding = 'valid', 
#                        activation = 'sigmoid')(attn_layer)
#    # fan it out to all of the channels
#    up_c2_w = np.ones((1, 1, 1, pt_depth))
#    up_c2 = Conv2D(pt_depth, kernel_size = (1,1), padding = 'same', 
#                   activation = 'linear', use_bias = False, weights = [up_c2_w])
#    up_c2.trainable = False
#    attn_layer = up_c2(attn_layer)
#    
#    mask_features = multiply([attn_layer, bn_features])
#    gap_features = GlobalAveragePooling2D()(mask_features)
#    gap_mask = GlobalAveragePooling2D()(attn_layer)
#    # to account for missing values from the attention model
#    gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
#    gap_dr = Dropout(0.25)(gap)
#    dr_steps = Dropout(0.25)(Dense(128, activation = 'relu')(gap_dr))
#    out_layer = Dense(2, activation = 'softmax')(dr_steps)
#    retina_model = Model(inputs = [in_lay], outputs = [out_layer])
#    from keras.metrics import top_k_categorical_accuracy
#    def top_2_accuracy(in_gt, in_pred):
#        return top_k_categorical_accuracy(in_gt, in_pred, k=2)
#    
#    retina_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
#                               metrics = ['categorical_accuracy', top_2_accuracy])
#    retina_model.summary()
    
    return model


def trainNet(training_set, validation_set):
    """
    Function to train Neural Network Created, save it as hd5 and plot the various parameters.
    
    Arguments
    ---------
    training_set:   ImageDataGenerator object
                    Training set with labels.
    validation_set: ImageDataGenerator object
                    Validation set with labels.
    
    Returns
    -------
    history: dictionary
             History of training and validation of model.
    """
    
    model = buildNet()
    history = History()
    callbacks = [EarlyStopping(monitor='val_loss', patience=6),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),history]
    
    history = model.fit_generator(training_set,
                                steps_per_epoch = len(training_set)*2,
                                epochs = 20,
                                validation_data = validation_set,
                                validation_steps = len(validation_set),
#                                use_multiprocessing = True,
#                                workers = 8,
                                callbacks=callbacks,
                                )
    model.save('model.hd5')
    hist_df = pd.DataFrame(history.history)
    with open('final.csv',mode='w') as f:
        hist_df.to_csv(f)
        
    plot_model_history(history)

    #model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=100,callbacks=callbacks,verbose=1)
    return history

#Since 255 is the maximin pixel value. Rescale 1./255 is to transform every pixel value from range [0(black),255(white)] -> [0,1] 
#the parameter rescale is to multiply every pixel in the preprocessing image (before applying any other transformation)
    
#Without scaling, the high pixel range images will have large amount of votes to determine how to update weights. 
#For example, black/white cat image could be higher pixel range than pure black cat image, 
#but it just doesn't mean black/white cat image is more important for training.
#But for visual understanding, you care about the contour(an outline representing) more than how strong is the contrast as long as the contour is reserved.
    
train_datagen = ImageDataGenerator (rescale = 1./255,rotation_range=25, width_shift_range=0.1,
                                        height_shift_range=0.1, shear_range=0.2, 
                                        zoom_range=0.2,horizontal_flip=True, 
                                        fill_mode="nearest")

test_datagen = ImageDataGenerator (rescale = 1./255)


training_set = train_datagen.flow_from_directory('dataset/train/',
                                                 target_size = image_size,
                                                 batch_size = 32)
print(len(training_set))

validation_set = test_datagen.flow_from_directory('dataset/test/',
                                                  target_size = image_size,
                                                  batch_size = 32,                                        
                                                  shuffle=False)

history = trainNet(training_set, validation_set)
    
from keras.models import load_model
mod=load_model('model.hd5')


def result():
    
    """
    Function to predict if the retina image has diabetic retinopathy or not.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    y_pred: bool
            Whether or not the retina has diabetic retinopathy.
    percent_chance: float
    
            Percentage of chance the retina image has diabetic retinopathy.
    """
    
    mod=load_model('model.hd5')

    test_gen = ImageDataGenerator(rescale = 1./255)

    test_data = test_gen.flow_from_directory('final/',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary', shuffle=False)
    predicted = mod.predict_generator(test_data)
    
    y_pred = predicted[0][0] > 0.4
    percent_chance = round(predicted[0][0]*100, 2)
    return y_pred, percent_chance

mod.predict_generator(test_data)


