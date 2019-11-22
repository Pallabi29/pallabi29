# pallabi29

https://colab.research.google.com/gist/Pallabi29/385bdd7a50e397932a48ce41e85cc945/copy-of-eighth.ipynb

from keras.layers import Activation
model = Sequential()
 
model.add(Convolution2D(16, 3, 3, activation='relu', input_shape=(28,28,1))) #26
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(16, 3, 3, activation='relu')) #24
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Convolution2D(10, 1, 1, activation='relu')) #22
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(MaxPooling2D(pool_size=(2, 2)))#11
model.add(Convolution2D(16, 3, 3, activation='relu'))#9
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, activation='relu'))#7
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, activation='relu'))#5
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(16, 3, 3, activation='relu'))#3
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Convolution2D(10, 4, 4))
model.add(BatchNormalization())
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Activation('softmax'))


model.summary()
________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_49 (Conv2D)           (None, 26, 26, 16)        160       
_________________________________________________________________
batch_normalization_44 (Batc (None, 26, 26, 16)        64        
_________________________________________________________________
dropout_44 (Dropout)         (None, 26, 26, 16)        0         
_________________________________________________________________
conv2d_50 (Conv2D)           (None, 24, 24, 16)        2320      
_________________________________________________________________
batch_normalization_45 (Batc (None, 24, 24, 16)        64        
_________________________________________________________________
dropout_45 (Dropout)         (None, 24, 24, 16)        0         
_________________________________________________________________
conv2d_51 (Conv2D)           (None, 24, 24, 10)        170       
_________________________________________________________________
batch_normalization_46 (Batc (None, 24, 24, 10)        40        
_________________________________________________________________
dropout_46 (Dropout)         (None, 24, 24, 10)        0         
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 12, 12, 10)        0         
_________________________________________________________________
conv2d_52 (Conv2D)           (None, 10, 10, 16)        1456      
_________________________________________________________________
batch_normalization_47 (Batc (None, 10, 10, 16)        64        
_________________________________________________________________
dropout_47 (Dropout)         (None, 10, 10, 16)        0         
_________________________________________________________________
conv2d_53 (Conv2D)           (None, 8, 8, 16)          2320      
_________________________________________________________________
batch_normalization_48 (Batc (None, 8, 8, 16)          64        
_________________________________________________________________
dropout_48 (Dropout)         (None, 8, 8, 16)          0         
_________________________________________________________________
conv2d_54 (Conv2D)           (None, 6, 6, 16)          2320      
_________________________________________________________________
batch_normalization_49 (Batc (None, 6, 6, 16)          64        
_________________________________________________________________
dropout_49 (Dropout)         (None, 6, 6, 16)          0         
_________________________________________________________________
conv2d_55 (Conv2D)           (None, 4, 4, 16)          2320      
_________________________________________________________________
batch_normalization_50 (Batc (None, 4, 4, 16)          64        
_________________________________________________________________
dropout_50 (Dropout)         (None, 4, 4, 16)          0         
_________________________________________________________________
conv2d_56 (Conv2D)           (None, 1, 1, 10)          2570      
_________________________________________________________________
batch_normalization_51 (Batc (None, 1, 1, 10)          40        
_________________________________________________________________
dropout_51 (Dropout)         (None, 1, 1, 10)          0         
_________________________________________________________________
flatten_7 (Flatten)          (None, 10)                0         
_________________________________________________________________
activation_7 (Activation)    (None, 10)                0         
=================================================================
Total params: 14,100
Trainable params: 13,868
Non-trainable params: 232
___________________________________________________________


Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 16s 272us/step - loss: 0.5295 - acc: 0.8540 - val_loss: 0.0990 - val_acc: 0.9796
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 12s 200us/step - loss: 0.2526 - acc: 0.9259 - val_loss: 0.0627 - val_acc: 0.9875
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 12s 198us/step - loss: 0.1983 - acc: 0.9424 - val_loss: 0.0600 - val_acc: 0.9859
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 12s 199us/step - loss: 0.1695 - acc: 0.9468 - val_loss: 0.0468 - val_acc: 0.9884
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 12s 195us/step - loss: 0.1505 - acc: 0.9514 - val_loss: 0.0347 - val_acc: 0.9910
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 12s 197us/step - loss: 0.1384 - acc: 0.9522 - val_loss: 0.0350 - val_acc: 0.9909
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 12s 194us/step - loss: 0.1302 - acc: 0.9532 - val_loss: 0.0271 - val_acc: 0.9924
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 12s 196us/step - loss: 0.1234 - acc: 0.9529 - val_loss: 0.0272 - val_acc: 0.9935
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 12s 195us/step - loss: 0.1147 - acc: 0.9557 - val_loss: 0.0250 - val_acc: 0.9933
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 12s 196us/step - loss: 0.1134 - acc: 0.9548 - val_loss: 0.0235 - val_acc: 0.9932
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 12s 195us/step - loss: 0.1097 - acc: 0.9557 - val_loss: 0.0266 - val_acc: 0.9931
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 12s 196us/step - loss: 0.1059 - acc: 0.9557 - val_loss: 0.0222 - val_acc: 0.9936
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 12s 198us/step - loss: 0.1017 - acc: 0.9574 - val_loss: 0.0265 - val_acc: 0.9927
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 12s 199us/step - loss: 0.1021 - acc: 0.9565 - val_loss: 0.0213 - val_acc: 0.9942
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 12s 194us/step - loss: 0.0984 - acc: 0.9573 - val_loss: 0.0221 - val_acc: 0.9938
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 12s 195us/step - loss: 0.0971 - acc: 0.9562 - val_loss: 0.0215 - val_acc: 0.9938
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 12s 196us/step - loss: 0.0957 - acc: 0.9578 - val_loss: 0.0230 - val_acc: 0.9941
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 12s 194us/step - loss: 0.0954 - acc: 0.9574 - val_loss: 0.0224 - val_acc: 0.9942
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 12s 198us/step - loss: 0.0919 - acc: 0.9587 - val_loss: 0.0238 - val_acc: 0.9934
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 12s 196us/step - loss: 0.0923 - acc: 0.9583 - val_loss: 0.0218 - val_acc: 0.9941
<keras.callbacks.History at 0x7fb0ff6f3d68>




RESULT:

[0.021833949559868778, 0.9941]


Strategy :

Added dropoutin the 3rd Convolution step to decrease number of params
