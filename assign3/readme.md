base model accuracy :
Accuracy on test data is: 82.34

my model accuracy reached  83.50 in 29th epoch
Epoch 29/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2851 - acc: 0.8986 - val_loss: 0.5203 - val_acc: 0.8350

my model:
# Define the model
model = Sequential()

model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(32, 32, 3)))
model.add(SeparableConv2D(96, (3, 3),name='sep_conv_2')) # o/p 96 x 30x30 RF 3x3 
model.add(BatchNormalization(name='norm_1'))
model.add(Activation('relu'))

#l = MaxPooling2D(pool_size=(2, 2))(l)

model.add(SeparableConv2D(96, 3)) # o/p 96 x 28x28 RF 5x5
model.add(BatchNormalization(name='norm_2'))
model.add(Activation('relu'))

model.add(SeparableConv2D(192, 3)) # o/p 192x26x26 RF 7x7
model.add(BatchNormalization(name='norm_3'))
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #o/p 192x13x13 RF 8x8
model.add(Dropout(0.25))


model.add(SeparableConv2D(96, 3))# o/p 96x11x11 RF 12x12
model.add(BatchNormalization(name='norm_4'))
model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(SeparableConv2D(100, 3))# o/p 100 x9x9 RF 16x16
model.add(BatchNormalization(name='norm_5'))
model.add(Activation('relu'))

model.add(SeparableConv2D(100, 3))# o/p 100x7x7 RF 20x20
model.add(BatchNormalization(name='norm_6'))
model.add(Activation('relu'))


model.add(GlobalAveragePooling2D(name='avg_pool'))
model.add(Dense(num_classes, activation='softmax'))


model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_2 (Conv2D)            (None, 32, 32, 48)        1344      
_________________________________________________________________
sep_conv_2 (SeparableConv2D) (None, 30, 30, 96)        5136      
_________________________________________________________________
norm_1 (BatchNormalization)  (None, 30, 30, 96)        384       
_________________________________________________________________
activation_7 (Activation)    (None, 30, 30, 96)        0         
_________________________________________________________________
separable_conv2d_6 (Separabl (None, 28, 28, 96)        10176     
_________________________________________________________________
norm_2 (BatchNormalization)  (None, 28, 28, 96)        384       
_________________________________________________________________
activation_8 (Activation)    (None, 28, 28, 96)        0         
_________________________________________________________________
separable_conv2d_7 (Separabl (None, 26, 26, 192)       19488     
_________________________________________________________________
norm_3 (BatchNormalization)  (None, 26, 26, 192)       768       
_________________________________________________________________
activation_9 (Activation)    (None, 26, 26, 192)       0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 13, 13, 192)       0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 13, 13, 192)       0         
_________________________________________________________________
separable_conv2d_8 (Separabl (None, 11, 11, 96)        20256     
_________________________________________________________________
norm_4 (BatchNormalization)  (None, 11, 11, 96)        384       
_________________________________________________________________
activation_10 (Activation)   (None, 11, 11, 96)        0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 11, 11, 96)        0         
_________________________________________________________________
separable_conv2d_9 (Separabl (None, 9, 9, 100)         10564     
_________________________________________________________________
norm_5 (BatchNormalization)  (None, 9, 9, 100)         400       
_________________________________________________________________
activation_11 (Activation)   (None, 9, 9, 100)         0         
_________________________________________________________________
separable_conv2d_10 (Separab (None, 7, 7, 100)         11000     
_________________________________________________________________
norm_6 (BatchNormalization)  (None, 7, 7, 100)         400       
_________________________________________________________________
activation_12 (Activation)   (None, 7, 7, 100)         0         
_________________________________________________________________
avg_pool (GlobalAveragePooli (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010      
=================================================================
Total params: 81,694
Trainable params: 80,334
Non-trainable params: 1,360
_________________________________________________________________

epoch logds:
Epoch 1/50
390/390 [==============================] - 34s 86ms/step - loss: 1.3504 - acc: 0.5145 - val_loss: 1.2294 - val_acc: 0.5630
Epoch 2/50
390/390 [==============================] - 32s 81ms/step - loss: 0.9802 - acc: 0.6528 - val_loss: 1.6453 - val_acc: 0.4358
Epoch 3/50
390/390 [==============================] - 31s 81ms/step - loss: 0.8378 - acc: 0.7053 - val_loss: 1.0110 - val_acc: 0.6611
Epoch 4/50
390/390 [==============================] - 32s 81ms/step - loss: 0.7457 - acc: 0.7388 - val_loss: 1.3237 - val_acc: 0.5806
Epoch 5/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6837 - acc: 0.7619 - val_loss: 1.0449 - val_acc: 0.6551
Epoch 6/50
390/390 [==============================] - 31s 81ms/step - loss: 0.6345 - acc: 0.7776 - val_loss: 0.8163 - val_acc: 0.7223
Epoch 7/50
390/390 [==============================] - 31s 80ms/step - loss: 0.5985 - acc: 0.7909 - val_loss: 0.8322 - val_acc: 0.7188
Epoch 8/50
390/390 [==============================] - 32s 81ms/step - loss: 0.5706 - acc: 0.8010 - val_loss: 1.0293 - val_acc: 0.6682
Epoch 9/50
390/390 [==============================] - 32s 81ms/step - loss: 0.5358 - acc: 0.8125 - val_loss: 0.8298 - val_acc: 0.7242
Epoch 10/50
390/390 [==============================] - 31s 81ms/step - loss: 0.5136 - acc: 0.8211 - val_loss: 0.6562 - val_acc: 0.7789
Epoch 11/50
390/390 [==============================] - 32s 83ms/step - loss: 0.4936 - acc: 0.8284 - val_loss: 1.0635 - val_acc: 0.6810
Epoch 12/50
390/390 [==============================] - 32s 82ms/step - loss: 0.4710 - acc: 0.8367 - val_loss: 0.7372 - val_acc: 0.7427
Epoch 13/50
390/390 [==============================] - 32s 81ms/step - loss: 0.4543 - acc: 0.8410 - val_loss: 0.7049 - val_acc: 0.7595
Epoch 14/50
390/390 [==============================] - 32s 81ms/step - loss: 0.4380 - acc: 0.8488 - val_loss: 0.6039 - val_acc: 0.7957
Epoch 15/50
390/390 [==============================] - 31s 81ms/step - loss: 0.4233 - acc: 0.8530 - val_loss: 0.6985 - val_acc: 0.7854
Epoch 16/50
390/390 [==============================] - 31s 79ms/step - loss: 0.4086 - acc: 0.8575 - val_loss: 0.9022 - val_acc: 0.7299
Epoch 17/50
390/390 [==============================] - 31s 78ms/step - loss: 0.3968 - acc: 0.8612 - val_loss: 0.7755 - val_acc: 0.7607
Epoch 18/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3813 - acc: 0.8662 - val_loss: 0.6846 - val_acc: 0.7801
Epoch 19/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3713 - acc: 0.8688 - val_loss: 0.6654 - val_acc: 0.7831
Epoch 20/50
390/390 [==============================] - 31s 78ms/step - loss: 0.3629 - acc: 0.8726 - val_loss: 0.5821 - val_acc: 0.8095
Epoch 21/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3506 - acc: 0.8748 - val_loss: 0.6908 - val_acc: 0.7865
Epoch 22/50
390/390 [==============================] - 31s 78ms/step - loss: 0.3444 - acc: 0.8781 - val_loss: 0.5996 - val_acc: 0.8117
Epoch 23/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3315 - acc: 0.8845 - val_loss: 0.6277 - val_acc: 0.8053
Epoch 24/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3269 - acc: 0.8837 - val_loss: 0.6129 - val_acc: 0.8104
Epoch 25/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3128 - acc: 0.8879 - val_loss: 0.6059 - val_acc: 0.8074
Epoch 26/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3099 - acc: 0.8916 - val_loss: 0.6160 - val_acc: 0.8031
Epoch 27/50
390/390 [==============================] - 31s 79ms/step - loss: 0.3050 - acc: 0.8931 - val_loss: 0.6069 - val_acc: 0.8067
Epoch 28/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2893 - acc: 0.8981 - val_loss: 0.5998 - val_acc: 0.8117
Epoch 29/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2851 - acc: 0.8986 - val_loss: 0.5203 - val_acc: 0.8350
Epoch 30/50
390/390 [==============================] - 31s 78ms/step - loss: 0.2832 - acc: 0.9002 - val_loss: 0.5912 - val_acc: 0.8237
Epoch 31/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2760 - acc: 0.9016 - val_loss: 0.6146 - val_acc: 0.8111
Epoch 32/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2684 - acc: 0.9043 - val_loss: 0.5887 - val_acc: 0.8224
Epoch 33/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2642 - acc: 0.9054 - val_loss: 0.5498 - val_acc: 0.8315
Epoch 34/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2566 - acc: 0.9082 - val_loss: 0.6918 - val_acc: 0.7952
Epoch 35/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2502 - acc: 0.9110 - val_loss: 0.5562 - val_acc: 0.8236
Epoch 36/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2503 - acc: 0.9099 - val_loss: 0.6120 - val_acc: 0.8155
Epoch 37/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2439 - acc: 0.9133 - val_loss: 0.5862 - val_acc: 0.8238
Epoch 38/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2383 - acc: 0.9146 - val_loss: 0.6170 - val_acc: 0.8230
Epoch 39/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2365 - acc: 0.9156 - val_loss: 0.6071 - val_acc: 0.8187
Epoch 40/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2304 - acc: 0.9171 - val_loss: 0.5395 - val_acc: 0.8399
Epoch 41/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2278 - acc: 0.9192 - val_loss: 0.5851 - val_acc: 0.8249
Epoch 42/50
390/390 [==============================] - 31s 80ms/step - loss: 0.2238 - acc: 0.9189 - val_loss: 0.5993 - val_acc: 0.8267
Epoch 43/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2145 - acc: 0.9227 - val_loss: 0.7066 - val_acc: 0.8126
Epoch 44/50
390/390 [==============================] - 31s 80ms/step - loss: 0.2188 - acc: 0.9214 - val_loss: 0.6420 - val_acc: 0.8133
Epoch 45/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2127 - acc: 0.9228 - val_loss: 0.5671 - val_acc: 0.8366
Epoch 46/50
390/390 [==============================] - 31s 79ms/step - loss: 0.2081 - acc: 0.9251 - val_loss: 0.5714 - val_acc: 0.8310
Epoch 47/50
390/390 [==============================] - 31s 80ms/step - loss: 0.2056 - acc: 0.9260 - val_loss: 0.5862 - val_acc: 0.8305
Epoch 48/50
390/390 [==============================] - 31s 80ms/step - loss: 0.2055 - acc: 0.9254 - val_loss: 0.6642 - val_acc: 0.8199
Epoch 49/50
390/390 [==============================] - 31s 80ms/step - loss: 0.2012 - acc: 0.9273 - val_loss: 0.7510 - val_acc: 0.7984
Epoch 50/50
390/390 [==============================] - 31s 80ms/step - loss: 0.1986 - acc: 0.9281 - val_loss: 0.6392 - val_acc: 0.8236
Model took 1555.10 seconds to train

Accuracy on test data is: 82.36
