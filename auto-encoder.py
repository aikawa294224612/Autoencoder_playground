import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import pandas as pd
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras import backend as K

training_path = "/kaggle/input/digit-recognizer/train.csv"
testing_path = "/kaggle/input/digit-recognizer/test.csv"

epoch = 30

digitMNIST = pd.read_csv(training_path)
traindata, valdata = digitMNIST[5000:], digitMNIST[:5000]
print(traindata.shape)

y_train, x_train = traindata['label'], traindata.drop('label', axis=1)
y_val, x_val = valdata['label'], valdata.drop('label', axis=1)


x_train = x_train/255
x_val = x_val/255

x_train = tf.reshape(x_train, [37000, 28, 28])
x_val = tf.reshape(x_val, [5000, 28, 28])

#Autoencoder topology
Stacked_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(100,activation="selu"),
    keras.layers.Dense(30,activation="selu"),
])

Stacked_decoder = keras.models.Sequential([
    keras.layers.Dense(100,activation="selu",input_shape=[30]),
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28,28])
])
model = keras.models.Sequential([Stacked_encoder,Stacked_decoder])

model.summary()

model.compile(loss="binary_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1.5))

history= model.fit(x_train,x_train,
                   epochs=epoch,
                   validation_data = (x_val, x_val), verbose=0)

#Visualise training process
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()

#Encode a sample of the test data
test = pd.read_csv(testing_path)
testPics = tf.convert_to_tensor((test/255))
testPics = tf.reshape(testPics, [28000, 28, 28])
result = model.predict(testPics[:10])

#Visualise the test image and its encoded&decoded representation
fig, axs = plt.subplots(nrows=1, ncols=2)
axs[0].imshow(testPics[0]*255,  vmin=0, vmax=255,)
axs[0].title.set_text("Origional")
axs[1].imshow(result[0]*255,  vmin=0, vmax=255)
axs[1].title.set_text("Encoded and Decoded")

#Show how each digit looks broken down to two dimentions
xCompressed = model.predict(x_val)
tsne = TSNE()
xCompressed = tf.convert_to_tensor(xCompressed)
xCompressed = tf.reshape(xCompressed, [5000, 784])
xCompressed2D = tsne.fit_transform(xCompressed)

plt.figure(figsize=(10,10))
plt.scatter(xCompressed2D[:,0],xCompressed2D[:,1],c = y_val, s=10, cmap= 'tab10')
plt.axis("off")

#Add ledgend
for i in range(10):
    plt.plot(0,0,'o',label = i, )
plt.legend()
plt.show()
