import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import tensorflow as tf
import keras

training_path = "/kaggle/input/digit-recognizer/train.csv"
testing_path = "/kaggle/input/digit-recognizer/test.csv"

epoch = 30

#Download Data
digitMNIST = pd.read_csv(training_path)
traindata, valdata = digitMNIST[5000:], digitMNIST[:5000]
print(traindata.shape)

y_train, x_train = traindata['label'], traindata.drop('label', axis=1)
y_val, x_val = valdata['label'], valdata.drop('label', axis=1)


x_train = x_train/255
x_val = x_val/255

x_train = tf.reshape(x_train, [37000, 28, 28])
x_val = tf.reshape(x_val, [5000, 28, 28])

# testing data處理
test = pd.read_csv(testing_path)
x_test = tf.convert_to_tensor((test/255))
x_test = tf.reshape(x_test, [28000, 28, 28])

# 加入noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)  # numpy.random.normal 函數裡的三個參數分别代表生成的高斯分布的均值、標準差以及輸出的 size
x_val_noisy = x_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_val.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)  # 把 array 限制在一定範圍内
x_val_noisy = np.clip(x_val_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# show出加過noise的traing images
for i in range(1,5):
    plt.subplot(2,2, i)
    plt.imshow(x_train_noisy[i]*255)

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
              optimizer='adam')

history= model.fit(x_train_noisy, x_train,
                   epochs=epoch,
                   validation_data = (x_val_noisy, x_val), verbose=0)

#Visualise training process
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()

#Visualise the test image and its encoded&decoded representation
result = model.predict(x_test[:10])

fig, axs = plt.subplots(nrows=1, ncols=3)
axs[0].imshow(x_test[0]*255,  vmin=0, vmax=255,)
axs[0].title.set_text("Origional")

axs[1].imshow(x_train_noisy[0]*255,  vmin=0, vmax=255,)
axs[1].title.set_text("add Noise")

axs[2].imshow(result[0]*255,  vmin=0, vmax=255)
axs[2].title.set_text("Encoded and Decoded")

#Show how each digit looks broken down to two dimentions
pred_val = model.predict(x_val)
tsne = TSNE()
pred_val = tf.convert_to_tensor(pred_val)
pred_val = tf.reshape(pred_val, [5000, 784])
pred_val_2D = tsne.fit_transform(pred_val)
print(pred_val_2D) # 到二維空間的座標 shape=(5000, 2)
# t-SNE裡面有PCA!
# t-SNE 的隨機性：每次執行跑出來的結果都是不一樣的，不像PCA

plt.figure(figsize=(10,10))
# 做散佈圖
plt.scatter(pred_val_2D[:,0],pred_val_2D[:,1],c = y_val, s=10, cmap= 'tab10')
plt.axis("off")

#Add ledgend
for i in range(10):
    plt.plot(0,0,'o',label = i, )
plt.legend()
plt.show()
