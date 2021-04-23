#%%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
import pandas as pd
from PIL import Image
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt

# %%
def read_data(name):
    file_name = "data/" + name + ".txt"
    folder_name = "data/" + name + "/"
    file = open(file_name, "r")
    lines = file.readlines()
    labels = []
    images_png = []
    for line in lines:
        id_image = line.split(',')[0]
        image = folder_name + id_image
        if name == "test":
            image = image[:-1]
        images_png.append(image)

        if name == "test":
            id_image = id_image[:-1]
            labels.append(id_image)
        else:
            labels.append(int(line.split(',')[1]))

    file.close()

    if name != "test":
        labels = np.array(labels).astype(np.int64)

    images = np.array([np.array(Image.open(fname)) for fname in images_png])
    images = images.reshape(-1, 50, 50, 1)
    images = images.astype('float32')

    return labels, images

train_labels, train_images = read_data("train")
validation_labels, validation_images = read_data("validation")
id_images, test_images = read_data("test")

train_one_hot = to_categorical(train_labels, 3)
validation_one_hot = to_categorical(validation_labels, 3)

std = np.std(train_images)
avg = np.mean(train_images)

train_images = (train_images - avg) / std
validation_images = (validation_images - avg) / std
test_images = (test_images - avg) / std

#%%

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomContrast(factor=0.6),
    layers.experimental.preprocessing.RandomRotation(factor=20),
    layers.experimental.preprocessing.RandomFlip()
])


model = Sequential()
# model.add(tf.keras.layers.experimental.preprocessing.Normalization(input_shape=(50, 50, 1))) 
# model.add(data_augmentation)

model.add(Conv2D(32, 7, activation='relu', input_shape=(50, 50, 1)))
            # kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(BatchNormalization())

model.add(Conv2D(64, 5, activation='relu'))
            # kernel_regularizer=tf.keras.regularizers.l2(0.001)))


model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, activation='relu'))
            # kernel_regularizer=tf.keras.regularizers.l2(0.001)))

model.add(BatchNormalization())

# model.add(Conv2D(32, 3, activation='relu'))
#             # kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, 3, activation='relu'))
#             # kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# model.add(BatchNormalization())

# model.add(Conv2D(64, 3, activation='relu'))
#             # kernel_regularizer=tf.keras.regularizers.l2(0.001)))


# model.add(BatchNormalization())

# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(100, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(100, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01)))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
#%%
hist = model.fit(
    train_images,
    train_one_hot,
    batch_size=64,
    epochs=1000,
    validation_data=(validation_images, validation_one_hot),
    callbacks=[mc]
)

# %%

model = load_model('best_model.h5')
print(model.evaluate(validation_images, validation_one_hot)[1])
#%%
y_prob = model.predict(validation_images)
validation_preds = y_prob.argmax(axis = -1)

confusion = [[0 for i in range(3)] for j in range(3)]
for i in range(len(y_prob)):
    confusion[validation_labels[i]][validation_preds[i]] += 1

print(np.array(confusion))
plt.imshow(confusion, cmap='gray')

# %%

predictions = model.predict(test_images)
predictions = predictions.argmax(axis=-1)
predictions = [str(val) for val in predictions]
my_submission = pd.DataFrame({'id': id_images, 'label': predictions})
my_submission.to_csv('submission.csv', index=False)