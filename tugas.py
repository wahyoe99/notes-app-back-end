import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt

# Mendefinisikan generator data latih
train_datagen = ImageDataGenerator(rescale=1./255)

# Membuat objek data latih
train_generator = train_datagen.flow_from_directory(
        'data/train',  # Direktori data latih
        target_size=(150, 150),  # Mengubah resolusi gambar menjadi 150x150 piksel
        batch_size=32,
        class_mode='categorical')  # Menggunakan kelas categorical untuk melakukan klasifikasi tiga kelas (gunting, batu, kertas)

# Membuat arsitektur model jaringan saraf tiruan
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')
])

# Melakukan kompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.optimizers.Adam(),
              metrics=['accuracy'])

# Melatih model
history = model.fit(
      train_generator,
      steps_per_epoch=25,  # Total batch yang dijalankan pada setiap epoch
      epochs=20,  # Jumlah epoch yang akan dilatih
      verbose=1)

# Menampilkan akurasi dan loss pada setiap epoch
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training Accuracy')
plt.title('Training Accuracy')
plt.legend(loc=0)
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training Loss')
plt.legend(loc=0)
plt.figure()

# Membuat objek data uji
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

# Menghitung akurasi pada data uji
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\nTest accuracy:', test_acc)

# Memprediksi gambar baru
from tensorflow.keras.preprocessing import image
import os

# Menentukan path gambar yang ingin diprediksi
path = 'data/new_image/test.jpg'

# Mengubah gambar menjadi array numpy
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Melakukan prediksi gambar
images = np.vstack([x])
classes = model.predict(images, batch_size=10)

# Menampilkan hasil prediksi
print('\nHasil prediksi:')
if classes[0][0] == 1:
    print('Gunting')
elif
