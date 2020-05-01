# -*- coding:utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import array_to_img, img_to_array,load_img
import numpy as np
import tkinter
from tkinter import *
from PIL import Image, ImageDraw

canvas_width = 280
canvas_hight = 280
button_width = 15
button_hight = 5

num_class = 10
ans = 0

class PyWindow:
      
      def image_recognition(self):

            img = img_to_array(load_img('outfile.jpg', target_size=(28, 28), color_mode='grayscale'))

            X = []
            X.append(img)
            X = np.asarray(X)
            X = X.astype('float32')
            X  = X / 255.0

            ans = new_model.predict(X).argmax()

            print('予測値 : ', ans)
            self.text.set(ans)

      def draw_line(self, event):
            self.mouse_positionX = event.x
            self.mouse_positionY = event.y
            self.canvas.create_oval(self.mouse_positionX, self.mouse_positionY, event.x, event.y, width=20, outline='white', tag="line")

      def mouse_dragged(self, event):
            self.canvas.create_line(self.mouse_positionX, self.mouse_positionY, event.x, event.y, width=20, fill='white', tag="line")
            self.draw.line([(self.mouse_positionX, self.mouse_positionY), (event.x, event.y)], width=20, fill='white')
            self.mouse_positionX = event.x
            self.mouse_positionY = event.y

      def mouse_release(self, event):
            self.canvas.update()
            self.newimg.save('outfile.jpg')
            self.image_recognition()

      def erase(self, event):
            self.canvas.delete("oval")
            self.canvas.delete("line")
            self.draw.rectangle([(0, 0),(canvas_width, canvas_hight)], fill='black')


      def __init__(self):
            window = tkinter.Tk()
            window.geometry('700x400')
            window.title("手書き文字認識")

            self.text = tkinter.StringVar()
            self.text.set(ans)
            self.numlabel = tkinter.Label(window, textvariable=self.text, font=("", 100))
            self.numlabel.place(x=300, y=10) 

            self.newimg = Image.new('RGB', (canvas_width, canvas_hight), 'black')
            self.draw = ImageDraw.Draw(self.newimg)

            self.canvas = tkinter.Canvas(window, width=canvas_width, height=canvas_hight, bg="black")
            self.canvas.place(x=10, y=10)
            self.canvas.create_rectangle(0, 0, canvas_width, canvas_hight, outline='white', width=3)
      
            self.deletebutton = tkinter.Button(window, text='削除', width=button_width)
            self.deletebutton.place(x=150, y=300)
            self.deletebutton.bind("<Button-1>", self.erase)

            self.canvas.bind("<Button-1>", self.draw_line)
            self.canvas.bind("<B1-Motion>", self.mouse_dragged)
            self.canvas.bind("<ButtonRelease>", self.mouse_release)

            window.mainloop()


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(keras.backend.image_data_format())
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
      keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
      keras.layers.Conv2D(64, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Dropout(0.25),
      keras.layers.Flatten(),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(num_class, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=6)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

model.summary()

print(test_acc)

model.save('learned_model.h5')

new_model = tf.keras.models.load_model('learned_model.h5')

PyWindow()





  


