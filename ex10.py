import numpy as np                                       # fast maths
import tensorflow as tf                                  # neural network toolbox
from sklearn.datasets import load_digits                 # the digit images
from sklearn.model_selection import train_test_split     # splits data

digits = load_digits()                                   # load 1797 digit images
X_train, X_test, y_train, y_test = train_test_split(     # split learn + exam
    digits.data, digits.target, test_size=0.2, random_state=1)

model = tf.keras.Sequential([                            # build a network, layer by layer
    tf.keras.layers.Dense(64, activation="relu", input_shape=(64,)),  # hidden layer (64 units)
    tf.keras.layers.Dense(10, activation="softmax")      # output: 10 digits (0-9)
])
model.compile(optimizer="adam",                          # how it learns
              loss="sparse_categorical_crossentropy",    # how it measures mistakes
              metrics=["accuracy"])                      # track accuracy
model.fit(X_train, y_train, epochs=10, verbose=0)        # train for 10 passes

loss, acc = model.evaluate(X_test, y_test, verbose=0)    # test on hidden images
print("Test accuracy:", round(acc, 3))                   # show the score

new_image = X_test[0:1]                                  # one new digit image the model never saw
prediction = model.predict(new_image, verbose=0)         # get 10 probabilities
digit = np.argmax(prediction)                            # pick the most likely digit
print("Predicted digit:", digit)                         # show it