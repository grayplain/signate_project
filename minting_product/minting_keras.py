import minting_product as mint
import tensorflow as tf
from keras import regularizers
import numpy as np

def main():
    x_train, y_train = mint.load_data_set(train_data_count=1150)

    model = build_model()
    model.compile(optimizer='nadam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    x_test, _ = mint.load_data_set(train_data_count=100, is_train_data=False)
    predict = model.predict(x_test)

    # boolPredict = predict[:, 0:1] == 1
    # print(boolPredict.astype(np.int))


    print("end.")

def build_model():
    return tf.keras.Sequential([
        # tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(100, activation="relu"),
        # tf.keras.layers.Dense(16, activation="relu"),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])


main()