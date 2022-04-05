import tensorflow as tf
import hand_number as hand
from sklearn.model_selection import train_test_split

def main():
    train_data_count = 100
    images = hand.load_image_datas(train_data_count)
    y = hand.read_pd_data('train_master.tsv')[:train_data_count]['category_id'].values
    train_images, test_images, train_y, test_y = train_test_split(images, y)

    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
    model.fit(train_images, train_y, batch_size=32, epochs=3)

    model.evaluate(train_images, train_y)



def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation="softmax")
    ])


main()