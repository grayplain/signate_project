import tensorflow as tf
from tensorflow import lite
import hand_number as hand
from sklearn.model_selection import train_test_split

def main():
    train_data_count = 50000
    images = hand.load_image_datas(train_data_count)

    # minmax 適用 (0〜255 しかない)
    images = images / 255

    y = hand.read_pd_data('train_master.tsv')[:train_data_count]['category_id'].values
    train_images, test_images, train_y, test_y = train_test_split(images, y)

    # モデルを学習させるプロセス
    model = build_model()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    model.fit(train_images, train_y, batch_size=32, epochs=5)
    model.save('hand_number_keras')

    # 学習済みモデルを読み込む
    # model = load_model('hand_number_keras')

    model.evaluate(test_images, test_y)



def load_model(file_path):
    return tf.keras.models.load_model(file_path)

def build_model():
    # return tf.keras.Sequential([
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(50, activation="relu"),
    #     tf.keras.layers.Dropout(0.1),
    #     tf.keras.layers.Dense(10, activation="softmax")
    # ])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    return model

def convert_tf_to_tflite(filePath):
    model = load_model(filePath)
    # converter = lite.TFLiteConverter.from_saved_model(model)
    converter = lite.TFLiteConverter.from_keras_model(model)
    tfmodel = converter.convert()
    hand.write_fitted_model(tfmodel, file_name='tflitetest.tflite')


main()

convert_tf_to_tflite('hand_number_keras')
