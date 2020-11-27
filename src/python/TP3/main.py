import numpy as np
import tensorflow as tf
import cv2


def tp3():
    model_path = "./saved_model/classifier"
    image_path = "./without_mask/1.png"
    # model = tf.saved_model.load(model_path)
    model = tf.keras.models.load_model(model_path)


    # image = cv2.imread(image_path)

    img_height = 180
    img_width = 180

    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width)
    )


    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch


    predictions = model.predict(img_array/255.)
    score = tf.nn.softmax(predictions[0])
    print(score)

    class_names = ['with_mask', 'without_mask']
    print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))


def face_classifier():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('./friends.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0))

    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    # tp3()
    face_classifier()
