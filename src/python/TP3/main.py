import numpy as np
import tensorflow as tf
import cv2


def face_classifier():
    model_path = "./saved_model/classifier"
    model = tf.keras.models.load_model(model_path)
    class_names = ['with_mask', 'without_mask']

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('./without_mask/1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for i, (x, y, w, h) in enumerate(faces):
        roi = img[y:y+h, x:x+w]
        resized = cv2.resize(roi, (180, 180))
        normalized = resized/255.
        reshaped = np.reshape(normalized, (1, 180, 180, 3))

        predictions = model.predict(reshaped)
        score = tf.nn.softmax(predictions[0])
        label = np.argmax(score)

        # Red color for no mask
        color = (0, 0, 255)
        if label == 0:
            # Green color for mask
            color = (0, 255, 0)

        cv2.putText(img, class_names[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x+w, y+h), color)

    cv2.imshow('img', img)
    cv2.waitKey()


if __name__ == '__main__':
    face_classifier()
