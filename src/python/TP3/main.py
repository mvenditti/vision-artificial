import numpy as np
import tensorflow as tf
import cv2

class_names = ['with_mask', 'without_mask']


def detect_faces_and_classify(img, face_cascade, model):
    global class_names
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i, (x, y, w, h) in enumerate(faces):
        roi = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi, (180, 180))
        # normalized = resized/255.
        reshaped = np.reshape(resized, (1, 180, 180, 1))

        predictions = model.predict(reshaped)
        score = tf.nn.softmax(predictions[0])
        label = np.argmax(score)

        # Red color for no mask
        color = (0, 0, 255)
        if label == 0:
            # Green color for mask
            color = (0, 255, 0)

        cv2.putText(img, class_names[label], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)


def face_classifier():
    model_path = "./saved_model/classifier"
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('./with_mask/1.png')
    img2 = cv2.imread('./without_mask/1.png')

    detect_faces_and_classify(img, face_cascade, model)
    detect_faces_and_classify(img2, face_cascade, model)

    cv2.imshow('With Mask', img)
    cv2.imshow('Without Mask', img2)
    cv2.waitKey()


def video_classifier():
    model_path = "./saved_model/classifier"
    model = tf.keras.models.load_model(model_path)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        flipped = cv2.flip(frame, 1)
        detect_faces_and_classify(flipped, face_cascade, model)

        cv2.imshow('MaskNet', flipped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # face_classifier()
    video_classifier()
