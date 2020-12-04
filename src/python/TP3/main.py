import numpy as np
import tensorflow as tf
import cv2
import time

class_names = ['With mask', 'Without mask']


def detect_faces_and_classify(img, face_cascade, model):
    global class_names
    # obtenemos las caras usando el detector aplicado a la imagen
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for i, (x, y, w, h) in enumerate(faces):
        # sacamos la region de interes
        roi = gray[y:y + h, x:x + w]
        # resize para ingresar al modelo
        resized = cv2.resize(roi, (180, 180))
        # normalized = resized/255.
        reshaped = np.reshape(resized, (1, 180, 180, 1))

        # obtenemos las probabilidades para cada clase
        predictions = model.predict(reshaped)
        score = tf.nn.softmax(predictions[0])
        # generamos la label partiendo de las probabilidades
        label = np.argmax(score)

        # mostramos el recuadro con label rojo o verde dependiendo de lo obtenido
        # Red color for no mask
        color = (0, 0, 255)
        if label == 0:
            # Green color for mask
            color = (0, 255, 0)

        # agregamos la label y el rectangulo a la imagen
        cv2.putText(img, class_names[label], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

# para probar la clasificacion en imagenes
def face_classifier():
    # tomamos el modelo guardado (proveniente del collab)
    model_path = "./saved_model/classifier"
    # cargamos el modelo
    model = tf.keras.models.load_model(model_path)
    # elegimos el detector de caras
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # leemos las imagenes
    img = cv2.imread('./with_mask/1.png')
    img2 = cv2.imread('./without_mask/1.png')

    # le pasamos al metodo la imagen, el detector de caras y el modelo. Obtenemos la misma imagen con tags
    detect_faces_and_classify(img, face_cascade, model)
    detect_faces_and_classify(img2, face_cascade, model)

    # mostramos las imagenes
    cv2.imshow('With Mask', img)
    cv2.imshow('Without Mask', img2)
    cv2.waitKey()


def video_capture_classifier():
    # tomamos el modelo guardado (proveniente del collab)
    model_path = "./saved_model/classifier"
    # cargamos el modelo
    model = tf.keras.models.load_model(model_path)
    # elegimos el detector de caras
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # capturamos imagenes de la webcam
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        flipped = cv2.flip(frame, 1)
        # a medida que lee cada frame, le pasamos el frame a la funcion. Obtenemos frame con tags
        detect_faces_and_classify(flipped, face_cascade, model)

        # mostramos el frame
        cv2.namedWindow('MaskNet', cv2.WINDOW_NORMAL)
        cv2.imshow('MaskNet', flipped)
        cv2.resizeWindow('MaskNet', 1000, 750)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def video_classifier():
    # tomamos el modelo guardado (proveniente del collab)
    model_path = "./saved_model/classifier"
    # cargamos el modelo
    model = tf.keras.models.load_model(model_path)
    # elegimos el detector de caras
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # elegimos el path del video que queremos seleccionar
    video_path = "diego2rapido.mp4"
    # video_path = "florblancorapido.mp4"

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        flipped = cv2.flip(frame, 1)
        # por cada frame del video aplicamos la funcion, obtenemos el frame con tags
        detect_faces_and_classify(flipped, face_cascade, model)

        # mostramos el frame
        cv2.namedWindow('MaskNet', cv2.WINDOW_NORMAL)
        cv2.imshow('MaskNet', flipped)
        cv2.resizeWindow('MaskNet', 1000, 750)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # face_classifier()
    # video_capture_classifier()
    video_classifier()
