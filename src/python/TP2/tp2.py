import cv2
import math

# from drawing_utils import mosaic_view
from src.python.TP2.Vehicle import Vehicle

video_path = 'race.mp4'
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
background_subs = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=25)

cap = cv2.VideoCapture(video_path)

while (cap.isOpened):
    ret, frame = cap.read()
    clean_frame = frame.copy()

    # OPERACIONES MORFOLOGICAS
    # apply background substraction
    detected_motion = background_subs.apply(frame)

    # marcar los autos mejor
    detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_CLOSE, kernel)  # Dilation followed by Erosion

    # sacar ruido del fondo
    detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation

    # buscamos contornos de parte de arriba y abajo por separado
    (contours, _) = cv2.findContours(detected_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def valid_countor(c):
        # boundingRect es para dibujar un rectangulo aprox al rededor de la img binaria
        (x, y, width, h) = cv2.boundingRect(c)
        return (cv2.contourArea(c) > 700 and width < 100)

    filtered_contours = list(filter(valid_countor, contours))
    # filtered_contours = contours

    for contour in filtered_contours:
        # get bounding box from countour
        (x, y, w, h) = cv2.boundingRect(contour)
        # draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('detected_motion', detected_motion)
    cv2.imshow('tp2', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
