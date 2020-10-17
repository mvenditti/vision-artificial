import cv2
import math


def tp2():
    # cantidad de metros equivalentes a un pixel
    meter_x_pixel = 0.026
    # cantidad de segundos equivalentes a un frame
    seconds_x_frame = 0.04

    # definimos para saber cada cuanto medir la distancia
    frame_counter = 0

    trackers = cv2.MultiTracker_create()

    video_path = 'traffic.mp4'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    background_subs = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=150)

    cap = cv2.VideoCapture(video_path)

    while (cap.isOpened):
        ret, frame = cap.read()
        clean_frame = frame.copy()
        frame_counter = frame_counter + 1

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
            return (cv2.contourArea(c) > 250 and width < 150)

        filtered_contours = list(filter(valid_countor, contours))
        # filtered_contours = contours

        for contour in filtered_contours:
            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(contour)
            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            moments = cv2.moments(contour)
            centre_x = moments["m10"]/moments["m00"]
            centre_y = moments["m01"]/moments["m00"]
            #
            # if ( frame_counter % 2 == 0 ):
            #     speed = calculateSpeed(centre_x, centre_y, )
            # else:


        cv2.imshow('detected_motion', detected_motion)
        cv2.imshow('tp2', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

tp2()