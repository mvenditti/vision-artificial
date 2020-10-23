import cv2
import math

# cantidad de metros equivalentes a un pixel
from src.python.TP2.Vehicle import Vehicle, nearest_vehicle_in_range

meter_x_pixel = 0.026
# cantidad de segundos equivalentes a un frame
seconds_x_frame = 0.4


def contour_to_img(img, contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    crop_image = img[y:y + h, x:x + w]
    return crop_image


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_speed(car, center_x, center_y):
    distance = calculate_distance(center_x, center_y, car.hist_x, car.hist_y)
    distance_meter = distance * meter_x_pixel
    meter_per_second = distance_meter / seconds_x_frame
    return int(meter_per_second * 3.6)


def draw_contour(contour, frame, speed):
    # get bounding box from countour
    (x, y, w, h) = cv2.boundingRect(contour)
    ROI = x, y, w, h
    # draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(frame, str(speed) + " km/h", (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)


def tp2():
    # definimos para saber cada cuanto medir la distancia
    frame_counter = 0
    car_list = []
    check_speed = False

    video_path = 'video.mp4'
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

        if frame_counter == 5:
            check_speed = True
            frame_counter = 0

        for car in car_list:
            car.updated = False

        for contour in filtered_contours:
            moments = cv2.moments(contour)
            centre_x = moments["m10"] / moments["m00"]
            centre_y = moments["m01"] / moments["m00"]

            speed = 0
            contour_has_matched = False
            for car in car_list:
                nearest_car = nearest_vehicle_in_range(car, car_list, 30)
                distance = calculate_distance(car.x, car.y, nearest_car.x, nearest_car.y)

                if 30 > distance > 0 and not contour_has_matched and not car.updated:  # es cercano y los considero igual
                    contour_has_matched = True
                    car.updated = True
                    car.x = centre_x
                    car.y = centre_y

                    # con 2 quedaba tan seguido que ponia lento el video
                    if check_speed:
                        if car.hist_x != 0 and car.hist_y != 0:
                            speed = calculate_speed(car, centre_x, centre_y)
                            car.speed = speed
                            car_img = contour_to_img(clean_frame, contour)
                            height, width, channels = car_img.shape
                            if width > 55 and height < 70:
                                car.img = car_img
                        car.hist_x = centre_x
                        car.hist_y = centre_y
                    else:
                        speed = car.speed

            if not contour_has_matched:
                car_list.append(Vehicle(centre_x, centre_y, 0))

            draw_contour(contour, frame, speed)

        for car in car_list:
            if not car.updated:
                car.inactive_counter = car.inactive_counter + 1
            else:
                car.inactive_counter = 0

        car_list = list(filter(lambda c: c.inactive_counter < 5, car_list))

        cv2.imshow('detected_motion', detected_motion)
        cv2.imshow('tp2', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


tp2()
