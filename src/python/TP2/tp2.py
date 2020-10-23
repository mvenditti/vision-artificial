import cv2
import math

# cantidad de metros equivalentes a un pixel
from src.python.TP2.Vehicle import Vehicle, nearest_vehicle_in_range

meter_x_pixel = 0.026
# cantidad de segundos equivalentes a un frame
seconds_x_frame = 0.4

# cada cuantos frames realizo una accion.
frame_frequency = 10

# distancia maxima a recorrer a partir de dicho punto se remueven autos.
max_y_distance = 500


def get_center(contour):
    moments = cv2.moments(contour)
    centre_x = moments["m10"] / moments["m00"]
    centre_y = moments["m01"] / moments["m00"]
    return centre_x, centre_y


def frame_update(frame_counter):
    global frame_frequency
    return frame_counter % frame_frequency == 0


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def calculate_speed(car):
    global frame_frequency
    distance = calculate_distance(car.x, car.y, car.hist_x, car.hist_y)
    distance_meter = distance * meter_x_pixel
    meter_per_second = distance_meter / (seconds_x_frame * frame_frequency)
    return int(meter_per_second * 3.6)


def draw_contour(contour, frame, car):
    # get bounding box from countour
    (x, y, w, h) = cv2.boundingRect(contour)
    ROI = x, y, w, h
    # draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), car.color, 2)
    cv2.putText(frame, str(car.speed) + " km/h", (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)


def tp2():
    global max_y_distance

    # definimos para saber cada cuanto medir la distancia
    frame_counter = 0
    car_list = []

    video_path = 'traffic.mp4'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    background_subs = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=150)

    cap = cv2.VideoCapture(video_path)

    deleted = 0

    while (cap.isOpened):
        ret, frame = cap.read()
        clean_frame = frame.copy()
        cv2.line(frame, (0, max_y_distance), (10000, max_y_distance), (0, 0, 255), 3)
        frame_counter = frame_counter + 1

        # OPERACIONES MORFOLOGICAS
        # apply background substraction
        detected_motion = background_subs.apply(frame)

        # marcar los autos mejor
        detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_CLOSE, kernel)  # Dilation followed by Erosion

        # sacar ruido del fondo
        detected_motion = cv2.morphologyEx(detected_motion, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation

        # buscamos contornos de parte de arriba y abajo por separado
        contours, _ = cv2.findContours(detected_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        def valid_countor(c):
            # boundingRect es para dibujar un rectangulo aprox al rededor de la img binaria
            (x, y, width, h) = cv2.boundingRect(c)
            return cv2.contourArea(c) > 250 and width < 150

        filtered_contours = list(filter(valid_countor, contours))

        for contour in filtered_contours:
            center_x, center_y = get_center(contour)
            car_list.append(Vehicle(center_x, center_y, 0))
            for car in car_list:
                if car.y > max_y_distance:
                    car.remove = True
                else:
                    nearest = nearest_vehicle_in_range(car, car_list, 30)
                    if nearest is None:
                        car.remove = True
                    else:
                        car.x = nearest.x
                        car.y = nearest.y
                        draw_contour(contour, frame, car)

                        if frame_update(frame_counter):
                            if car.hist_x is not 0 and car.hist_y is not 0:
                                car.speed = calculate_speed(car)

                            car.hist_x = nearest.x
                            car.hist_y = nearest.y

        filtered = list(filter(lambda c: c.remove is True, car_list))
        deleted += len(filtered)
        car_list = filtered
        cv2.imshow('detected_motion', detected_motion)
        cv2.putText(frame, "Autos procesados: " + str(deleted), (20, 550), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        cv2.imshow('tp2', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tp2()
