import cv2
import math

# cantidad de metros equivalentes a un pixel
from src.python.TP2.Vehicle import Vehicle, nearest_vehicle_to_contour_in_range

# car dimensions in meter
car_length_meter = 3

# cantidad de segundos equivalentes a un frame
seconds_x_frame = 0.4

# cada cuantos frames realizo una accion.
frame_frequency = 3

# distancia maxima a recorrer a partir de dicho punto se remueven autos.
max_y_distance = 250


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


def calculate_speed(car, contour):
    global frame_frequency
    global car_length_meter
    (x, y, w, h) = get_bounding_rect(contour)
    dimension = None
    if w > h:
        dimension = w
    else:
        dimension = h
    meter_x_pixel = car_length_meter/dimension

    distance = calculate_distance(car.x, car.y, car.hist_x, car.hist_y)
    distance_meter = distance * meter_x_pixel

    meter_per_second = distance_meter / (seconds_x_frame * frame_frequency)
    return int(meter_per_second * 3.6)


def draw_contour(contour, frame, car):
    # get bounding box from countour
    (x, y, w, h) = get_bounding_rect(contour)
    # draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), car.color, 2)
    # cv2.putText(frame, str(car.speed) + " km/h", (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)


def in_bound(y):
    global max_y_distance
    return y > max_y_distance

def get_bounding_rect(contour):
    (x, y, w, h) = cv2.boundingRect(contour)
    return x, y, w, h


def tp2():
    global max_y_distance

    # definimos para saber cada cuanto medir la distancia
    frame_counter = 0
    car_list = []
    id_counter = 0

    video_path = 'video.mp4'
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
            nearest_car, index = nearest_vehicle_to_contour_in_range(contour, car_list, 100)
            car = None
            if nearest_car is None:
                # agregar como un auto nuevo, en vez de remover los autos creados innecesariamente
                car = Vehicle(center_x, center_y, 0, id_counter)
                car_list.append(car)
                id_counter += 1
            else:
                car = car_list[index]
                if not in_bound(car.y):
                    car.remove = True
                else:
                    # update existing car
                    car.x = center_x
                    car.y = center_y

                    if frame_update(frame_counter):
                        if car.hist_x != 0 and car.hist_y != 0:
                            speed = calculate_speed(car, contour)
                            car.speed = speed

                        car.hist_x = nearest_car.x
                        car.hist_y = nearest_car.y
                    car_list[index] = car
                    draw_contour(contour, frame, car)
                    cv2.putText(frame, str(car.id), (int(center_x), int(center_y)), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
                    # cv2.putText(frame, str(car.speed), (int(center_x), int(center_y) - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)

        before_filter = len(car_list)
        filtered = list(filter(lambda c: c.remove is False, car_list))
        filtered_amount = before_filter - len(filtered)
        deleted += filtered_amount
        car_list = filtered

        cv2.imshow('detected_motion', detected_motion)
        cv2.putText(frame, "Autos procesados: " + str(deleted), (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.75, (0, 0, 255), 1)
        cv2.imshow('tp2', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    tp2()

# str(int(deleted * 0.2))