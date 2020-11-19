import numpy as np
import math
import random
import cv2


class Vehicle:

    def __init__(self, x, y, speed, id):
        self.id = id
        self.x = x
        self.y = y
        self.hist_x = 0
        self.hist_y = 0
        self.speed = speed
        self.img = np.array([])
        self.color = (255, 0, 0)
        self.remove = False

    # Distancia euclideana para comparar el centroide del vehiculo contra el de otro.
    def euclidean_distance(self, vehicle):
        return math.sqrt((self.x - vehicle.x) ** 2 + (self.y - vehicle.y) ** 2)


# Funciones de utilidad

# Genera un color RGB aleatorio.
def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return r, g, b


# Recibe listado de vehiculos a iterar, junto con la distancia maxima a modo de threshold.
# Devuelve el vehiculo mas cercano al pasado como argumento cumpliendo
# con la condición de encontrarse en un radio máximo
# En caso de que ningun vehiculo cumpla los requisitos, o que la lista de vehiculos este vacia, se devuelve None
# todo no devolveria none porque falla
def nearest_vehicle_in_range(vehicle, vehicles, max_distance):
    nearest = None
    if not vehicles:
        return nearest

    min_distance = vehicle.euclidean_distance(vehicles[0])
    for v in vehicles:
        distance = vehicle.euclidean_distance(v)
        if distance < max_distance and distance < min_distance:
            nearest = v
            min_distance = distance

    return nearest


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def get_center(contour):
    moments = cv2.moments(contour)
    centre_x = moments["m10"] / moments["m00"]
    centre_y = moments["m01"] / moments["m00"]
    return centre_x, centre_y


def nearest_contour_in_range(vehicle, contours, max_distance):
    nearest = None
    if not contours:
        return nearest

    (x1, y1) = get_center(contours[0])
    min_distance = calculate_distance(x1, y1, vehicle.x, vehicle.y)
    for c in contours:
        (x1, y1) = get_center(c)
        distance = calculate_distance(x1, y1, vehicle.x, vehicle.y)
        if distance < max_distance and distance < min_distance:
            nearest = c
            min_distance = distance

    return nearest


def nearest_vehicle_to_contour_in_range(contour, vehicles, max_distance):
    nearest = None
    index = 0
    if not vehicles:
        return nearest, index

    (x1, y1) = get_center(contour)
    vehicle = vehicles[0]
    min_distance = calculate_distance(x1, y1, vehicle.x, vehicle.y)
    for i, v in enumerate(vehicles):
        distance = calculate_distance(x1, y1, v.x, v.y)
        if distance < min_distance:
            nearest = v
            min_distance = distance
            index = i

    if min_distance < max_distance:
        return nearest, index
    else:
        return None, 0