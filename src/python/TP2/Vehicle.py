import numpy as np
import math
import random


class Vehicle:

    def __init__(self, x, y, speed):
        self.x = x
        self.y = y
        self.hist_x = 0
        self.hist_y = 0
        self.speed = speed
        self.img = np.array([])
        self.color = generate_random_color()
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

    return nearest
