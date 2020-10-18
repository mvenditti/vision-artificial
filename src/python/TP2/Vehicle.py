import numpy as np
import math
import random


class Vehicle:

    def __init__(self, x, y, speed, initial_frame=True):
        self.x = x
        self.y = y
        self.hist_x = 0
        self.hist_y = 0
        self.updated = True
        self.inactive_counter = 0
        self.speed = speed
        self.img = np.array([])
        self.color = generate_random_color()
        self.initial_frame = initial_frame

    def euclidean_distance(self, vehicle):
        return math.sqrt((self.x - vehicle.x)**2 + (self.y - vehicle.y)**2)


def generate_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return r, g, b
